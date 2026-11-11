"""SCOPE: the next read as an AMERICAN CALL OPTION ON THE VERIFIER'S BELIEF.

The underlying is the belief path p_0, ..., p_T of a question on a lattice of K belief cells.  Stopping at round tau
earns the call payoff (p_tau - theta)^+ : the excess of the belief over the gate theta that the answer certificate
will apply; every read costs the price kappa of the budget.  Three objects, nothing else:

  the law      ONE recurrent network with ONE head (the kernel field) reads the history the loop already produces and
               emits, for every question and every round t, a time-inhomogeneous Markov chain on the cells for the
               rounds to come,
                   Q_t[tau](k, .) = softmax( c(g_t, tau - t, k) ),        tau = t, ..., T-1.
               It is fitted by maximum likelihood on belief paths alone -- no label enters: the realised next cell
               under the kernel row of the realised cell (path likelihood) and the realised cell at t + j under the
               chain's j-step law from the current cell (predictive likelihood, the quantity the planner uses).
  the price    the option is priced exactly on that chain by backward induction, for EVERY strike of a grid and EVERY
               price of a grid in one pass; the policy re-plans at every round.  The value is non-increasing in the
               price, so the stopping round is non-increasing in the price on every trajectory: one nested deployment
               curve per strike, and the budget certificate fixes one index of it.
  the strike   the certificate-consistent fixed point theta = Gamma_C(sigma_theta^B), found by EXHAUSTIVE residual
               minimisation over the strike grid on the labelled slice C of the current split: no iteration, no
               initialisation, no state shared between splits.

Also here: the controls (population chain, kernel field written as a correction to the population kernel, one read
ahead, deterministic horizons, digital payoff), the label hindsight, the Lagrangian dual bound, the isotonic gate."""
import copy
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
import fpi

DEV = fpi.DEV
THETAS = np.round(np.arange(100) / 100.0, 2)        # the strike grid, resolution 0.01


# ------------------------------------------------------------------------------------------------- the law ---------
def _zero_last(m):
    """zero the output layer of an fpi.mlp so that the head starts as an exact zero correction."""
    nn.init.zeros_(m[-1].weight); nn.init.zeros_(m[-1].bias)
    return m


def kernel_anchor(b, K, smooth=0.5):
    """population kernel of a set of belief paths (cells b [n, T1]): logK [T, K, K], the log law of the cell at
    round t+1 given the cell at round t (add-1/2 smoothing).  Label-free."""
    T1 = b.shape[1]; T = T1 - 1
    logK = np.zeros((T, K, K), np.float32)
    for t in range(T):
        for k in range(K):
            cnt = np.bincount(b[b[:, t] == k, t + 1], minlength=K) + smooth
            logK[t, k] = np.log(cnt / cnt.sum())
    return logK


class PathNet(nn.Module):
    """Causal encoder over rounds and one head: the kernel field.  For the state g_t of round t, the horizon
    j >= 1 and a cell k, the head returns the logits of the kernel row of the step into round t + j, added to the
    buffer logK [T, K, K]: zeros for the method (the field is the kernel), the log population kernel for the control
    that writes the field as a correction to it (anchors used in training are then cross-fitted)."""

    def __init__(self, dh, dx, K, T, logK, edges, H=128, D=32):
        super().__init__()
        self.enc = fpi.Enc(dh, dx + 2, H)
        self.hor, self.cel = nn.Embedding(T + 1, D), nn.Embedding(K, D)
        self.kh = _zero_last(fpi.mlp(H + 2 * D, H, K))
        self.register_buffer('logK', torch.as_tensor(np.asarray(logK), dtype=torch.float32))      # [T, K, K]
        self.register_buffer('edges', torch.as_tensor(np.asarray(edges), dtype=torch.float32))    # [K-1]
        jk = torch.arange(1, T + 1); tt = torch.arange(T + 1)
        self.register_buffer('step', torch.clamp(tt[:, None] + jk[None, :] - 1, 0, T - 1))         # [T1, T] step of (t, j)
        self.K, self.T = K, T

    def trunk(self, h, x, p):
        lp = torch.logit(p.clamp(1e-4, 1 - 1e-4)) / 4
        return self.enc(h, torch.cat([x, p[..., None], lp[..., None]], -1))          # [B, T1, H]

    def field(self, g, base=None):
        """log kernels [n, T1, T, K, K]: entry (t, j-1) is the law of the cell at round t + j given the cell at
        round t + j - 1 and the history up to round t.  base: per-question anchors [n, T, K, K] (training)."""
        n, T1, _ = g.shape; T, K = self.T, self.K
        anchor = self.logK[self.step][None] if base is None else base[:, self.step]             # [1|n, T1, T, K, K]
        j = torch.arange(1, T + 1, device=g.device); k = torch.arange(K, device=g.device)
        z = torch.cat([g[:, :, None, None, :].expand(n, T1, T, K, -1),
                       self.hor(j)[None, None, :, None, :].expand(n, T1, T, K, -1),
                       self.cel(k)[None, None, None, :, :].expand(n, T1, T, K, -1)], -1)
        return torch.log_softmax(anchor + self.kh(z), -1)

    def binof(self, p):
        return torch.searchsorted(self.edges, p.contiguous(), right=True)


def _early_stop(step, val_loss, params_state, epochs, patience, rng, ti):
    """generic loop: step(perm) runs one epoch over the permutation, val_loss() the validation loss; keeps the best
    state (the initial one included)."""
    best, state, bad = val_loss(), params_state(), 0
    for ep in range(epochs):
        step(ti[rng.permutation(len(ti))])
        vl = val_loss()
        if vl < best - 1e-4:
            best, state, bad = vl, params_state(), 0
        else:
            bad += 1
            if bad >= patience:
                break
    return state


def chain_nll(logQ, b):
    """(path, predictive) negative log-likelihood of realised cells b [n, T1] under the chains logQ [n, T1, T, K, K]:
    path: the realised next cell under the kernel row of the realised cell, for every round t and every later step;
    predictive: the realised cell at t + j under the j-step law of the chain started at the current cell."""
    n, T1 = b.shape; K = logQ.shape[-1]; ar = torch.arange(n, device=b.device)
    tf, fr = [], []
    for t in range(T1 - 1):
        m = F.one_hot(b[:, t], K).to(logQ.dtype)
        for j in range(1, T1 - t):
            Q = logQ[:, t, j - 1]
            tf.append(-Q[ar, b[:, t + j - 1], b[:, t + j]].mean())
            m = torch.einsum('nk,nkl->nl', m, Q.exp())
            fr.append(-torch.log(m[ar, b[:, t + j]] + 1e-9).mean())
    return torch.stack(tf).mean(), torch.stack(fr).mean()


def fit_path(h, x, p, lat, seed=0, epochs=80, lr=1e-3, wd=1e-2, bs=128, val=0.15, patience=10, dev=DEV, H=128, folds=5,
             anchor=False, predictive=True):
    """Fit the kernel field on belief paths p [N, T1] (out-of-fold beliefs of the verifier) and the observations
    (h, x) of the same rounds.  No labels.  anchor=True: the field as a zero-initialised correction to the population
    kernel (control); predictive=False: path likelihood only (control)."""
    rng = np.random.RandomState(seed); torch.manual_seed(seed)
    N, T1 = p.shape; T = T1 - 1; K = lat.K
    b = lat.binof(p)
    ti, vi = fpi._split(N, val, rng); VI = torch.as_tensor(vi, device=dev)
    LK = np.zeros((N, T, K, K), np.float32)
    if anchor:
        logK_all = kernel_anchor(b, K)
        LK[vi] = kernel_anchor(b[ti], K)
        fo = rng.permutation(len(ti)) % folds
        for m in range(folds):
            LK[ti[fo == m]] = kernel_anchor(b[ti[fo != m]], K)
    else:
        logK_all = np.zeros((T, K, K), np.float32)
    net = PathNet(0 if h is None else h.shape[-1], x.shape[-1], K, T, logK_all, lat.edges, H).to(dev)
    t_ = lambda a: None if a is None else torch.as_tensor(a, dtype=torch.float32, device=dev)
    Ht, Xt, Pt, LKt = t_(h), t_(x), t_(p), t_(LK)
    Bt = torch.as_tensor(b, device=dev)
    sub = lambda A, i: None if A is None else A[i]

    def loss_on(idx):
        tf, fr = chain_nll(net.field(net.trunk(sub(Ht, idx), Xt[idx], Pt[idx]), LKt[idx]), Bt[idx])
        return tf + fr if predictive else tf

    opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=wd)

    def step(perm):
        net.train()
        for k in range(0, len(perm), bs):
            loss = loss_on(torch.as_tensor(perm[k:k + bs], device=dev))
            opt.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(net.parameters(), 5.0); opt.step()

    @torch.no_grad()
    def val_loss():
        net.eval()
        return float(np.mean([float(loss_on(VI[i:i + 512])) for i in range(0, len(VI), 512)]))
    net.load_state_dict(_early_stop(step, val_loss, lambda: copy.deepcopy(net.state_dict()), epochs, patience, rng, ti))
    net.eval()
    return net


@torch.no_grad()
def predict_path(net, h, x, p, dev=DEV, bs=256):
    """P [N, T1, T, K, K]: P[:, t, j-1] = kernel of the step into round t + j as seen from round t (entries with
    t + j > T are unused)."""
    N, T1 = p.shape; T = T1 - 1; K = net.K
    out = np.zeros((N, T1, T, K, K), np.float32)
    for i0 in range(0, N, bs):
        t_ = lambda a: None if a is None else torch.as_tensor(a[i0:i0 + bs], dtype=torch.float32, device=dev)
        pb = t_(p)
        out[i0:i0 + len(pb)] = net.field(net.trunk(t_(h), t_(x), pb)).exp().cpu().numpy()
    return out


class CrossPath:
    """K-fold cross-fitted kernel fields on the verifier's folds: oof(...) on training questions, the fold average
    elsewhere (an average of kernels is a kernel)."""

    def __init__(self, h, x, p_oof, lat, fold, seed=0, **kw):
        self.fold, self.nets = fold, []
        for k in range(fold.max() + 1):
            tr = fold != k
            self.nets.append(fit_path(None if h is None else h[tr], x[tr], p_oof[tr], lat, seed=5000 * seed + 19 * k, **kw))

    def oof(self, h, x, p):
        P = None
        for k, n in enumerate(self.nets):
            m = self.fold == k
            Pk = predict_path(n, None if h is None else h[m], x[m], p[m])
            if P is None:
                P = np.zeros((len(x),) + Pk.shape[1:], np.float32)
            P[m] = Pk
        return P

    def __call__(self, h, x, p):
        P = None
        for n in self.nets:
            Pk = predict_path(n, h, x, p)
            P = Pk if P is None else P + Pk
        return P / len(self.nets)


def pop_chain(lat, N):
    """Population chain: the kernels K_t of the lattice, identical for every question (control)."""
    T1, T, K = lat.T1, lat.T, lat.K
    P = np.zeros((T1, T, K, K), np.float32)
    for t in range(T1):
        for j in range(1, T1 - t):
            P[t, j - 1] = lat.Kt[t + j - 1]
    return np.repeat(P[None], N, 0)


def path_nll(P, b):
    """Diagnostics on a pool: (path, predictive) NLL of the realised cells b [N, T1] under the chains P."""
    N, T1 = b.shape; K = P.shape[-1]; ar = np.arange(N)
    tf, fr = [], []
    for t in range(T1 - 1):
        m = np.zeros((N, K), np.float64); m[ar, b[:, t]] = 1
        for j in range(1, T1 - t):
            tf.append(-np.log(P[ar, t, j - 1, b[:, t + j - 1], b[:, t + j]] + 1e-9).mean())
            m = np.einsum('nk,nkl->nl', m, P[:, t, j - 1])
            fr.append(-np.log(m[ar, b[:, t + j]] + 1e-9).mean())
    return float(np.mean(tf)), float(np.mean(fr))


# ------------------------------------------------------------------------------------------------- payoffs ---------
def call_table(p_tr, lat, thetas=THETAS, cells=None):
    """call[theta, k] = E[(p - theta)^+ | cell k] on the out-of-fold training beliefs: the stop payoff of a future
    cell (cells: a given cell assignment, for monotone transforms of the belief that keep the quantile cells)."""
    flat = p_tr.reshape(-1); bf = lat.binof(flat) if cells is None else cells.reshape(-1); th = np.asarray(thetas)
    out = np.zeros((len(th), lat.K), np.float32)
    for k in range(lat.K):
        v = flat[bf == k]
        out[:, k] = np.maximum(v[None, :] - th[:, None], 0).mean(1) if len(v) else np.maximum(lat.mid[k] - th, 0)
    return out


def frac_table(p_tr, lat, thetas=THETAS, cells=None):
    """frac[theta, k] = P(p >= theta | cell k) on the out-of-fold training beliefs: the digital payoff (control)."""
    flat = p_tr.reshape(-1); bf = lat.binof(flat) if cells is None else cells.reshape(-1)
    out = np.zeros((len(thetas), lat.K), np.float32)
    for k in range(lat.K):
        m = bf == k
        out[:, k] = np.mean(flat[m][None, :] >= np.asarray(thetas)[:, None], 1) if m.any() else (lat.mid[k] >= np.asarray(thetas))
    return out


def _now(pb, th, payoff):
    """stop payoff of the current round, where the belief is known: [n, Theta]."""
    return torch.clamp(pb[:, None] - th[None], min=0) if payoff == 'call' else (pb[:, None] >= th[None]).float()


# ------------------------------------------------------------------------------------------------- the planner ------
def price_grid(dc, J=200):
    """Finite price grid (payoff per unit of resource) up to the largest price that can matter, 1 / min cost."""
    hi = 1.0 / max(float(np.min(dc)), 1e-3)
    return (np.geomspace(1e-4, hi, J) - 1e-4 + 1e-7).astype(np.float64)


@torch.no_grad()
def plan(P, b, p, U, thetas, kap, dc, last=None, myopic=False, payoff='call', dev=DEV, bs=96):
    """Deployment curves S [Theta, N, J+3] (int8): stopping round of every question at every price of the ascending
    grid kap, for every strike of thetas.  Column 0 never stops early (price -inf), columns 1..J the grid, column J+1
    stops at once, column J+2 stops at once and abstains.
    P [N, T1, T, K, K] chains, b [N, T1] cells, p [N, T1] beliefs, U [Theta, K] stop payoff of a future cell
    (call_table, or frac_table with payoff='digital'), dc[tau] = resource of the read from round tau to tau + 1 (budget
    units), last = forced stopping round, myopic = value the next round by its stop payoff only (control)."""
    N, T1 = p.shape; T = T1 - 1; last = T if last is None else int(last)
    K = U.shape[1]; Th = len(thetas); J = len(kap)
    th = torch.as_tensor(np.asarray(thetas), dtype=torch.float32, device=dev)
    Ut = torch.as_tensor(U, dtype=torch.float32, device=dev)
    ka = torch.as_tensor(np.asarray(kap), dtype=torch.float32, device=dev)
    dct = torch.as_tensor(np.asarray(dc), dtype=torch.float32, device=dev)
    out = np.zeros((Th, N, J + 3), np.int8); out[:, :, 0] = last
    for i0 in range(0, N, bs):
        Pb = torch.as_tensor(P[i0:i0 + bs], dtype=torch.float32, device=dev)
        bb = torch.as_tensor(b[i0:i0 + bs], dtype=torch.long, device=dev)
        pb = torch.as_tensor(p[i0:i0 + bs], dtype=torch.float32, device=dev)
        n = pb.shape[0]
        S = torch.full((Th, n, J), last, dtype=torch.int16, device=dev)
        U4 = Ut[None, :, None, :].expand(n, Th, J, K)
        for t in range(last - 1, -1, -1):
            V = U4                                                                                          # value at t+1 (myopic) / at last
            if not myopic:
                for tau in range(last - 1, t, -1):
                    cont = -ka[None, None, :, None] * dct[tau] + torch.einsum('nkl,nojl->nojk', Pb[:, t, tau - t], V)
                    V = torch.maximum(U4, cont)
            P0 = Pb[:, t, 0].gather(1, bb[:, t, None, None].expand(n, 1, K)).squeeze(1)                      # [n, K]
            cont_t = -ka[None, None, :] * dct[t] + torch.einsum('nl,nojl->noj', P0, V)                        # [n, Th, J]
            read = (cont_t > _now(pb[:, t], th, payoff)[:, :, None]).permute(1, 0, 2)                         # [Th, n, J]
            S = torch.where(read, S, torch.full_like(S, t))
        out[:, i0:i0 + n, 1:J + 1] = S.cpu().numpy().astype(np.int8)
    return out


@torch.no_grad()
def plan_dh(P, b, p, U, thetas, kap, cost, last=None, payoff='call', dev=DEV, bs=256):
    """Deterministic-horizon price on the SAME chain (control: no branching): law of the cell at every future round
    from the kernels, best forecast payoff gain per unit of resource over fixed continuation lengths, first crossing
    of the price grid.  cost [N, T1] resource when stopping at a round."""
    N, T1 = p.shape; T = T1 - 1; last = T if last is None else int(last)
    K = U.shape[1]; Th = len(thetas)
    th = torch.as_tensor(np.asarray(thetas), dtype=torch.float32, device=dev)
    Ut = torch.as_tensor(U, dtype=torch.float32, device=dev)
    rho = np.full((Th, N, last), -np.inf, np.float32)
    for i0 in range(0, N, bs):
        Pb = torch.as_tensor(P[i0:i0 + bs], dtype=torch.float32, device=dev)
        bb = torch.as_tensor(b[i0:i0 + bs], dtype=torch.long, device=dev)
        pb = torch.as_tensor(p[i0:i0 + bs], dtype=torch.float32, device=dev)
        cb = torch.as_tensor(cost[i0:i0 + bs], dtype=torch.float32, device=dev)
        n = pb.shape[0]
        for t in range(last):
            u_t = _now(pb[:, t], th, payoff)                                                               # [n, Th]
            m = F.one_hot(bb[:, t], K).float()
            best = torch.full((n, Th), -np.inf, device=dev)
            for tau in range(t + 1, last + 1):
                m = torch.einsum('nk,nkl->nl', m, Pb[:, t, tau - t - 1])                                   # law at tau
                best = torch.maximum(best, (m @ Ut.T - u_t) / (cb[:, tau] - cb[:, t]).clamp(min=1e-9)[:, None])
            rho[:, i0:i0 + n, t] = best.T.cpu().numpy()
    out = np.zeros((Th, N, len(kap) + 3), np.int8)
    for o in range(Th):
        out[o] = fpi.index_path(rho[o], np.concatenate([[-np.inf], kap, [np.inf]]), last).astype(np.int8)
    return out


def nested_check(S):
    return float(np.mean(np.all(np.diff(S.astype(np.int16), axis=1) <= 0, 1)))


# ------------------------------------------------------------------------------------------------- the strike -------
def gamma_iso(p_stop, y_stop, target, n_pool, free=False):
    """Isotonic plug-in gate of the certificate on a stopped sample (sklearn PAVA).  Batch target: smallest belief of
    the longest prefix (beliefs sorted downwards) whose isotonic cumulative precision is >= min(1-q, highest reached);
    risk target: longest prefix with cumulative smoothed wrong count / n_pool <= alpha.  Returns (gate, reached)."""
    from sklearn.isotonic import isotonic_regression
    fin = np.isfinite(p_stop); s = p_stop[fin]
    if len(s) == 0:
        return 1.0, False
    o = np.argsort(-s, kind='mergesort'); s = s[o]
    yy = s if free else y_stop[fin][o]
    m = isotonic_regression(yy.astype(np.float64), increasing=False)
    k = np.arange(1, len(s) + 1)
    if target[0] == 'cs':
        prec = np.cumsum(m) / k
        goal = min(1 - target[1], float(prec[0]))
        ok = np.where(prec >= goal - 1e-12)[0]
        return float(s[ok[-1]]), bool(prec[0] >= 1 - target[1] - 1e-12)
    wrong = np.cumsum(1 - m) / n_pool
    ok = np.where(wrong <= target[1] + 1e-12)[0]
    return (float(s[ok[-1]]) if len(ok) else float(s[0])), bool(len(ok) > 0)


class Family:
    """A strike-indexed family of deployment curves SS [Theta, N, J'] over a cost table cost [N, T1]; cost paths and
    their monotone envelopes are computed on demand and cached for the certified members."""

    def __init__(self, SS, cost, thetas=THETAS):
        self.SS, self.cost, self.thetas = SS, cost, np.asarray(thetas)
        self._cache = {}

    def idx(self, theta):
        return int(np.argmin(np.abs(self.thetas - theta)))

    def prep(self, o):
        """(S, cost path, cost envelope) of curve o."""
        import cert14 as C
        if o not in self._cache:
            cst = fpi.path_cost(self.SS[o], self.cost).astype(np.float32)
            self._cache[o] = (self.SS[o], cst, C.env_cols(cst))
            if len(self._cache) > 40:
                self._cache.pop(next(iter(self._cache)))
        return self._cache[o]

    def stops_at_budget(self, o, idx, B):
        """member of curve o whose mean cost on the questions idx is closest to B (no dithering: the gate only)."""
        S = self.SS[o][idx]
        m = fpi.path_cost(S, self.cost[idx]).mean(0)
        return S[:, int(np.argmin(np.abs(m[:-1] - B)))]

    def gates(self, p, yy, cost, B, target, idx, free=False):
        """Gamma_C(sigma_theta^B) for every strike of the grid: (gates [Theta], reached [Theta])."""
        g = np.zeros(len(self.thetas)); r = np.zeros(len(self.thetas), bool)
        for o in range(len(self.thetas)):
            st = self.stops_at_budget(o, idx, B)
            g[o], r[o] = gamma_iso(p[idx, st], yy[idx, st], target, len(idx), free=free)
        return g, r

    def solve(self, p, yy, cost, B, target, idx, free=False):
        """Exhaustive certificate-consistent strike on the slice idx: the grid strike with the smallest residual
        |theta - Gamma_C(sigma_theta^B)| (ties: the largest).  Returns (o, theta, residual, reached, gates)."""
        g, r = self.gates(p, yy, cost, B, target, idx, free=free)
        res = np.abs(self.thetas - g)
        o = int(np.flatnonzero(res <= res.min() + 1e-12)[-1])
        return o, float(self.thetas[o]), float(res[o]), bool(r[o]), g

    def km(self, gates, init, iters=8, damp=0.5, tol=0.005):
        """Krasnoselskii-Mann iteration on the SAME gate map (diagnostic): theta <- (1-damp) theta + damp Gamma(theta)
        with Gamma read off the grid; returns the final strike and the number of iterations."""
        th = float(init)
        for it in range(iters):
            new = float(np.clip((1 - damp) * th + damp * gates[self.idx(th)], 0.0, 1.0))
            done = abs(new - th) < tol
            th = new
            if done:
                break
        return th, it + 1


# ------------------------------------------------------------------------------------------------- hindsight --------
def hindsight(p, y, cost, theta, kap, last):
    """Label hindsight: stop where the realised Lagrangian payoff 1{p >= theta}(y - theta) - kappa c is largest,
    nested in kappa (an upper bound on the reward of every adapted rule at the same strike and price)."""
    N = len(p)
    R = (p[:, : last + 1] >= theta) * (y[:, : last + 1] - theta)
    U = R[:, None, :] - np.asarray(kap)[None, :, None] * cost[:, None, : last + 1]
    S = np.zeros((N, len(kap) + 3), np.int16); S[:, 0] = last; S[:, 1:len(kap) + 1] = U.argmax(-1)
    S[:, :len(kap) + 2] = np.minimum.accumulate(S[:, :len(kap) + 2], 1)
    return S


def reward(p, y, st, theta):
    """Realised Lagrangian reward of stopping at st (per question)."""
    ar = np.arange(len(p))
    return (p[ar, st] >= theta) * (y[ar, st] - theta)


def dual_bound(p, y, cost, theta, B, kap, last):
    """min over kappa >= 0 of E[max_t (R_t - kappa c_t)] + kappa B on the given questions: the Lagrangian bound on
    E[R_sigma] over every adapted stopping rule with E[c_sigma] <= B."""
    R = (p[:, : last + 1] >= theta) * (y[:, : last + 1] - theta)
    ks = np.concatenate([[0.0], np.asarray(kap)])
    vals = [(R - k * cost[:, : last + 1]).max(1).mean() + k * B for k in ks]
    return float(np.min(vals))

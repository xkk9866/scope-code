"""Round 16: SCOPE as a MODEL-BASED planner on the reader's belief lattice.

The verifier belief p_t (cross-fitted) is the state of a finite-horizon Markov decision process:
    cells  : K cells of [0,1] (quantiles of out-of-fold training beliefs)
    acc_t  : calibration table, accuracy of the stopped answer in cell b at round t          [T+1, K]
    K_t    : population law of the belief under one more read, cell b -> cell b'           [T, K, K]
    Q_i,t  : the ONE-STEP model: a network reads the reader's state history and predicts the law of the NEXT
             belief cell for THIS question if it reads once more                              [N, T, K]
Reading costs kappa per unit resource; stopping in cell b at round t earns 1{p >= g}(acc_t[b] - theta): an answer
above the admission gate g earns its calibrated accuracy minus the strike theta (the Lagrangian of the certified
objective).  Backward induction on the lattice gives, for every price kappa, the value V_t(b) of every cell; the
question-specific one-step law prices the NEXT read, and the population law governs the reads after it.  The
planner stops at the first round where stopping beats reading.  Because V is non-increasing in kappa, the stopping
round is non-increasing in kappa on every trajectory: the price grid yields a nested deployment curve and the
certificates of cert14 apply unchanged.  One fitted model serves every reliability target, budget, horizon and
cost unit; only the dynamic programme is re-run.

Gate and strike are the two multipliers of the certified objective and are set on TRAINING questions only
(select_gate): the gate from a coarse grid by certified training coverage, the strike = calibrated accuracy at the
gate (the marginal answer the certificate admits is worth zero).
"""
import copy
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
import fpi

DEV = fpi.DEV
GATES = (0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.95)


# ----------------------------------------------------------------------------------------------- lattice ----------
class Lattice:
    def __init__(self, p_tr, y_tr, K=20, prior=2.0, smooth=0.5):
        T1 = p_tr.shape[1]; self.T1, self.T, self.K = T1, T1 - 1, K
        flat = p_tr.reshape(-1)
        self.edges = np.quantile(flat, np.linspace(0, 1, K + 1)[1:-1])
        b = self.binof(p_tr); bf = self.binof(flat)
        self.mid = np.array([flat[bf == k].mean() if (bf == k).any() else (k + .5) / K for k in range(K)])
        self.acc = np.zeros((T1, K)); self.Kt = np.zeros((self.T, K, K))
        for t in range(T1):
            for k in range(K):
                m = b[:, t] == k
                self.acc[t, k] = (y_tr[m, t].sum() + prior * self.mid[k]) / (m.sum() + prior)
                if t < self.T:
                    cnt = np.bincount(b[m, t + 1], minlength=K) + smooth
                    self.Kt[t, k] = cnt / cnt.sum()

    def binof(self, p):
        return np.searchsorted(self.edges, p, side='right')

    def pop_onestep(self, b):
        """population one-step law for cells b [N, T1] -> Q [N, T, K] (the 'population' ablation)."""
        return np.stack([self.Kt[t][b[:, t]] for t in range(self.T)], 1).astype(np.float32)

    def theta_at(self, g):
        """strike = calibrated accuracy of the gate cell (mean over rounds >= 1)."""
        k = int(self.binof(np.array([g]))[0])
        return float(np.mean(self.acc[1:, k])) if self.T1 > 1 else float(self.acc[0, k])


# ----------------------------------------------------------------------------------------------- one-step model ---
class OneStep(nn.Module):
    """Causal encoder over rounds; at round t predicts the law of the belief cell after one more read."""

    def __init__(self, dh, dx, K, H=128, dpeek=0):
        super().__init__()
        self.enc = fpi.Enc(dh, dx + 2, H)
        self.head = fpi.mlp(H + 1 + dpeek, H, K)
        self.K, self.dpeek = K, dpeek

    def forward(self, h, x, p, rem, peek=None):
        lp = torch.logit(p.clamp(1e-4, 1 - 1e-4)) / 4
        g = self.enc(h, torch.cat([x, p[..., None], lp[..., None]], -1))
        f = [g, rem[..., None]]
        if self.dpeek:
            f.append(peek)
        return self.head(torch.cat(f, -1))                               # [B, T1, K] logits


def fit_onestep(h, x, p, y, lat, peek=None, seed=0, epochs=80, lr=1e-3, wd=1e-2, bs=128, val=0.15, patience=10, dev=DEV, H=128):
    rng = np.random.RandomState(seed); torch.manual_seed(seed)
    N, T1 = y.shape; T = T1 - 1; K = lat.K
    b = lat.binof(p)
    tgt = np.zeros((N, T1), np.int64); tgt[:, :T] = b[:, 1:]
    net = OneStep(0 if h is None else h.shape[-1], x.shape[-1], K, H, 0 if peek is None else peek.shape[-1]).to(dev)
    opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=wd)
    t_ = lambda a: None if a is None else torch.as_tensor(a, dtype=torch.float32, device=dev)
    Ht, Xt, Pt, Pk = t_(h), t_(x), t_(p), t_(peek)
    Yt = torch.as_tensor(tgt, device=dev)
    rem = (1 - torch.arange(T1, device=dev, dtype=torch.float32) / max(T, 1)).clamp(min=0)[None].expand(N, -1)
    sub = lambda A, i: None if A is None else A[i]
    ti, vi = fpi._split(N, val, rng)

    def loss_on(idx):
        o = net(sub(Ht, idx), Xt[idx], Pt[idx], rem[idx], sub(Pk, idx))[:, :T]
        return F.cross_entropy(o.reshape(-1, K), Yt[idx][:, :T].reshape(-1))
    best, state, bad = np.inf, None, 0
    for ep in range(epochs):
        net.train()
        perm = ti[rng.permutation(len(ti))]
        for k in range(0, len(perm), bs):
            bi = torch.as_tensor(perm[k:k + bs], device=dev)
            loss = loss_on(bi)
            opt.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(net.parameters(), 5.0); opt.step()
        net.eval()
        with torch.no_grad():
            vl = float(loss_on(torch.as_tensor(vi, device=dev)))
        if vl < best - 1e-4:
            best, state, bad = vl, copy.deepcopy(net.state_dict()), 0
        else:
            bad += 1
            if bad >= patience:
                break
    net.load_state_dict(state); net.eval()
    return net


@torch.no_grad()
def predict_onestep(net, h, x, p, peek=None, dev=DEV, bs=1024):
    out = []
    N, T1 = p.shape
    rem_all = (1 - np.arange(T1, dtype=np.float32) / max(T1 - 1, 1)).clip(min=0)
    for k in range(0, N, bs):
        t_ = lambda a: None if a is None else torch.as_tensor(a[k:k + bs], dtype=torch.float32, device=dev)
        rem = torch.as_tensor(np.repeat(rem_all[None], min(bs, N - k), 0), device=dev)
        o = net(t_(h), t_(x), t_(p), rem, t_(peek))
        out.append(torch.softmax(o, -1).cpu().numpy())
    return np.concatenate(out)[:, :T1 - 1].astype(np.float32)        # [N, T, K]


class CrossOneStep:
    """K-fold cross-fitted one-step models on the verifier's folds: oof(...) on training questions, fold average
    elsewhere (an average of laws is a law; the planner stays nested)."""

    def __init__(self, h, x, p_oof, y, lat, fold, peek=None, seed=0, **kw):
        self.fold, self.nets = fold, []
        for k in range(fold.max() + 1):
            tr = fold != k
            self.nets.append(fit_onestep(None if h is None else h[tr], x[tr], p_oof[tr], y[tr], lat,
                                         None if peek is None else peek[tr], seed=3000 * seed + 11 * k, **kw))

    def oof(self, h, x, p, peek=None):
        out = None
        for k, n in enumerate(self.nets):
            m = self.fold == k
            o = predict_onestep(n, None if h is None else h[m], x[m], p[m], None if peek is None else peek[m])
            if out is None:
                out = np.zeros((len(x),) + o.shape[1:], np.float32)
            out[m] = o
        return out

    def __call__(self, h, x, p, peek=None):
        return np.mean([predict_onestep(n, h, x, p, peek) for n in self.nets], 0)


# ----------------------------------------------------------------------------------------------- the planner ------
def price_grid(J=200, hi=1.0):
    return np.concatenate([[-np.inf], np.geomspace(1e-4, hi, J) - 1e-4 + 1e-7, [np.inf]])


def plan(lat, Q, b, p, g, th, grid, dc, last=None, myopic=False):
    """Deployment curve S [N, J+3]: stopping round of every question at every price of the ascending grid
    (column 0: never stop early; column J+1: stop at once; appended column J+2: stop at once and abstain).
    Q [N, T, K] one-step law at each decision round, b [N, T1] cells, p [N, T1] beliefs, dc [T] resource cost of
    round t+1 in budget units, last: forced stopping round (the horizon or a deadline), myopic: value the next
    cell by its stop payoff only."""
    K = lat.K
    last = lat.T if last is None else int(last)
    N = len(b)
    stopv = (lat.mid[None, :] >= g) * (lat.acc[: last + 1] - th)                                   # [last+1, K]
    stop_i = (p[:, : last + 1] >= g) * (lat.acc[np.arange(last + 1)[None, :], b[:, : last + 1]] - th)  # [N, last+1]
    kap = grid[np.isfinite(grid)]; J = len(kap)
    V = np.zeros((J, last + 1, K)); V[:, last] = stopv[last][None]
    for t in range(last - 1, -1, -1):
        cont = -kap[:, None] * dc[t] + np.einsum('bk,jk->jb', lat.Kt[t], V[:, t + 1])
        V[:, t] = np.maximum(stopv[t][None], cont)
    Vnext = np.repeat(stopv[1: last + 1][None], J, 0) if myopic else V[:, 1: last + 1]            # [J, last, K]
    cont_i = np.einsum('ntk,jtk->ntj', Q[:, :last], Vnext) - kap[None, None, :] * np.asarray(dc[:last], float)[None, :, None]
    read = cont_i > stop_i[:, :last, None]                                                           # [N, last, J]
    S = np.full((N, J), last, np.int16)
    for t in range(last - 1, -1, -1):
        S = np.where(read[:, t], S, t).astype(np.int16)
    out = np.zeros((N, J + 3), np.int16)
    out[:, 0] = last
    out[:, 1:J + 1] = S
    return out


def nested_check(S):
    return float(np.mean(np.all(np.diff(S, axis=1) <= 0, 1)))


# ----------------------------------------------------------------------------------------------- multi-horizon ----
class Forecast(nn.Module):
    """Causal encoder over rounds; at round t forecasts the belief cell at EVERY future round tau > t
    (a term structure of belief forecasts).  Output logits [B, T1, T, K]: entry (t, tau-1) is the law of the cell
    at round tau; entries with tau <= t are ignored."""

    def __init__(self, dh, dx, K, T, H=128):
        super().__init__()
        self.enc = fpi.Enc(dh, dx + 2, H)
        self.head = fpi.mlp(H + 1, H, T * K)
        self.K, self.T = K, T

    def forward(self, h, x, p, rem):
        lp = torch.logit(p.clamp(1e-4, 1 - 1e-4)) / 4
        g = self.enc(h, torch.cat([x, p[..., None], lp[..., None]], -1))
        o = self.head(torch.cat([g, rem[..., None]], -1))
        return o.reshape(o.shape[0], o.shape[1], self.T, self.K)


def fit_forecast(h, x, p, y, lat, seed=0, epochs=80, lr=1e-3, wd=1e-2, bs=128, val=0.15, patience=10, dev=DEV, H=128):
    rng = np.random.RandomState(seed); torch.manual_seed(seed)
    N, T1 = y.shape; T = T1 - 1; K = lat.K
    b = lat.binof(p)
    tgt = np.repeat(b[:, None, 1:], T1, 1)                        # [N, T1(t), T(tau-1)] cell at round tau
    valid = np.zeros((T1, T), np.float32)
    for t in range(T1):
        valid[t, t:] = 1.0                                        # tau - 1 >= t  <=>  tau > t
    net = Forecast(0 if h is None else h.shape[-1], x.shape[-1], K, T, H).to(dev)
    opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=wd)
    t_ = lambda a: None if a is None else torch.as_tensor(a, dtype=torch.float32, device=dev)
    Ht, Xt, Pt = t_(h), t_(x), t_(p)
    Yt = torch.as_tensor(tgt, device=dev); Mt = torch.as_tensor(valid, device=dev)
    rem = (1 - torch.arange(T1, device=dev, dtype=torch.float32) / max(T, 1)).clamp(min=0)[None].expand(N, -1)
    sub = lambda A, i: None if A is None else A[i]
    ti, vi = fpi._split(N, val, rng)

    def loss_on(idx):
        o = net(sub(Ht, idx), Xt[idx], Pt[idx], rem[idx])                       # [b, T1, T, K]
        ce = F.cross_entropy(o.reshape(-1, K), Yt[idx].reshape(-1), reduction='none').reshape(len(idx), T1, T)
        return (ce * Mt[None]).sum() / (Mt.sum() * len(idx))
    best, state, bad = np.inf, None, 0
    for ep in range(epochs):
        net.train()
        perm = ti[rng.permutation(len(ti))]
        for k in range(0, len(perm), bs):
            bi = torch.as_tensor(perm[k:k + bs], device=dev)
            loss = loss_on(bi)
            opt.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(net.parameters(), 5.0); opt.step()
        net.eval()
        with torch.no_grad():
            vl = float(loss_on(torch.as_tensor(vi, device=dev)))
        if vl < best - 1e-4:
            best, state, bad = vl, copy.deepcopy(net.state_dict()), 0
        else:
            bad += 1
            if bad >= patience:
                break
    net.load_state_dict(state); net.eval()
    return net


@torch.no_grad()
def predict_forecast(net, h, x, p, dev=DEV, bs=1024):
    out = []
    N, T1 = p.shape
    rem_all = (1 - np.arange(T1, dtype=np.float32) / max(T1 - 1, 1)).clip(min=0)
    for k in range(0, N, bs):
        t_ = lambda a: None if a is None else torch.as_tensor(a[k:k + bs], dtype=torch.float32, device=dev)
        rem = torch.as_tensor(np.repeat(rem_all[None], min(bs, N - k), 0), device=dev)
        out.append(torch.softmax(net(t_(h), t_(x), t_(p), rem), -1).cpu().numpy())
    return np.concatenate(out).astype(np.float32)                     # [N, T1, T, K]


class CrossForecast:
    def __init__(self, h, x, p_oof, y, lat, fold, seed=0, **kw):
        self.fold, self.nets = fold, []
        for k in range(fold.max() + 1):
            tr = fold != k
            self.nets.append(fit_forecast(None if h is None else h[tr], x[tr], p_oof[tr], y[tr], lat, seed=4000 * seed + 13 * k, **kw))

    def oof(self, h, x, p):
        out = None
        for k, n in enumerate(self.nets):
            m = self.fold == k
            o = predict_forecast(n, None if h is None else h[m], x[m], p[m])
            if out is None:
                out = np.zeros((len(x),) + o.shape[1:], np.float32)
            out[m] = o
        return out

    def __call__(self, h, x, p):
        return np.mean([predict_forecast(n, h, x, p) for n in self.nets], 0)


def payoffs(lat, b, p, g, th, last, payoff='belief'):
    """cell payoffs stopv [last+1, K] and current payoffs stop_i [N, last+1].
    payoff='belief': the belief is the probability of correctness (calibration): 1{p >= g}(p - theta), cells valued
                     at their mean belief;  'table': the calibration table acc_t[cell] replaces the belief."""
    if payoff == 'belief':
        stopv = np.repeat(((lat.mid >= g) * (lat.mid - th))[None], last + 1, 0)
        stop_i = (p[:, : last + 1] >= g) * (p[:, : last + 1] - th)
    else:
        stopv = (lat.mid[None, :] >= g) * (lat.acc[: last + 1] - th)
        stop_i = (p[:, : last + 1] >= g) * (lat.acc[np.arange(last + 1)[None, :], b[:, : last + 1]] - th)
    return stopv, stop_i


def fair_price(lat, Fc, b, p, g, th, cost, last=None, myopic=False, payoff='belief'):
    """Fair price of reading from the multi-horizon forecast Fc [N, T1, T, K]: the best forecasted stop payoff
    gain per unit of resource over deterministic continuation lengths,
        rho_t = max_{t < tau <= last} ( sum_k Fc[t, tau](k) stopv[tau, k] - stop_i,t ) / (cost_tau - cost_t).
    cost [N, T1] (resource spent when stopping at a round); myopic: tau = t + 1 only.  Returns rho [N, last]."""
    last = lat.T if last is None else int(last)
    N = len(b)
    stopv, stop_i = payoffs(lat, b, p, g, th, last, payoff)
    rho = np.full((N, last), -np.inf)
    for t in range(last):
        best = np.full(N, -np.inf)
        for tau in range(t + 1, (t + 2) if myopic else (last + 1)):
            ev = Fc[:, t, tau - 1] @ stopv[tau]                                                     # [N]
            dc = cost[:, tau] - cost[:, t]
            best = np.maximum(best, (ev - stop_i[:, t]) / np.maximum(dc, 1e-9))
        rho[:, t] = best
    return rho


def index_curve(rho, grid, last):
    """nested deployment curve from a scalar index along an ascending price grid (as fpi.index_path), with the
    same column convention as plan(): col 0 never stops early, col J+1 stops at once, col J+2 abstains."""
    kap = grid[np.isfinite(grid)]
    S = fpi.index_path(rho, np.concatenate([[-np.inf], kap, [np.inf]]), last)     # [N, J+2+1]
    return S.astype(np.int16)


def pop_gate(s, yy, target):
    """Population admission gate of a stopped distribution (scores s, correctness yy): batch target ('cs', q) ->
    score of the largest prefix (by descending score) whose precision is >= 1 - q (the population analogue of
    Benjamini-Hochberg selection); risk target ('risk', alpha) -> smallest score g with
    mean(1{s >= g} wrong) <= alpha.  Abstentions carry s = -inf."""
    fin = np.isfinite(s)
    s, yy = s[fin], yy[fin]
    if len(s) == 0:
        return 1.0
    o = np.argsort(-s); k = np.arange(1, len(s) + 1)
    if target[0] == 'cs':
        ok = np.where(np.cumsum(yy[o]) / k >= 1 - target[1])[0]
    else:
        ok = np.where(np.cumsum(1 - yy[o]) / len(fin) <= target[1])[0]
    return float(s[o[ok[-1]]]) if len(ok) else 1.0


def anchor_gate(p_tr, y_tr, B, target, T):
    """gate of the fixed horizon at this budget: population gate of Mix(B) on the out-of-fold training beliefs
    (randomised rounds floor(B)/ceil(B) with the same frequencies as Mix)."""
    import cert14 as C
    st = C.mix_stops(len(p_tr), B, T)
    ar = np.arange(len(p_tr))
    return pop_gate(p_tr[ar, st], y_tr[ar, st], target)


def refine_gate(S_tr, p_tr, y_tr, cost_tr, cmax, B, target, R=20, seed=0):
    """one fixed-point step: population gate of the planner's OWN stopped distribution on training halvings."""
    import cert14 as C
    N = len(p_tr); ar = np.arange(N)
    rng = np.random.RandomState(31 + seed); U = rng.rand(N)
    co = C.env_cols(fpi.path_cost(S_tr, cost_tr))
    gs = []
    for _ in range(R):
        pm = rng.permutation(N); Bh = pm[: N // 4]
        st, ab, _ = C.deploy_once(S_tr, co, cmax, U, Bh, B, cost_is_envelope=True)
        gs.append(pop_gate(np.where(ab, -np.inf, p_tr[ar, st]), y_tr[ar, st], target))
    return float(np.median(gs))


def select_gate(lat, Q_tr, b_tr, p_tr, L_tr, cost_tr, cmax, B, target, grid, dc, gates=GATES, R=20, seed=0, last=None, myopic=False, curve_fn=None):
    """Gate and strike on TRAINING questions: deploy the planner at every candidate gate on random halvings of the
    training questions with the same budget and answer certificates as at evaluation; keep the gate with the
    highest mean certified coverage (ties -> the larger gate).  Returns (g, theta, scores)."""
    import cert14 as C
    N = len(p_tr); ar = np.arange(N)
    rng = np.random.RandomState(31 + seed)
    U = rng.rand(N)
    perms = [rng.permutation(N) for _ in range(R)]
    sgrid = C.strike_grid()
    best, bg, scores = -1.0, None, {}
    for g in gates:
        th = lat.theta_at(g)
        S = plan(lat, Q_tr, b_tr, p_tr, g, th, grid, dc, last=last, myopic=myopic) if curve_fn is None else curve_fn(g, th)
        co = C.env_cols(fpi.path_cost(S, cost_tr))
        cv = []
        for pm in perms:
            cal, te = pm[: N // 2], pm[N // 2:]
            A, Bh = cal[: len(cal) // 2], cal[len(cal) // 2:]
            st, ab, _ = C.deploy_once(S, co, cmax, U, Bh, B, cost_is_envelope=True)
            s = np.where(ab, -np.inf, p_tr[ar, st]); Lw = L_tr[ar, st]
            sel = C._answer(target[0], target[1], s, Lw, A, cal, te, sgrid)
            cv.append(sel.mean())
        scores[g] = float(np.mean(cv))
        if scores[g] > best + 1e-12 or (abs(scores[g] - best) <= 1e-12 and g > bg):
            best, bg = scores[g], g
    return bg, lat.theta_at(bg), scores

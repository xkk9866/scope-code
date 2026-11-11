"""Round 12: SCOPE learns the FAIR PRICE OF THE NEXT SEARCH.

Theory (paper Sec. 4).  Fix a strike theta (the dual price of the reliability target).  Stopping at round t earns
R_t = 1{p_t >= theta} (y_t - theta) (answer iff the verifier clears the strike) and each unit of resource (a retrieval
or a reader token) costs kappa.  For EVERY kappa in R, the optimal controller stops at the first round whose fair price

        rho_t(theta) = sup_{t < tau <= T}  E[R_tau - R_t | F_t] / E[c_tau - c_t | F_t]

is at most kappa (a Gittins-type index: the largest gain per unit of resource that any continuation can still buy).
Hence one index gives the optimal policy at every retrieval price simultaneously, and the stopping round
min{t : rho_t <= kappa} is non-increasing in kappa for every question -- whatever a network outputs.  A learned index is
therefore always a nested deployment curve, the property the tight budget certificate needs.

Learning (counterfactual policy optimisation).  A full-horizon trajectory reveals R_t and c_t for every round (A2), so
the utility of the index policy at any price is an exact function of the index on every training question; one network
is trained to maximise it at random (kappa, theta) -- consistent because the true index maximises it at every price at
once -- with a smoothed first crossing, after a warm start on the one-step (myopic) index.  The verifier beliefs it
reads are cross-fitted, so the index is trained on the beliefs it will see at deployment, not on memorised ones.
Round 13: the strike is GATE-CONSISTENT -- the fixed point theta = Gamma(theta) of the map from a strike to the answering
gate the certificate applies to the policy's stopped beliefs (complementary slackness of the Lagrangian), computed on
out-of-fold training questions (gate_theta); select_theta (max certified coverage over a grid) is kept as an ablation.

Components
  views()          per-round observations the loop produces for free (no extra reader calls)
  CrossFit         K-fold cross-fitted verifier: out-of-fold beliefs on training questions, fold-ensemble elsewhere
  CrossPolicy      K-fold cross-fitted index networks (kind='index') or matched scalar value networks (kind='value',
                   least-squares Monte Carlo, the strongest learned baseline and the main ablation)
  index_path()     stopping rounds along an ascending price grid (nested by construction)
  select_theta()   dual strike from the out-of-fold training questions
"""
import os, copy, math
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F

DEV = os.environ.get('STOPRAG_DEV', 'cuda' if torch.cuda.is_available() else 'cpu')

FREE = ['lp_mean', 'lp_min', 'ent_mean', 'p_suff', 'suff_said', 'ret_mean', 'ret_max', 'novelty', 'votes_new',
        'votes_cum', 'any_support', 'same_prev', 'run_len', 'n_changes', 'ans_len', 'is_yesno', 'n_gen', 'fmt_ok']
GROUND = ['p_suff', 'suff_said', 'votes_new', 'votes_cum', 'any_support']      # evidence-grounding view
COMPACT = ['lp_mean', 'novelty']                                                 # the 3-signal input of rounds 1-11
VIEWS = ('multi', 'compact', 'noground', 'blackbox')


# ----------------------------------------------------------------------------------------------- observations ------
def views(z, H, mode='multi'):
    """Per-round observations X [N, T1, F] and whether the hidden state is used.
    multi    : every free signal (generation, retrieval, evidence grounding, answer dynamics) + reader state
    compact  : answer log-prob, novelty, state drift + reader state (rounds 1-11)
    noground : multi without the evidence-grounding signals
    blackbox : multi without anything computed from the reader state (no hidden state, no state drift)."""
    names = list(z['u_names']); u = np.asarray(z['u'], np.float32)
    cols = {'multi': FREE, 'blackbox': FREE, 'noground': [c for c in FREE if c not in GROUND], 'compact': COMPACT}[mode]
    N, T1 = u.shape[:2]
    X = [u[:, :, [names.index(c) for c in cols]]]
    use_h = H is not None and mode != 'blackbox'
    if use_h:
        hn = H / (np.linalg.norm(H, axis=-1, keepdims=True) + 1e-6)
        X.append(np.concatenate([np.zeros((N, 1), np.float32), (hn[:, 1:] * hn[:, :-1]).sum(-1)], 1)[..., None])
    if mode != 'compact':
        X.append(np.asarray(z['d'], np.float32)[:, :, 1:2])            # answer-embedding similarity to round t-1
    X.append(np.repeat((np.arange(T1, dtype=np.float32) / max(T1 - 1, 1))[None, :, None], N, 0))
    return np.concatenate(X, -1).astype(np.float32), use_h


class Std:
    def __init__(self, X):
        f = X.reshape(-1, X.shape[-1]); self.m, self.s = f.mean(0), f.std(0) + 1e-6

    def __call__(self, X):
        return ((X - self.m) / self.s).astype(np.float32)


def pca_fit(H_tr, k, dev='cpu', seed=0):
    torch.manual_seed(seed)
    X = torch.as_tensor(H_tr.reshape(-1, H_tr.shape[-1]), dtype=torch.float32, device=dev)
    mu = X.mean(0)
    _, _, V = torch.pca_lowrank(X - mu, q=k, center=False, niter=6)
    zs = ((X - mu) @ V).std(0) + 1e-6
    return mu, V, zs


def pca_apply(H, P, dev='cpu', bs=20000):
    mu, V, zs = P
    X = H.reshape(-1, H.shape[-1]); out = []
    for i in range(0, len(X), bs):
        x = torch.as_tensor(X[i:i + bs], dtype=torch.float32, device=dev)
        out.append((((x - mu) @ V) / zs).cpu().numpy())
    return np.concatenate(out).reshape(H.shape[0], H.shape[1], -1).astype(np.float32)


# ----------------------------------------------------------------------------------------------- networks ----------
class Enc(nn.Module):
    """Multi-view round encoder + causal GRU over rounds (the controller only sees F_t)."""

    def __init__(self, dh, dx, H=128, p=0.25):
        super().__init__()
        self.hp = nn.Sequential(nn.Dropout(p), nn.Linear(dh, H), nn.LayerNorm(H)) if dh else None
        self.xp = nn.Sequential(nn.Linear(dx, H), nn.GELU(), nn.Linear(H, H), nn.LayerNorm(H))
        self.gru = nn.GRU(H * (2 if dh else 1), H, batch_first=True)

    def forward(self, h, x):
        e = self.xp(x)
        if self.hp is not None:
            e = torch.cat([self.hp(h), e], -1)
        return self.gru(e)[0]


def mlp(i, h, o, p=0.1):
    return nn.Sequential(nn.Linear(i, h), nn.GELU(), nn.Dropout(p), nn.Linear(h, h), nn.GELU(), nn.Linear(h, o))


class Verifier(nn.Module):
    def __init__(self, dh, dx, H=128):
        super().__init__()
        self.enc, self.head = Enc(dh, dx, H), mlp(H, H, 1)

    def forward(self, h, x):
        return self.head(self.enc(h, x)).squeeze(-1)


class PolicyNet(nn.Module):
    """kind='index': fair price rho_t(theta) of the next search (price units).
    kind='value' : continuation value C_t(theta, kappa) (utility units), the matched least-squares baseline.
    Both read the same encoder over [observations, cross-fitted belief p_t, its log-odds]; the head sees the strike,
    the moneyness p_t - theta and the remaining resource fraction."""

    def __init__(self, dh, dx, kind='index', H=128, ps=1.0):
        super().__init__()
        self.kind, self.ps = kind, float(ps)
        self.enc = Enc(dh, dx + 2, H)
        self.head = mlp(H + 4 + (kind == 'value'), H, 1)

    def feats(self, h, x, p):
        lp = torch.logit(p.clamp(1e-4, 1 - 1e-4)) / 4
        return self.enc(h, torch.cat([x, p[..., None], lp[..., None]], -1))

    def forward(self, g, p, rem, th, kap=None):
        """g [B,T1,H], p/rem [B,T1], th [B,S], kap [B,S] (value only) -> [B,S,T1]."""
        Bn, T1, Hd = g.shape
        S = th.shape[1]
        tt = th[:, :, None].expand(Bn, S, T1)
        pp = p[:, None].expand(Bn, S, T1)
        f = [g[:, None].expand(Bn, S, T1, Hd), torch.stack([tt, pp - tt, (pp - tt).abs(), rem[:, None].expand(Bn, S, T1)], -1)]
        if self.kind == 'value':
            f.append((kap / self.ps)[:, :, None, None].expand(Bn, S, T1, 1).clamp(-5, 5))
        out = self.head(torch.cat(f, -1)).squeeze(-1)
        return out * self.ps if self.kind == 'index' else out


# ----------------------------------------------------------------------------------------------- utilities ---------
def _t(a, dev):
    return None if a is None else torch.as_tensor(a, dtype=torch.float32, device=dev)


def rewards(p, y, th, mode='label'):
    """R [B,S,T1] of stopping at t: 1{p_t >= theta}(y_t - theta) ('label', the realised Lagrangian payoff) or
    (p_t - theta)^+ ('belief')."""
    pp, tt = p[:, None], th[:, :, None]
    if mode == 'belief':
        return (pp - tt).clamp(min=0)
    return (pp >= tt).float() * (y[:, None] - tt)


def soft_utility(rho, R, cost, kap, tau, last):
    """Expected utility of the smoothed index policy: hazard sigmoid((kappa - rho_t)/tau) for t < last, forced stop at
    last.  rho/R [B,S,T1], cost [B,T1], kap [B,S]."""
    hz = torch.sigmoid((kap[..., None] - rho[..., :last]) / tau)
    hz = torch.cat([hz, torch.ones_like(hz[..., :1])], -1)
    surv = torch.cumprod(torch.cat([torch.ones_like(hz[..., :1]), 1 - hz[..., :-1]], -1), -1)
    pay = R[..., : last + 1] - kap[..., None] * cost[:, None, : last + 1]
    return (surv * hz * pay).sum(-1)


def hard_stop_index(rho, kap, last):
    """rho [..., T1] -> first t < last with rho_t <= kappa, else last."""
    hit = rho[..., :last] <= kap
    t = hit.float().argmax(-1)
    return torch.where(hit.any(-1), t, torch.full_like(t, last))


def index_path(rho, grid, last):
    """Stopping rounds [N, J+1] along an ascending price grid (column 0 should be -inf: the fixed horizon 'last');
    last column = stop at once and abstain.  Non-increasing in the column for every question (Theorem 3)."""
    cr = np.minimum.accumulate(rho[:, :last], 1)
    st = np.zeros((len(rho), len(grid)), np.int16)
    for k in range(last):
        st += cr[:, k:k + 1] > grid[None, :]
    return np.concatenate([st, np.zeros((len(rho), 1), np.int16)], 1)


def price_grid(rho_ref, last, J=400):
    q = np.unique(np.quantile(rho_ref[:, :last], np.linspace(0, 1, J - 2)))
    return np.concatenate([[-np.inf], q, [np.inf]])


def value_path(C, p, th, grid_fin, last):
    """Value baseline: stop at the first t < last with (p_t - theta)^+ >= C_t(kappa) for every kappa of grid_fin;
    C [N, J, T1].  Columns: -inf (fixed horizon), grid_fin, +inf (stop at 0), abstain."""
    N = len(p)
    imm = np.clip(p - th, 0, None)[:, None, :last]
    hit = imm >= C[:, :, :last]
    t = hit.argmax(-1); t[~hit.any(-1)] = last
    return np.concatenate([np.full((N, 1), last), t, np.zeros((N, 2), int)], 1).astype(np.int16)


# ----------------------------------------------------------------------------------------------- training ----------
def _split(N, val, rng):
    perm = rng.permutation(N); nv = max(int(val * N), 1)
    return perm[nv:], perm[:nv]


def fit_verifier(h, x, y, seed=0, epochs=100, lr=1e-3, wd=1e-2, bs=128, val=0.15, patience=12, dev=DEV, H=128):
    rng = np.random.RandomState(seed); torch.manual_seed(seed)
    ti, vi = _split(len(y), val, rng)
    net = Verifier(0 if h is None else h.shape[-1], x.shape[-1], H).to(dev)
    opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=wd)
    Ht, Xt, Yt = _t(h, dev), _t(x, dev), _t(y, dev)
    sub = lambda A, i: None if A is None else A[i]
    best, state, bad = np.inf, None, 0
    for ep in range(epochs):
        net.train()
        perm = ti[rng.permutation(len(ti))]
        for k in range(0, len(perm), bs):
            b = torch.as_tensor(perm[k:k + bs], device=dev)
            loss = F.binary_cross_entropy_with_logits(net(sub(Ht, b), Xt[b]), Yt[b])
            opt.zero_grad(); loss.backward(); opt.step()
        net.eval()
        with torch.no_grad():
            vb = torch.as_tensor(vi, device=dev)
            vl = float(F.binary_cross_entropy_with_logits(net(sub(Ht, vb), Xt[vb]), Yt[vb]))
        if vl < best - 1e-4:
            best, state, bad = vl, copy.deepcopy(net.state_dict()), 0
        else:
            bad += 1
            if bad >= patience:
                break
    net.load_state_dict(state); net.eval()
    return net


@torch.no_grad()
def predict(net, h, x, dev=DEV, bs=1024):
    out = []
    for k in range(0, len(x), bs):
        out.append(torch.sigmoid(net(_t(None if h is None else h[k:k + bs], dev), _t(x[k:k + bs], dev))).cpu().numpy())
    return np.concatenate(out).astype(np.float32)


class CrossFit:
    """K-fold cross-fitted verifier on the training questions: oof [Ntr, T1] out-of-fold beliefs; __call__ averages
    the K fold models (each trained on (K-1)/K of the training questions)."""

    def __init__(self, h, x, y, K=5, seed=0, **kw):
        self.K = K
        self.fold = np.random.RandomState(7000 + seed).permutation(len(y)) % K
        self.nets, self.oof = [], np.zeros(y.shape, np.float32)
        for k in range(K):
            tr = self.fold != k
            net = fit_verifier(None if h is None else h[tr], x[tr], y[tr], seed=1000 * seed + k, **kw)
            self.nets.append(net)
            self.oof[~tr] = predict(net, None if h is None else h[~tr], x[~tr])

    def __call__(self, h, x):
        return np.mean([predict(n, h, x) for n in self.nets], 0)


def fit_policy(h, x, p, y, cost, kind='index', last=None, seed=0, epochs=60, warm=6, lr=1e-3, wd=1e-2, bs=128, S=16,
               th_lo=0.3, th_hi=0.97, val=0.15, patience=10, reward='label', dev=DEV, H=128, verbose=False):
    """Train one index network (counterfactual policy optimisation) or one matched value network (least-squares Monte
    Carlo).  p: cross-fitted beliefs; cost [N,T1]: resource spent if stopping at t (non-decreasing); last: forced
    stopping round (T, or a deadline)."""
    rng = np.random.RandomState(seed); torch.manual_seed(seed)
    N, T1 = y.shape
    last = T1 - 1 if last is None else int(last)
    dc = float(np.mean(cost[:, last] - cost[:, 0]) / max(last, 1))         # mean resource per round
    ps = 1.0 / max(dc, 1e-9)                                                # price scale: utility per unit resource
    net = PolicyNet(0 if h is None else h.shape[-1], x.shape[-1], kind, H, ps).to(dev)
    opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=wd)
    Ht, Xt, Pt, Yt, Ct = _t(h, dev), _t(x, dev), _t(p, dev), _t(y, dev), _t(cost, dev)
    # Remaining round fraction is observable; final per-question token cost is not.
    rem = (1 - torch.arange(T1, device=dev, dtype=torch.float32) / max(last, 1)).clamp(min=0)
    rem = rem[None].expand(N, -1)
    sub = lambda A, i: None if A is None else A[i]
    ti, vi = _split(N, val, rng)
    # validation: hard-policy utility on a fixed (theta, kappa) grid
    vth = torch.tensor([0.5, 0.7, 0.8, 0.9, 0.95], device=dev)
    vka = torch.tensor([-0.2, 0.0, 0.02, 0.05, 0.1, 0.2, 0.4], device=dev) * ps
    VT, VK = torch.meshgrid(vth, vka, indexing='ij'); VT, VK = VT.reshape(-1), VK.reshape(-1)

    def val_util():
        net.eval()
        with torch.no_grad():
            b = torch.as_tensor(vi, device=dev)
            g = net.feats(sub(Ht, b), Xt[b], Pt[b])
            th = VT[None].expand(len(vi), -1); ka = VK[None].expand(len(vi), -1)
            R = rewards(Pt[b], Yt[b], th, reward)
            if kind == 'index':
                st = hard_stop_index(net(g, Pt[b], rem[b], th), ka[..., None], last)
            else:
                C = net(g, Pt[b], rem[b], th, ka)
                imm = (Pt[b][:, None] - th[..., None]).clamp(min=0)
                hit = imm[..., :last] >= C[..., :last]
                st = hit.float().argmax(-1); st = torch.where(hit.any(-1), st, torch.full_like(st, last))
            u = R.gather(-1, st[..., None]).squeeze(-1) - ka * Ct[b].gather(1, st)
        return float(u.mean())

    best, state, bad = -np.inf, None, 0
    tau0, tau1 = 0.05 * ps, 0.005 * ps
    for ep in range(warm + epochs):
        net.train()
        perm = ti[rng.permutation(len(ti))]
        tau = tau0 * (tau1 / tau0) ** (max(ep - warm, 0) / max(epochs - 1, 1))
        for k in range(0, len(perm), bs):
            b = torch.as_tensor(perm[k:k + bs], device=dev); nb = len(b)
            th = th_lo + (th_hi - th_lo) * torch.rand(nb, S, device=dev)
            R = rewards(Pt[b], Yt[b], th, reward)
            g = net.feats(sub(Ht, b), Xt[b], Pt[b])
            if kind == 'index':
                rho = net(g, Pt[b], rem[b], th)
                if ep < warm:            # one-step (myopic) index: E[R_{t+1} - R_t | F_t] / E[c_{t+1} - c_t | F_t]
                    tgt = (R[..., 1: last + 1] - R[..., :last]) / (Ct[b, 1: last + 1] - Ct[b, :last] + 1e-9)[:, None]
                    loss = F.smooth_l1_loss(rho[..., :last] / ps, tgt / ps)
                else:
                    ku = ps * (-0.5 + 1.5 * torch.rand(nb, S, device=dev))
                    tr_ = torch.randint(0, last, (nb, S, 1), device=dev)
                    ka_ = rho.detach().gather(-1, tr_).squeeze(-1) + 0.02 * ps * torch.randn(nb, S, device=dev)
                    kap = torch.where(torch.rand(nb, S, device=dev) < 0.5, ku, ka_)
                    loss = -soft_utility(rho, R, Ct[b], kap, tau, last).mean()
            else:
                kap = ps * torch.where(torch.rand(nb, S, device=dev) < 0.2, torch.zeros(nb, S, device=dev),
                                       -0.5 + 1.5 * torch.rand(nb, S, device=dev))
                C = net(g, Pt[b], rem[b], th, kap)
                with torch.no_grad():    # Longstaff-Schwartz: realised payoff of the current policy from t+1
                    imm = (Pt[b][:, None] - th[..., None]).clamp(min=0)
                    W = R[..., last].clone(); tgt = torch.zeros_like(C)
                    for t in range(last - 1, -1, -1):
                        tgt[..., t] = W - kap * (Ct[b, t + 1] - Ct[b, t])[:, None]
                        stop = imm[..., t] >= C[..., t].detach()
                        W = torch.where(stop, R[..., t], tgt[..., t])
                loss = F.mse_loss(C[..., :last], tgt[..., :last])
            opt.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(net.parameters(), 5.0); opt.step()
        if ep >= warm - 1 or kind == 'value':      # the myopic index (end of the warm start) is a candidate too
            vu = val_util()
            if verbose:
                print(kind, 'ep', ep, 'val utility', round(vu, 5), flush=True)
            if vu > best + 1e-5:
                best, state, bad = vu, copy.deepcopy(net.state_dict()), 0
            else:
                bad += 1
                if bad >= patience:
                    break
    net.load_state_dict(state); net.eval()
    net.last = last
    return net


@torch.no_grad()
def policy_eval(net, h, x, p, cost, ths, kaps=None, dev=DEV, bs=None):
    """Index: rho [N, len(ths), T1].  Value: C [N, len(ths), len(kaps), T1] (one strike at a time, small batches)."""
    last = net.last
    bs = bs or (512 if net.kind == 'index' else 96)
    out = []
    for k in range(0, len(x), bs):
        P = _t(p[k:k + bs], dev); Cc = _t(cost[k:k + bs], dev)
        rem = (1 - torch.arange(P.shape[1], device=dev, dtype=torch.float32) / max(last, 1)).clamp(min=0)
        rem = rem[None].expand(len(P), -1)
        g = net.feats(_t(None if h is None else h[k:k + bs], dev), _t(x[k:k + bs], dev), P)
        nb = len(P)
        if net.kind == 'index':
            th = torch.tensor(ths, dtype=torch.float32, device=dev)[None].expand(nb, -1)
            out.append(net(g, P, rem, th).cpu().numpy())
        else:
            ka = torch.tensor(kaps, dtype=torch.float32, device=dev)[None].expand(nb, -1)
            per = [net(g, P, rem, torch.full_like(ka, float(t_)), ka).cpu().numpy() for t_ in ths]
            out.append(np.stack(per, 1))
    return np.concatenate(out)


class CrossPolicy:
    """K-fold cross-fitted policy networks on the same folds as the verifier: oof(...) gives out-of-fold outputs on
    the training questions (for choosing the strike), __call__ the fold average on other questions (an average of
    indices is an index, so the deployed curve stays nested)."""

    def __init__(self, h, x, p_oof, y, cost, fold, kind='index', seed=0, **kw):
        self.fold, self.kind, self.nets = fold, kind, []
        for k in range(fold.max() + 1):
            tr = fold != k
            self.nets.append(fit_policy(None if h is None else h[tr], x[tr], p_oof[tr], y[tr], cost[tr], kind=kind,
                                        seed=2000 * seed + 17 * k + (kind == 'value'), **kw))
        self.last, self.ps = self.nets[0].last, self.nets[0].ps

    def oof(self, h, x, p, cost, ths, kaps=None):
        out = None
        for k, n in enumerate(self.nets):
            m = self.fold == k
            o = policy_eval(n, None if h is None else h[m], x[m], p[m], cost[m], ths, kaps)
            if out is None:
                out = np.zeros((len(x),) + o.shape[1:], np.float32)
            out[m] = o
        return out

    def __call__(self, h, x, p, cost, ths, kaps=None):
        return np.mean([policy_eval(n, h, x, p, cost, ths, kaps) for n in self.nets], 0)


# ----------------------------------------------------------------------------------------------- the dual strike ----
def path_cost(S, cost):
    """Resource spent along a curve: cost [N, T1] of stopping at round t; the stop-and-abstain column spends 0."""
    co = cost[np.arange(len(S))[:, None], np.minimum(S, cost.shape[1] - 1)].astype(float)
    co[:, -1] = 0.0
    return co


def select_theta(paths, p, L, cost, cmax, B, target, R=40, seed=0):
    """Dual strike: among candidate strikes (paths: theta -> stopping curve on out-of-fold training questions), the
    one whose deployment curve certifies the most coverage for `target` = ('cs', q) or ('risk', alpha) at budget B,
    averaged over R random calibration/test halvings of the training questions.  Uses training questions only, so
    every certificate on the evaluation questions is unaffected."""
    import cert14 as C
    N = len(p); ar = np.arange(N)
    rng = np.random.RandomState(31 + seed)
    U = rng.rand(N)
    perms = [rng.permutation(N) for _ in range(R)]
    grid = C.strike_grid()
    best, bth, scores = -1.0, None, {}
    for th, S in paths.items():
        cv = []
        co = C.env_cols(path_cost(S, cost))
        for pm in perms:
            cal, te = pm[: N // 2], pm[N // 2:]
            A, Bh = cal[: len(cal) // 2], cal[len(cal) // 2:]
            st, ab, _ = C.deploy_once(S, co, cmax, U, Bh, B, cost_is_envelope=True)
            s = np.where(ab, -np.inf, p[ar, st]); Lw = L[ar, st]
            sel = C._answer(target[0], target[1], s, Lw, A, cal, te, grid)
            cv.append(sel.mean())
        scores[th] = float(np.mean(cv))
        if scores[th] > best + 1e-12 or (abs(scores[th] - best) <= 1e-12 and th > bth):
            best, bth = scores[th], th
    return bth, scores


# ----------------------------------------------------------------------------------------------- the gate strike -----
STRIKES = tuple(float(v) for v in np.round(np.linspace(0.31, 0.97, 23), 2))   # the training range, step 0.03


def gate_map(paths, p, L, cost, cmax, B, target, R=40, seed=0):
    """Finite-sample gate proxy for stopped beliefs when
    the index policy at strike theta is deployed at budget B -- the smallest belief among the questions the
    certificate answers (batch target: minimum selected score, not an exact deterministic belief threshold;
    risk target: the CRC threshold), median over R random
    calibration/test halvings of the TRAINING questions.  NaN when the certificate answers nothing."""
    import cert14 as C
    N = len(p); ar = np.arange(N)
    rng = np.random.RandomState(31 + seed)
    U = rng.rand(N)
    perms = [rng.permutation(N) for _ in range(R)]
    grid = C.strike_grid()
    G = {}
    for th, S in paths.items():
        co = C.env_cols(path_cost(S, cost))
        g = []
        for pm in perms:
            cal, te = pm[: N // 2], pm[N // 2:]
            A, Bh = cal[: len(cal) // 2], cal[len(cal) // 2:]
            st, ab, _ = C.deploy_once(S, co, cmax, U, Bh, B, cost_is_envelope=True)
            s = np.where(ab, -np.inf, p[ar, st]); Lw = L[ar, st]
            if target[0] == 'risk':
                gt = float(C.crc_strike(s, Lw, A, grid, target[1]))
                g.append(gt if gt <= 1.0 else np.nan)
            else:
                sel = C.conf_select(s, Lw, A, te, target[1])
                ss = s[te][sel]; ss = ss[np.isfinite(ss)]
                g.append(float(ss.min()) if len(ss) else np.nan)
        g = np.asarray(g, float)
        G[th] = float(np.nanmedian(g)) if np.isfinite(g).any() else np.nan
    return G


def gate_theta(paths, p, L, cost, cmax, B, target, R=40, seed=0):
    """Empirical gate-proxy strike, motivated by the population condition of Proposition 'gate'.
    Finite-sample conformal selections do not identify the population optimal strike.
    Returns the candidate strike closest to a fixed point (ties: the larger strike) and the map Gamma.  If the
    certificate answers nothing at any strike, use the predefined fallback (nearest to 1 - q for a batch
    target, the largest strike for a risk target); this says nothing about population dual attainment."""
    G = gate_map(paths, p, L, cost, cmax, B, target, R, seed)
    ths = sorted(G)
    fin = [t for t in ths if np.isfinite(G[t])]
    if not fin:
        lim = 1 - target[1] if target[0] == 'cs' else ths[-1]
        return min(ths, key=lambda t: (abs(t - lim), -t)), G
    return min(fin, key=lambda t: (abs(G[t] - t), -t)), G

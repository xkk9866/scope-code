"""Round 12: the certification wrapper, self-contained (no dependence on the round 1-11 model code).

A stopping family with a monotone knob is a deployment curve S [N, J] (stopping round of every question at every
knob value, non-increasing in the column index; the last column stops at once and abstains).  On one random split
of the evaluation questions into a calibration half (itself halved into A and B) and a test half:
  budget  : the dithered conformal budget certificate on half B fixes the column (Theorem 2),
  risk    : conformal risk control of 1{s >= theta} * wrong on half A                  (Theorem 1),
  batch   : conformal selection (BH on clipped conformal p-values) at level q on half A  (Proposition 4),
and the deployed policy is evaluated on the test half.  The randomised fixed horizon (Mix) spends exactly B and needs
no budget certificate.  Every policy answers with the same score s (the shared verifier)."""
import numpy as np

COLS = ('risk', 'cov', 'ret', 'tok', 'serr', 'cost', 'ccov', 'p99ret', 'p99tok')


def strike_grid():
    return np.concatenate([np.linspace(0.0, 0.995, 200), [1.01]])


def envelope(loss):
    """Smallest non-increasing (in the column index) function above loss."""
    return np.maximum.accumulate(loss[:, ::-1], axis=1)[:, ::-1]


def certify_table(loss, alpha):
    """Conformal risk control over an increasing strike grid on the monotone envelope; returns the grid index."""
    env = envelope(loss)
    n = len(loss)
    ok = (env.sum(0) + 1) / (n + 1) <= alpha
    idx = np.where(ok)[0]
    return int(idx[0]) if len(idx) else loss.shape[1] - 1


def env_cols(S):
    return np.maximum.accumulate(S[:, ::-1], 1)[:, ::-1]


def budget_index_env(S, cost, U, idx, B, cmax, cost_is_envelope=False):
    """Dithered conformal budget certificate on half idx, on the monotone envelope of the cost path (valid for any
    path; tight for nested paths).  Returns (j, omega): a question uses column j iff U <= omega, else column j - 1."""
    Cst = cost if cost_is_envelope else env_cols(cost)
    n = len(idx)
    tot = Cst[idx].sum(0).astype(float)
    target = B * (n + 1) - cmax
    ok = tot <= target
    if not ok.any():
        return S.shape[1] - 1, 1.0
    j = int(np.argmax(ok))
    if j == 0:
        return 0, 1.0
    dif = Cst[idx, j - 1] - Cst[idx, j]
    m = dif > 0
    if not m.any():
        return j, 1.0
    us, dd = U[idx][m], dif[m]
    o = np.argsort(us)
    k = int(np.argmax(tot[j - 1] - np.cumsum(dd[o]) <= target))
    return j, float(us[o][k])


def deployed(S, U, j, om, dither=True):
    col = np.where((U <= om) | (not dither) | (j == 0), j, max(j - 1, 0))
    return S[np.arange(len(S)), col], col == S.shape[1] - 1, col


def crc_strike(s, Lw, idx, grid, alpha):
    tab = (s[idx][:, None] >= grid[None, :]) * Lw[idx][:, None]
    return grid[certify_table(tab, alpha)]


def tie_u(N, te=None):
    """Tie-breaking uniforms of the conformal p-values, independent of scores and labels and drawn AFRESH FOR EVERY
    SPLIT (seeded by the test indices of the split, so every policy certified on the same split shares them and the
    comparison stays paired).  A uniform that stayed attached to a question across the repeated splits of one pool
    would make the splits replicate one draw of the randomisation instead of averaging over it."""
    if te is None:
        return np.random.RandomState(4242).rand(N)
    t = np.asarray(te, np.int64)[:64]
    seed = int((t * (np.arange(len(t), dtype=np.int64) + 1)).sum() % 2147483629) + 4242
    return np.random.RandomState(seed).rand(N)


def conf_select(s, Lw, idxA, te, q, tie=None):
    """Conformal selection with the clipped score V = M 1{correct} - s (Jin & Candes, 2023), exactly as analysed there:
    p_j = (#{i in A: wrong_i, s_i > s_j} + U_j (1 + #{i in A: wrong_i, s_i = s_j})) / (|A| + 1), U_j ~ U[0, 1], then
    Benjamini-Hochberg at level q over the test batch.  A question at the stop-and-abstain end of the curve carries the
    lowest score s = -inf (calibration and test alike); predeclared test abstentions receive p=1 before BH."""
    ws = np.sort(s[idxA][Lw[idxA] > 0])
    nA = len(idxA)
    st = s[te]
    hi = np.searchsorted(ws, st, side='right'); lo = np.searchsorted(ws, st, side='left')
    u = tie_u(len(s), te)[te] if tie is None else tie[te]
    pv = ((len(ws) - hi) + u * (1 + hi - lo)) / (nA + 1.0)
    # A pre-declared abstention cannot become an answer through random tie-breaking.
    pv[~np.isfinite(st)] = 1.0
    m = len(te)
    o = np.sort(pv)
    ks = np.where(o <= q * np.arange(1, m + 1) / m)[0]
    if not len(ks):
        return np.zeros(m, bool)
    return pv <= q * (ks[-1] + 1) / m


def first_true(Bm, default):
    t = Bm.argmax(1)
    t[~Bm.any(1)] = default
    return t


def first_cross_path(score, thr_desc, last):
    """'Stop at the first round t < last with score_t >= c' for thresholds c in descending order (column 0 never
    stops early); appended stop-and-abstain column."""
    N = score.shape[0]
    cols = [first_true(score[:, :last] >= c, last) for c in thr_desc]
    return np.concatenate([np.stack(cols, 1), np.zeros((N, 1), int)], 1)


def thr_desc(x, n=160, top=np.inf):
    return np.concatenate([[top], np.unique(np.quantile(x, np.linspace(0, 1, n)))[::-1]])


def mix_stops(N, B, T, seed=7):
    lo = int(np.floor(B)); fr = B - lo
    return np.minimum(lo + (np.random.RandomState(seed).rand(N) < fr).astype(int), T)


def row(sel, Lw_te, st_te, tk_te, cs_te):
    n = sel.sum(); risk = float(np.mean(sel * Lw_te)); cov = float(np.mean(sel))
    return [risk, cov, float(st_te.mean()), float(tk_te.mean()), float((sel * Lw_te).sum() / max(n, 1)),
            float(cs_te.mean()), cov - risk, float(np.quantile(st_te, 0.99)), float(np.quantile(tk_te, 0.99))]


def _answer(kind, lev, s, Lw, A, c, te, grid):
    if kind == 'risk':
        return s[te] >= crc_strike(s, Lw, A, grid, lev)
    if kind == 'cs':
        return conf_select(s, Lw, A, te, lev)
    raise ValueError(kind)


def certify_curve(S, cost, cmax, s_all, L, tok, cal_idx, tests, U, B, targets, grid=None, dither=True):
    """targets: list of (kind, level) with kind in {'risk', 'cs'}.  Returns {(kind, level): [R, 9]}."""
    grid = strike_grid() if grid is None else grid
    N = len(S); ar = np.arange(N)
    rec = {t: [] for t in targets}
    cost_env = env_cols(cost)
    for r, c in enumerate(cal_idx):
        A, Bh = c[: len(c) // 2], c[len(c) // 2:]
        te = tests[r]
        j, om = budget_index_env(S, cost_env, U, Bh, B, cmax, cost_is_envelope=True)
        st, ab, col = deployed(S, U, j, om, dither)
        s = np.where(ab, -np.inf, s_all[ar, st]); Lw = L[ar, st]
        cs_ = cost[ar, col]; tk = np.where(ab, 0.0, tok[ar, st])
        for kind, lev in targets:
            sel = _answer(kind, lev, s, Lw, A, c, te, grid)
            rec[(kind, lev)].append(row(sel, Lw[te], st[te], tk[te], cs_[te]))
    return {k: np.array(v) for k, v in rec.items()}


def certify_fixed(st, s_all, L, tok, cal_idx, tests, targets, cost=None, grid=None):
    """A policy whose stopping rounds st [N] are fixed in advance (Mix): risk on the whole calibration half, batch
    selection on half A (the same half as the curves)."""
    grid = strike_grid() if grid is None else grid
    N = len(st); ar = np.arange(N)
    s = s_all[ar, st]; Lw = L[ar, st]; tk = tok[ar, st]
    cs_ = st.astype(float) if cost is None else cost
    rec = {t: [] for t in targets}
    for r, c in enumerate(cal_idx):
        A = c[: len(c) // 2]
        te = tests[r]
        for kind, lev in targets:
            sel = _answer(kind, lev, s, Lw, c if kind == 'risk' else A, c, te, grid)
            rec[(kind, lev)].append(row(sel, Lw[te], st[te], tk[te], cs_[te]))
    return {k: np.array(v) for k, v in rec.items()}


def deploy_once(S, cost, cmax, U, Bh, B, dither=True, cost_is_envelope=False):
    j, om = budget_index_env(S, cost, U, Bh, B, cmax, cost_is_envelope=cost_is_envelope)
    return deployed(S, U, j, om, dither)


def peek_inflation(p, L, cal_idx, tests, alphas, last, grid=None):
    """RQ1: one fixed-look conformal threshold per round (each valid at its own round), used inside the loop
    ('answer at the first round whose score clears its own threshold').  Returns {alpha: [R, 3]} = (stopped risk,
    largest single-round test risk, coverage)."""
    grid = strike_grid() if grid is None else grid
    N, T1 = p.shape
    out = {a: [] for a in alphas}
    for r, c in enumerate(cal_idx):
        te = tests[r]
        for a in alphas:
            th = np.array([crc_strike(p[:, t], L[:, t], c, grid, a) for t in range(last + 1)])
            hit = p[te][:, : last + 1] >= th[None]
            st = first_true(hit, last)
            ans = hit.any(1)
            risk = float(np.mean(ans * L[te, st]))
            single = max(float(np.mean(hit[:, t] * L[te, t])) for t in range(last + 1))
            out[a].append([risk, single, float(ans.mean())])
    return {a: np.array(v) for a, v in out.items()}

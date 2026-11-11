"""Round 21 experiment (results/r21), one unit per call: a reader on one benchmark, or a reader on the MIXED WORKLOAD
`mix3` (the three benchmarks pooled by mkmix21.py; `grp` = benchmark of every question).  The method and every part of
round 20 are unchanged (same code path); the parts added here are
  wl    : evaluation workloads cut from the evaluation pool of a mixed unit with the networks trained once on the
          mixed training pool -- each benchmark alone (g0, g1, g2) and compositions with a given share of the hardest
          benchmark (h000 ... h100); every workload gets its own splits, strikes, budget and answer certificates
  flow  : where the reads go and where the certified answers come from, per benchmark; reads spent on questions that
          end without a certified answer
  vcert : valid per-round certificates as baselines -- the error budget is split across the rounds (uniform split =
          Bonferroni; split in proportion to the questions first answerable at a round = MiCP-style allocation)
  plain : accuracy against reads with no certificate and no abstention (strike 0 for the option)
  reach : certified coverage on a fine budget grid (reads needed to reach a share of the full-depth coverage)
  valid : the two certificates under i.i.d. resampling of the workload (the pool as the population): realised error,
          risk and reads of every replicate, paired across policies (also the paired bootstrap of the differences)

SCOPE = cross-fitted verifier + the next read priced as an AMERICAN CALL ON THE BELIEF: a one-head kernel field emits
a history-conditioned Markov chain of the future belief path (fitted on belief paths alone), the option with stop
payoff (p - theta)^+ is priced on it by EXACT backward induction for every strike and price, and the strike is the
certificate-consistent fixed point found by EXHAUSTIVE residual minimisation on the labelled slice C of each split (no
iteration, no warm start, no state shared across splits or bootstrap replicates).  Three-way protocol per split:
C = first quarter (strike), A/B = second quarter (answer / budget certificates), test = second half.
Published comparison set (2023-2026): IRCoT, FLARE, Adaptive-RAG, DRAGIN, SeaKR, Probing-RAG, Stop-RAG, HALT, plus
the fixed horizon.  Label hindsight = upper bound on the Lagrangian reward of every adapted rule (reported as such).
Parts: main, boot, peek, oracle, calsize, p1, lab, abl, sabl (strike rules only), inv, sdiag, tok, dl, contracts.
  abl : the law     pop (population chain), anch (field = correction to the population kernel), tf (path likelihood only),
                    ctrlp / ctrlbb (the kernel field reads the belief path only / the scalar signals without the state)
        the payoff  digit (digital payoff P(p >= theta)), acc (strike 0: expected belief, i.e. accuracy),
                    rel (three-head model with label reliabilities in the payoff, bpm20rel)
        the price   myopic (one read ahead), dh (deterministic horizons on the same chain),
                    marg (multi-horizon marginal forecaster), lsmc (Longstaff-Schwartz value regression)
        the strike  th1q / sctr / free / oracle
        the reader  verbb / blackbox (reader-state 2 x 2 with ctrlbb)
  inv : monotone distortions of the verifier score (p^2, sqrt p)
  sdiag: strike diagnostics (residual of the exhaustive solution, number of near-crossings, KM iteration from 3 starts)"""
import os, sys, json, argparse, time
import numpy as np, torch

sys.path.insert(0, os.path.dirname(__file__))
import fpi, cert14 as C, lattice16 as Lt, bpm20 as B
from main16 import views16
from main15 import load_unit, auroc, norm_ans, NONANS, budgets

QS = (0.1, 0.15, 0.2)
ALPHAS = (0.1, 0.15, 0.2)
TOKF = (0.15, 0.25, 0.35, 0.5)
BASE = ('mix', 'ircot', 'flare', 'adapt', 'dragin', 'seakr', 'probe', 'stoprag', 'halt')
THETAS = B.THETAS
SGRID = C.strike_grid()
GRID_OLD = Lt.price_grid(200)


class Unit:
    """verifier (cross-fitted), cells, path model and reliability of one observation mode.
    base: reuse observations AND verifier (only the controller differs); obs: reuse observations only (new label);
    ctrl: the unit whose observations the PATH MODEL reads (default: its own)."""

    def __init__(self, z, H, y, tr, ev, view, args, log, base=None, obs=None, ctrl=None, fit_path=True):
        self.view, self.tr, self.ev = view, tr, ev
        src = base if base is not None else obs
        if src is not None:
            self.X, self.Hp = src.X, src.Hp
        else:
            X, use_h = views16(z, H, view)
            self.X = fpi.Std(X[tr])(X)
            if use_h:
                dv = os.environ.get('PCA_DEVICE', 'cpu')
                self.Hp = fpi.pca_apply(H, fpi.pca_fit(H[tr], args.pca, dv, seed=args.seed), dv)
            else:
                self.Hp = None
        hp = lambda idx: None if self.Hp is None else self.Hp[idx]
        if base is not None:
            self.V, self.p_tr, self.p_ev = base.V, base.p_tr, base.p_ev
        else:
            kw = dict(epochs=args.vep) if args.vep else {}
            self.V = fpi.CrossFit(hp(tr), self.X[tr], y[tr], K=args.K, seed=args.seed, **kw)
            self.p_tr, self.p_ev = self.V.oof, self.V(hp(ev), self.X[ev])
            log(f'verifier[{view}] ready')
        self.lat = Lt.Lattice(self.p_tr, y[tr], K=args.cells)
        self.b_tr, self.b_ev = self.lat.binof(self.p_tr), self.lat.binof(self.p_ev)
        self.call = B.call_table(self.p_tr, self.lat)          # stop payoff of a future cell: E[(p - theta)^+ | cell]
        self.frac = B.frac_table(self.p_tr, self.lat)          # digital payoff (control)
        c = self if ctrl is None else ctrl
        self.c = c
        chp = lambda idx: None if c.Hp is None else c.Hp[idx]
        if fit_path:
            kw = dict(epochs=args.fep) if args.fep else {}
            self.path = B.CrossPath(chp(tr), c.X[tr], self.p_tr, self.lat, self.V.fold, seed=args.seed, **kw)
            self.P_ev = self.path(chp(ev), c.X[ev], self.p_ev)
            self.P_tr = self.path.oof(chp(tr), c.X[tr], self.p_tr)
            log(f'path model[{view}{"" if ctrl is None else " ctrl=" + ctrl.view}] trained')

    def family(self, cost_ev, dc, last=None, myopic=False, P=None, p=None, U=None, dh=False, payoff='call'):
        P = self.P_ev if P is None else P; p = self.p_ev if p is None else p
        U = (self.call if payoff == 'call' else self.frac) if U is None else U
        kap = B.price_grid(dc)
        if dh:
            SS = B.plan_dh(P, self.b_ev, p, U, THETAS, kap, cost_ev, last=last, payoff=payoff)
        else:
            SS = B.plan(P, self.b_ev, p, U, THETAS, kap, dc, last=last, myopic=myopic, payoff=payoff)
        return B.Family(SS, cost_ev), kap


def solve_all(fam, p, yy, Bq, targets, idx, free=False):
    """Family.solve for several targets at one budget on the slice idx (the stopped sample of a strike does not
    depend on the target, so it is computed once): {target: (index, strike, residual, reached)} of the
    certificate-consistent strike -- the same exhaustive residual minimisation as Family.solve."""
    sts = [fam.stops_at_budget(o, idx, Bq) for o in range(len(fam.thetas))]
    out = {}
    for tg in targets:
        gr = [B.gamma_iso(p[idx, st], yy[idx, st], tg, len(idx), free=free) for st in sts]
        g = np.array([a for a, _ in gr])
        res = np.abs(fam.thetas - g)
        o = int(np.flatnonzero(res <= res.min() + 1e-12)[-1])
        out[tg] = (o, float(fam.thetas[o]), float(res[o]), bool(gr[o][1]))
    return out


class WL:
    """An evaluation workload: the questions `sub` of an evaluation pool with their own random splits (three-way
    protocol: C = first quarter, A | B = second quarter, test = second half), dither and certificates."""

    def __init__(self, sub, p, y, cost, tok, seed, R):
        self.sub, self.N = sub, len(sub)
        self.ar = np.arange(self.N)
        self.p, self.y, self.cost, self.tok = p[sub], y[sub], cost[sub], tok[sub]
        self.L = 1 - self.y
        rng = np.random.RandomState(123 + seed)
        self.perms = [rng.permutation(self.N) for _ in range(R)]
        self.Ud = np.random.RandomState(999).rand(self.N)

    def splits(self, r):
        pm, N = self.perms[r], self.N
        return pm[: N // 4], pm[N // 4: N // 2], pm[N // 2:]

    def cert(self, S, cst, cenv, cmax, cal, te, Bq, tg):
        """budget certificate on half B, answer certificate on half A, evaluation on the test half."""
        A, Bh = cal[: len(cal) // 2], cal[len(cal) // 2:]
        j, om = C.budget_index_env(S, cenv, self.Ud, Bh, Bq, cmax, cost_is_envelope=True)
        st, ab, col = C.deployed(S, self.Ud, j, om, True)
        s = np.where(ab, -np.inf, self.p[self.ar, st]); Lw = self.L[self.ar, st]
        sel = C._answer(tg[0], tg[1], s, Lw, A, cal, te, SGRID)
        row = C.row(sel, Lw[te], st[te], np.where(ab, 0.0, self.tok[self.ar, st])[te], cst[self.ar, col][te])
        return row, st, ab, sel

    def cert_fixed(self, st, cal, te, tg):
        A = cal[: len(cal) // 2]
        s = self.p[self.ar, st]; Lw = self.L[self.ar, st]
        sel = C._answer(tg[0], tg[1], s, Lw, cal if tg[0] == 'risk' else A, cal, te, SGRID)
        return C.row(sel, Lw[te], st[te], self.tok[self.ar, st][te], st.astype(float)[te]), sel

    def family(self, tag, fam, cmax, targets, Bs_, add, strike='sc'):
        for r in range(len(self.perms)):
            Cidx, cal, te = self.splits(r)
            for Bq in Bs_:
                oo = {tg: v[0] for tg, v in solve_all(fam, self.p, self.y, Bq, targets, Cidx).items()} if strike == 'sc' else \
                    {tg: fam.idx((1 - tg[1]) if tg[0] == 'cs' else 0.5) for tg in targets}
                for tg in targets:
                    add(f'{tag}|{tg[0]}|{tg[1]}|B{Bq}', self.cert(*fam.prep(oo[tg]), cmax, cal, te, Bq, tg)[0])
                    if strike == 'sc':
                        add(f'{tag}|strike|{tg[0]}|{tg[1]}|B{Bq}', [float(fam.thetas[oo[tg]])])

    def curve(self, tag, S, cmax, targets, Bs_, add):
        cst = fpi.path_cost(S, self.cost); cenv = C.env_cols(cst)
        for r in range(len(self.perms)):
            _, cal, te = self.splits(r)
            for Bq in Bs_:
                for tg in targets:
                    add(f'{tag}|{tg[0]}|{tg[1]}|B{Bq}', self.cert(S, cst, cenv, cmax, cal, te, Bq, tg)[0])

    def mix(self, tag, T, targets, Bs_, add):
        for r in range(len(self.perms)):
            _, cal, te = self.splits(r)
            for Bq in Bs_:
                st = C.mix_stops(self.N, Bq, T)
                for tg in targets:
                    add(f'{tag}|{tg[0]}|{tg[1]}|B{Bq}', self.cert_fixed(st, cal, te, tg)[0])


def workloads(grp, seed, hard=2, shares=(0.0, 0.2, 0.4, 0.6, 0.8, 1.0)):
    """Evaluation workloads of a mixed unit (indices into the evaluation pool): every benchmark alone, and
    compositions of a common size in which the benchmark `hard` holds a given share and the others split the rest
    evenly.  Questions are drawn without replacement."""
    G = int(grp.max()) + 1
    rng = np.random.RandomState(4100 + seed)
    pools = [rng.permutation(np.where(grp == g)[0]) for g in range(G)]
    out = {f'g{g}': np.sort(pools[g]) for g in range(G)}
    n = min(len(q) for q in pools)
    for h in shares:
        nh = int(round(n * h)); rest = n - nh
        take = [nh if g == hard else 0 for g in range(G)]
        others = [g for g in range(G) if g != hard]
        for i, g in enumerate(others):
            take[g] = rest // len(others) + (1 if i < rest % len(others) else 0)
        out[f'h{int(round(100 * h)):03d}'] = np.sort(np.concatenate([pools[g][: take[g]] for g in range(G)]))
    return out


def flow_rows(st, sel, wrong, grp, G):
    """[G + 1, 6] per benchmark and overall (last row): share of the questions, mean reads, certified coverage, error
    among the answered questions, share of all reads, share of the group's reads spent on unanswered questions."""
    out = np.full((G + 1, 6), np.nan, np.float64)
    tot = max(float(st.sum()), 1e-9)
    for g in range(G + 1):
        m = np.ones(len(st), bool) if g == G else grp == g
        if not m.any():
            continue
        rd = float(st[m].sum())
        out[g] = [m.mean(), st[m].mean(), sel[m].mean(), (sel[m] * wrong[m]).sum() / max(sel[m].sum(), 1), rd / tot,
                  float(st[m][~sel[m]].sum()) / max(rd, 1e-9)]
    return out


def seq_certificate(p, Lm, cost, Ud, cal, te, Bq, tg, T, w):
    """Valid per-round certificates (baseline).  The level of the target is split across the rounds with weights w
    (sum 1): at round t the questions still active are answered when their score clears a threshold certified at level
    w_t * level on the active calibration questions (conformal risk control for a risk target, conformal selection for
    a batch error target); unanswered questions read on up to a depth cap, and the cap (dithered between two depths)
    is fixed by the budget certificate.  By the union bound the released answers meet the target.
    Returns [risk, coverage, mean reads, error among answered]."""
    A, Bh = cal[: len(cal) // 2], cal[len(cal) // 2:]
    N = len(p); ar = np.arange(N)
    act = np.ones(N, bool); cross = np.full(N, T + 1)
    for t in range(T + 1):
        lev = tg[1] * w[t]
        s = np.where(act, p[:, t], -np.inf)
        if tg[0] == 'risk':
            tau = C.crc_strike(s, Lm[:, t], A, SGRID, lev)
        else:
            ta = te[act[te]]
            sel = C.conf_select(s, Lm[:, t], A[act[A]], ta, lev) if len(ta) else np.zeros(0, bool)
            tau = float(s[ta][sel].min()) if sel.any() else np.inf
        hit = act & (p[:, t] >= tau)
        cross[hit] = t; act &= ~hit
    S = np.stack([np.minimum(cross, T - j) for j in range(T + 1)] + [np.zeros(N, np.int64)], 1)
    cenv = C.env_cols(fpi.path_cost(S, cost))
    j, om = C.budget_index_env(S, cenv, Ud, Bh, Bq, T, cost_is_envelope=True)
    st, ab, col = C.deployed(S, Ud, j, om, True)
    ans = (~ab) & (cross <= T - col)
    wrong = Lm[ar, st]
    return [float(np.mean(ans[te] * wrong[te])), float(ans[te].mean()), float(st[te].mean()),
            float((ans[te] * wrong[te]).sum() / max(ans[te].sum(), 1))]


def run(args):
    t0 = time.time()
    ds, sd = args.ds, args.seed
    log = lambda m: print(ds, args.rtag, args.fresh_tag, m, f'{time.time() - t0:.0f}s', flush=True)
    parts = set(args.parts.split(','))
    z, H, rows, extra = load_unit(args.traj, ds, args.layer, args.fresh_tag)
    y = z['y'].astype(np.float32); split = z['split']
    tr, ev = np.where(split == 0)[0], np.where(split == 1)[0]
    grp = (np.asarray(z['grp']) if 'grp' in z and len(z['grp']) == len(y) else np.zeros(len(y), np.int8))[ev]
    T1 = y.shape[1]; T = T1 - 1
    tok_all = np.cumsum(np.array([[s['n_prompt_tok'] + s['n_gen_tok'] for s in r['steps']] for r in rows], float), 1)
    tokmean = float(tok_all[tr, -1].mean())
    cost_ret = np.repeat(np.arange(T1, dtype=np.float32)[None], len(y), 0)
    tok = tok_all[ev]; L = 1 - y[ev]; N = len(ev); ar = np.arange(N); yev = y[ev]
    cost_ev, cost_tr = cost_ret[ev], cost_ret[tr]
    dc_ret = np.ones(T, np.float32)
    Bs = budgets(T)
    rng = np.random.RandomState(123 + sd)
    perms = [rng.permutation(N) for _ in range(args.R)]
    Ud = np.random.RandomState(999).rand(N)
    out = {'meta|T': np.array([T]), 'meta|N': np.array([N, len(tr)]), 'meta|tok': np.array([tokmean, float(tok[:, -1].mean())])}
    TG = [('cs', q) for q in QS] + [('risk', 0.15)]
    A_ = argparse.Namespace(pca=args.pca, seed=sd, K=args.K, vep=args.vep, fep=args.fep, cells=args.cells)
    rec = {}

    def add(key, row):
        rec.setdefault(key, []).append(row)

    def save():
        for k, v in rec.items():
            out[k] = np.array(v, np.float32)
        np.savez_compressed(os.path.join(args.out, f'{ds}{args.rtag}{args.fresh_tag}_s{sd}_r21.npz'), **out)

    def cert_split(S, cst, cenv, cmax, p_ev, Lev, cal, te, B, tg, tokc=None):
        A, Bh = cal[: len(cal) // 2], cal[len(cal) // 2:]
        j, om = C.budget_index_env(S, cenv, Ud, Bh, B, cmax, cost_is_envelope=True)
        st, ab, col = C.deployed(S, Ud, j, om, True)
        s = np.where(ab, -np.inf, p_ev[ar, st]); Lw = Lev[ar, st]
        sel = C._answer(tg[0], tg[1], s, Lw, A, cal, te, SGRID)
        tk = tok if tokc is None else tokc
        return C.row(sel, Lw[te], st[te], np.where(ab, 0.0, tk[ar, st])[te], cst[ar, col][te]), st, ab

    def cert_fixed_split(st, p_ev, Lev, cal, te, tg):
        A = cal[: len(cal) // 2]
        s = p_ev[ar, st]; Lw = Lev[ar, st]
        sel = C._answer(tg[0], tg[1], s, Lw, cal if tg[0] == 'risk' else A, cal, te, SGRID)
        return C.row(sel, Lw[te], st[te], tok[ar, st][te], st.astype(float)[te])

    def splits(r):
        pm = perms[r]
        return pm[: N // 4], pm[N // 4: N // 2], pm[N // 2:], pm[: N // 2]        # C, cal (A|B), test, cal_P1

    def bkey(B):
        return f'f{B[2]}' if isinstance(B, tuple) else f'B{B}'

    # ------------------------------------------------------------------ strike-indexed families -------------------
    def run_family(tag, fam, p_ev, yev_, cost, cmax, targets, Bs_, strike='sc', Lev=None, nsplit=None, store_curve=False,
                   fixed_theta=None, tokc=None, keep_theta=True):
        """Certify a strike-indexed family under the three-way protocol for every split.  strike in {'sc' (exhaustive
        labelled gate on C), 'free' (label-free gate on C), 'th1q' (fixed), 'fixed' (fixed_theta[(tg, B)]),
        'oracle' (labelled gate on the whole pool)}.  Returns {(r, tg, B): theta}."""
        Lev = L if Lev is None else Lev
        stats, thetas_used, whole = {}, {}, {}
        for r in range(args.R if nsplit is None else nsplit):
            Cidx, cal, te, _ = splits(r)
            for Bq in Bs_:
                Bg, Bev = (Bq[1], Bq[1]) if isinstance(Bq, tuple) else (Bq, Bq)
                if strike in ('sc', 'free'):
                    sol = solve_all(fam, p_ev, yev_, Bg, targets, Cidx, free=(strike == 'free'))
                elif strike == 'oracle':
                    if Bq not in whole:
                        whole[Bq] = solve_all(fam, p_ev, yev_, Bg, targets, ar)
                    sol = whole[Bq]
                for tg in targets:
                    res, reached = 0.0, True
                    if strike == 'th1q':
                        th = (1 - tg[1]) if tg[0] == 'cs' else 0.5; o = fam.idx(th)
                    elif strike == 'acc':
                        th = 0.0; o = fam.idx(0.0)
                    elif strike == 'fixed':
                        th = fixed_theta[(tg, Bq)]; o = fam.idx(th)
                    else:
                        o, th, res, reached = sol[tg]
                    thetas_used[(r, tg, Bq)] = th
                    stats.setdefault((tg, Bq), []).append([th, res, float(not reached)])
                    row, _, _ = cert_split(*fam.prep(o), cmax, p_ev, Lev, cal, te, Bev, tg, tokc=tokc)
                    add(f'{tag}|{tg[0]}|{tg[1]}|{bkey(Bq)}', row)
                    if store_curve and r == 0:
                        out[f'curve|{tag}|{tg[0]}|{tg[1]}|{bkey(Bq)}'] = fam.SS[o]
        if keep_theta:
            for (tg, Bq), v in stats.items():
                v = np.array(v)
                out[f'theta|{tag}|{tg[0]}|{tg[1]}|{bkey(Bq)}'] = np.array([v[:, 0].mean(), v[:, 0].std(), v[:, 1].mean(), v[:, 2].mean()], np.float32)
                out[f'strike|{tag}|{tg[0]}|{tg[1]}|{bkey(Bq)}'] = v[:, 0].astype(np.float32)
        return thetas_used

    def run_fixed_curve(tag, S, p_ev, cost, cmax, targets, Bs_, Lev=None, prefix='', tokc=None):
        Lev = L if Lev is None else Lev
        cst = fpi.path_cost(S, cost); cenv = C.env_cols(cst)
        for r in range(args.R):
            Cidx, cal, te, _ = splits(r)
            for Bq in Bs_:
                Bev = Bq[1] if isinstance(Bq, tuple) else Bq
                for tg in targets:
                    row, _, _ = cert_split(S, cst, cenv, cmax, p_ev, Lev, cal, te, Bev, tg, tokc=tokc)
                    add(f'{prefix}{tag}|{tg[0]}|{tg[1]}|{bkey(Bq)}', row)

    def run_mix(p_ev, targets, Bs_, Lev=None, prefix=''):
        Lev = L if Lev is None else Lev
        for r in range(args.R):
            Cidx, cal, te, _ = splits(r)
            for Bq in Bs_:
                st = C.mix_stops(N, Bq, T)
                for tg in targets:
                    add(f'{prefix}mix|{tg[0]}|{tg[1]}|B{Bq}', cert_fixed_split(st, p_ev, Lev, cal, te, tg))

    # ------------------------------------------------------------------ the main unit ---------------------------------
    t1 = time.time()
    main = Unit(z, H, y, tr, ev, 'state', A_, log)
    out['meta|train_time|ours'] = np.array([time.time() - t1])
    p_ev, p_tr = main.p_ev, main.p_tr
    hp = lambda idx: None if main.Hp is None else main.Hp[idx]
    auc = [auroc(yev[:, t], p_ev[:, t]) for t in range(T1)]
    out['meta|auc'] = np.array(auc); out['meta|acc'] = yev.mean(0); out['meta|oracle'] = np.array([yev.max(1).mean()])
    log('AUROC per round ' + ' '.join(f'{a:.3f}' for a in auc))
    Pp_ev = B.pop_chain(main.lat, N)
    # (path, predictive) NLL of the evaluation belief paths: learned chain, then population chain
    out['meta|path_nll'] = np.array(list(B.path_nll(main.P_ev, main.b_ev)) + list(B.path_nll(Pp_ev, main.b_ev)))
    log('path NLL (path, predictive): learned %.4f %.4f | population %.4f %.4f' % tuple(out['meta|path_nll']))
    out['meta|cells'] = np.array([args.cells]); out['meta|edges'] = main.lat.edges; out['meta|mid'] = main.lat.mid
    bev = main.b_ev
    out['meta|reliability_ev'] = np.array([[yev[bev[:, t] == k, t].mean() if (bev[:, t] == k).sum() >= 20 else np.nan for k in range(args.cells)] for t in range(T1)])
    t1 = time.time()
    fam_main, kap = main.family(cost_ev, dc_ret)
    out['meta|plan_time|ours'] = np.array([(time.time() - t1) / len(THETAS)])
    out['mono|ours'] = np.array([min(B.nested_check(S) for S in fam_main.SS)])
    log(f'main family built, nested {out["mono|ours"][0]:.3f}')

    def published(y_lab):
        """The eight published stopping rules, trained and scored on y_lab (HALT needs no training)."""
        from baselines import confidence_curves, probing_curve, adaptive_rag_curve, stoprag_curve
        curves = confidence_curves(z, tr, ev, T)
        need = np.where(y_lab.max(1) > 0, y_lab.argmax(1), T)
        x0 = np.concatenate([main.Hp[:, 0], main.X[:, 0]], -1) if main.Hp is not None else main.X[:, 0]
        curves['adapt'] = adaptive_rag_curve(x0[tr], need[tr], x0[ev], T, sd, args.M, T1)
        curves['probe'] = probing_curve(hp(tr), y_lab[tr], hp(ev), main.V.fold, T, seed=sd, dev=fpi.DEV)
        curves['stoprag'] = stoprag_curve(hp(tr), main.X[tr], y_lab[tr], hp(ev), main.X[ev], T, sd, args.M, dev=fpi.DEV)
        if extra.get('halt') is not None:
            hs = np.nan_to_num(extra['halt'][ev].astype(float), nan=0.0)
            curves['halt'] = C.first_cross_path(hs, np.r_[1.01, np.linspace(1.0, 0.0, 160)], T)
        return curves

    base_curves = {}
    ours_theta = None
    if parts & {'main', 'boot', 'lab', 'tok', 'oracle', 'p1', 'wl', 'flow', 'plain', 'reach', 'valid', 'cap'}:
        base_curves = published(y)
        log('baselines trained: ' + ','.join(base_curves))
    if parts & {'main', 'boot', 'lab', 'oracle'}:
        ours_theta = run_family('ours', fam_main, p_ev, yev, cost_ev, T, TG, Bs, strike='sc', store_curve=True)
        log('ours certified; strikes q=.15: ' + ' '.join(f"B{B}:{out[f'theta|ours|cs|0.15|B{B}'][0]:.2f}(res {out[f'theta|ours|cs|0.15|B{B}'][2]:.3f})" for B in Bs))
        for k, S in base_curves.items():
            run_fixed_curve(k, S, p_ev, cost_ev, T, TG, Bs)
        run_mix(p_ev, TG, Bs)
        log('main certified')
        save()
    # ------------------------------------------------------------------ standard protocol (supplement) ---------------
    if 'p1' in parts and base_curves:
        for r in range(args.R):
            _, _, te, cal1 = splits(r)
            for Bq in Bs:
                for tg in TG:
                    add(f'P1|mix|{tg[0]}|{tg[1]}|B{Bq}', cert_fixed_split(C.mix_stops(N, Bq, T), p_ev, L, cal1, te, tg))
                    o = fam_main.idx((1 - tg[1]) if tg[0] == 'cs' else 0.5)
                    add(f'P1|ours_th1q|{tg[0]}|{tg[1]}|B{Bq}', cert_split(*fam_main.prep(o), T, p_ev, L, cal1, te, Bq, tg)[0])
                    for k, S in base_curves.items():
                        cst = fpi.path_cost(S, cost_ev)
                        add(f'P1|{k}|{tg[0]}|{tg[1]}|B{Bq}', cert_split(S, cst, C.env_cols(cst), T, p_ev, L, cal1, te, Bq, tg)[0])
        log('P1 protocol'); save()
    # ------------------------------------------------------------------ label hindsight and Lagrangian reward ------
    if 'oracle' in parts and ours_theta is not None:
        hcache = {}
        bprep = {}
        for k, S in base_curves.items():
            cst_k = fpi.path_cost(S, cost_ev); bprep[k] = (S, cst_k, C.env_cols(cst_k))
        for r in range(args.R):
            _, cal, te, _ = splits(r)
            for Bq in Bs:
                for tg in TG:
                    th = ours_theta[(r, tg, Bq)]
                    if th not in hcache:
                        Sh = B.hindsight(p_ev, yev, cost_ev, th, kap, T); csth = fpi.path_cost(Sh, cost_ev)
                        hcache[th] = (Sh, csth, C.env_cols(csth))
                        if len(hcache) > 110:
                            hcache.pop(next(iter(hcache)))
                    Sh, csth, cenvh = hcache[th]
                    row, sth, abh = cert_split(Sh, csth, cenvh, T, p_ev, L, cal, te, Bq, tg)
                    add(f'oracle|{tg[0]}|{tg[1]}|B{Bq}', row)
                    o = fam_main.idx(th)
                    _, sto, abo = cert_split(*fam_main.prep(o), T, p_ev, L, cal, te, Bq, tg)
                    for nm, st_, ab_ in (('ours', sto, abo), ('oracle', sth, abh)):
                        rw = np.where(ab_, 0.0, B.reward(p_ev, yev, st_, th))
                        add(f'lag|{nm}|{tg[0]}|{tg[1]}|B{Bq}', [float(rw[te].mean()), float(cost_ev[ar, st_][te].mean()), float(np.mean(~ab_[te]))])
                    add(f'lag|bound|{tg[0]}|{tg[1]}|B{Bq}', [B.dual_bound(p_ev[te], yev[te], cost_ev[te], th, Bq, kap, T), Bq, 1.0])
                    # the same reward, at the same strike and certified budget, for every published rule
                    for k, (S, cst_k, cenv_k) in bprep.items():
                        _, stb, abb = cert_split(S, cst_k, cenv_k, T, p_ev, L, cal, te, Bq, tg)
                        rw = np.where(abb, 0.0, B.reward(p_ev, yev, stb, th))
                        add(f'lag|{k}|{tg[0]}|{tg[1]}|B{Bq}', [float(rw[te].mean()), float(cost_ev[ar, stb][te].mean()), float(np.mean(~abb[te]))])
                    stm = C.mix_stops(N, Bq, T)
                    add(f'lag|mix|{tg[0]}|{tg[1]}|B{Bq}', [float(B.reward(p_ev, yev, stm, th)[te].mean()), float(stm[te].mean()), 1.0])
        log('hindsight'); save()
    # ------------------------------------------------------------------ RQ1: repeated looks ------------------------
    if 'peek' in parts:
        cal_idx = [perms[r][N // 4: N // 2] for r in range(args.R)]; tests = [perms[r][N // 2:] for r in range(args.R)]
        for a, v in peek_inflation(p_ev, L, cal_idx, tests, ALPHAS, T).items():
            out[f'peek|{a}'] = v
        log('peek')
    # ------------------------------------------------------------------ paired bootstrap ---------------------------
    if 'boot' in parts and base_curves:
        brng = np.random.RandomState(9000 + sd)
        bt = {}
        Pb = {k: (S, C.env_cols(fpi.path_cost(S, cost_ev))) for k, S in base_curves.items()}
        for b in range(args.nboot):
            idx = brng.randint(0, N, N)
            Cidx, cal, te = idx[: N // 4], idx[N // 4: N // 2], idx[N // 2:]
            A, Bh = cal[: len(cal) // 2], cal[len(cal) // 2:]
            for Bq in Bs:
                for q in QS:
                    tg = ('cs', q)
                    o, th, _, _, _ = fam_main.solve(p_ev, yev, cost_ev, Bq, tg, Cidx)
                    pols = {'ours': (fam_main.SS[o], fam_main.prep(o)[2])}
                    pols.update(Pb)
                    for k, (S, cenv) in pols.items():
                        j, om = C.budget_index_env(S, cenv, Ud, Bh, Bq, T, cost_is_envelope=True)
                        st, ab, _ = C.deployed(S, Ud, j, om, True)
                        s = np.where(ab, -np.inf, p_ev[ar, st])
                        sel = C.conf_select(s, L[ar, st], A, te, q)
                        bt.setdefault(f'{k}|cs|{q}|B{Bq}', []).append([sel.mean(), (sel * (1 - L[ar, st][te])).mean()])
                    st = C.mix_stops(N, Bq, T)
                    sel = C.conf_select(p_ev[ar, st], L[ar, st], A, te, q)
                    bt.setdefault(f'mix|cs|{q}|B{Bq}', []).append([sel.mean(), (sel * (1 - L[ar, st][te])).mean()])
        for k, v in bt.items():
            out[f'boot|{k}'] = np.array(v)
        log('bootstrap'); save()
    # ------------------------------------------------------------------ calibration size --------------------------
    if 'calsize' in parts:
        for r in range(args.R):
            pm = perms[r]
            Cidx, te = pm[: N // 4], pm[N // 2:]
            o, th, _, _, _ = fam_main.solve(p_ev, yev, cost_ev, 1.0, ('cs', 0.15), Cidx)
            for ncal in sorted({n for n in (100, 200, 400, 800) if n <= N // 4} | {N // 4}):
                cal = pm[N // 4: N // 4 + ncal]
                row, _, _ = cert_split(*fam_main.prep(o), T, p_ev, L, cal, te, 1.0, ('cs', 0.15))
                add(f'calsize|{ncal}', [row[1], row[4], row[2]])
        log('calsize'); save()
    # ------------------------------------------------------------------ strike diagnostics -------------------------
    if 'sdiag' in parts:
        for tg, Bq in ((('cs', 0.15), 1.0), (('cs', 0.15), 2.0), (('risk', 0.15), 1.0)):
            for r in range(args.R):
                Cidx, _, _, _ = splits(r)
                o, th, res, reached, gates = fam_main.solve(p_ev, yev, cost_ev, Bq, tg, Cidx)
                ncross = int(np.sum(np.abs(THETAS - gates) <= 0.01))
                kms = [fam_main.km(gates, s0) for s0 in (0.1, 0.5, 0.9)]
                add(f'sdiag|{tg[0]}|{tg[1]}|B{Bq}', [th, res, ncross, kms[0][0], kms[1][0], kms[2][0], np.mean([k[1] for k in kms]), gates[o], float(not reached)])
                if r == 0:
                    out[f'sdiag_curve|{tg[0]}|{tg[1]}|B{Bq}'] = gates.astype(np.float32)
        log('strike diagnostics'); save()
    # ------------------------------------------------------------------ labels -------------------------------------
    if 'lab' in parts:
        ans = np.array([[norm_ans(s['answer']) for s in r['steps']] for r in rows], dtype=object)
        nonans = np.vectorize(lambda a: (not a) or bool(NONANS.search(a)))(ans)
        labs = {'em': z['yem'].astype(np.float32)}
        if extra.get('sem') is not None:
            sem = extra['sem'].astype(float)
            if sem.shape == y.shape:
                sem = np.where(np.isfinite(sem), sem, y)
                labs['judge'] = ((sem > 0.5) & ~nonans).astype(np.float32)
                out['meta|judge_agree'] = np.array([float(np.mean(labs['judge'][ev] == (yev > 0.5))), float(np.mean(nonans[ev] & (sem[ev] > 0.5)))])
        out['meta|nonans_rate'] = np.array([float(nonans[ev].mean()), float(np.mean(nonans[ev] & (yev > 0.5)))])
        for ln, yl in labs.items():
            Ll_ev = 1 - yl[ev]
            ul = Unit(z, H, yl, tr, ev, 'state', A_, log, obs=main)
            out[f'meta|auc_{ln}'] = np.array([auroc(yl[ev][:, t], ul.p_ev[:, t]) for t in range(T1)])
            out[f'meta|acc_{ln}'] = yl[ev].mean(0)
            fam_l, _ = ul.family(cost_ev, dc_ret)
            run_family(f'lab|{ln}|ours', fam_l, ul.p_ev, yl[ev], cost_ev, T, TG, Bs[:3], Lev=Ll_ev)
            for k, S in published(yl).items():
                run_fixed_curve(k, S, ul.p_ev, cost_ev, T, TG, Bs[:3], Lev=Ll_ev, prefix=f'lab|{ln}|')
            run_mix(ul.p_ev, TG, Bs[:3], Lev=Ll_ev, prefix=f'lab|{ln}|')
            del fam_l
            log(f'labels {ln}'); save()
    # ------------------------------------------------------------------ contract library: planning cost ----------
    if 'contracts' in parts:
        t1 = time.time()
        fam_main.solve(p_ev, yev, cost_ev, 1.0, ('cs', 0.15), perms[0][: N // 4])
        out['meta|contract_time|ours_strike'] = np.array([time.time() - t1])
        out['meta|contract_time|ours_price'] = out['meta|plan_time|ours']
        log('contracts'); save()
    # ------------------------------------------------------------------ ablations ----------------------------------
    B3 = Bs[:3]
    if parts & {'abl', 'sabl', 'abl0'}:
        for Bq in B3:
            for t_ in TG:
                if f'ours|{t_[0]}|{t_[1]}|B{Bq}' in rec:
                    rec[f'abl|full|{t_[0]}|{t_[1]}|B{Bq}'] = rec[f'ours|{t_[0]}|{t_[1]}|B{Bq}']
        if 'main' not in parts:
            run_family('abl|full', fam_main, p_ev, yev, cost_ev, T, TG, B3, strike='sc')
    if parts & {'abl', 'sabl'}:
        # ---- the strike: 1 - q, solved on the training pool, label-free gate, all labels, and strike 0 (accuracy)
        run_family('abl|th1q', fam_main, p_ev, yev, cost_ev, T, TG, B3, strike='th1q')
        run_family('abl|acc', fam_main, p_ev, yev, cost_ev, T, TG, B3, strike='acc', keep_theta=False)
        fam_tr = B.Family(B.plan(main.P_tr, main.b_tr, p_tr, main.call, THETAS, kap, dc_ret), cost_tr)
        sctr = {}
        for tg in TG:
            for Bq in B3:
                sctr[(tg, Bq)] = fam_tr.solve(p_tr, y[tr], cost_tr, Bq, tg, np.arange(len(tr)))[1]
                out[f'theta|abl|sctr|{tg[0]}|{tg[1]}|B{Bq}'] = np.array([sctr[(tg, Bq)], 0, 0, 0], np.float32)
        del fam_tr
        run_family('abl|sctr', fam_main, p_ev, yev, cost_ev, T, TG, B3, strike='fixed', fixed_theta=sctr, keep_theta=False)
        run_family('abl|free', fam_main, p_ev, yev, cost_ev, T, TG, B3, strike='free')
        run_family('abl|oracle', fam_main, p_ev, yev, cost_ev, T, TG, B3, strike='oracle')
        log('abl strike rules'); save()
    if parts & {'abl', 'abl0'}:
        fkw = dict(epochs=args.fep) if args.fep else {}
        # ---- the law: population chain; the field as a correction to the population kernel; path likelihood only
        fam, _ = main.family(cost_ev, dc_ret, P=Pp_ev)
        run_family('abl|pop', fam, p_ev, yev, cost_ev, T, TG, B3); del fam
        # ---- the payoff: digital instead of the call
        fam, _ = main.family(cost_ev, dc_ret, payoff='digital')
        run_family('abl|digit', fam, p_ev, yev, cost_ev, T, TG, B3); del fam
        # ---- the price: one read ahead; deterministic horizons on the same chain
        fam, _ = main.family(cost_ev, dc_ret, myopic=True)
        run_family('abl|myopic', fam, p_ev, yev, cost_ev, T, TG, B3); del fam
        fam, _ = main.family(cost_ev, dc_ret, dh=True)
        run_family('abl|dh', fam, p_ev, yev, cost_ev, T, TG, B3); del fam
        log('abl pop / digital / myopic / dh'); save()
        for nm, kw in (('anch', dict(anchor=True)), ('tf', dict(predictive=False))):
            cpn = B.CrossPath(hp(tr), main.X[tr], p_tr, main.lat, main.V.fold, seed=sd, **kw, **fkw)
            Pn = cpn(hp(ev), main.X[ev], p_ev)
            out[f'meta|path_nll|{nm}'] = np.array(list(B.path_nll(Pn, main.b_ev)))
            fam, _ = main.family(cost_ev, dc_ret, P=Pn)
            run_family(f'abl|{nm}', fam, p_ev, yev, cost_ev, T, TG, B3); del fam, cpn, Pn
        log('abl population anchor / path likelihood only'); save()
        # ---- the kernel field reads the belief path only
        cpn = B.CrossPath(None, main.X[tr][..., :0], p_tr, main.lat, main.V.fold, seed=sd, **fkw)
        Pn = cpn(None, main.X[ev][..., :0], p_ev)
        out['meta|path_nll|ctrlp'] = np.array(list(B.path_nll(Pn, main.b_ev)))
        fam, _ = main.family(cost_ev, dc_ret, P=Pn)
        run_family('abl|ctrlp', fam, p_ev, yev, cost_ev, T, TG, B3); del fam, cpn, Pn
        log('abl belief-only law'); save()
        # ---- label reliabilities in the payoff: the three-head model (kernel, marginal and reliability heads)
        import bpm20rel as BR
        t1 = time.time()
        cpr = BR.CrossPath(hp(tr), main.X[tr], p_tr, y[tr], main.lat, main.V.fold, seed=sd, **fkw)
        Pr, Ar = cpr(hp(ev), main.X[ev], p_ev)
        out['meta|train_time|rel'] = np.array([time.time() - t1])
        fam = B.Family(BR.plan(Pr, Ar, main.b_ev, p_ev, main.frac, THETAS, kap, dc_ret), cost_ev)
        run_family('abl|rel', fam, p_ev, yev, cost_ev, T, TG, B3); del fam, cpr, Pr, Ar
        log('abl label reliabilities'); save()
        # ---- multi-horizon marginal forecaster + belief payoff (deterministic horizons)
        t1 = time.time()
        fcs = Lt.CrossForecast(hp(tr), main.X[tr], p_tr, y[tr], main.lat, main.V.fold, seed=sd)
        F_ev = fcs(hp(ev), main.X[ev], p_ev)
        out['meta|train_time|marg'] = np.array([time.time() - t1])
        marg_curves = lambda lat_, p_: np.stack([Lt.index_curve(Lt.fair_price(lat_, F_ev, main.b_ev, p_, th, th, cost_ev, payoff='belief'), GRID_OLD, T) for th in THETAS]).astype(np.int8)
        fam = B.Family(marg_curves(main.lat, p_ev), cost_ev)
        run_family('abl|marg', fam, p_ev, yev, cost_ev, T, TG, B3); del fam, fcs, F_ev
        log('abl marg'); save()
    if 'abl' in parts:
        # ---- Longstaff-Schwartz value regression on the same inputs
        t1 = time.time()
        pol = fpi.CrossPolicy(hp(tr), main.X[tr], p_tr, y[tr], cost_tr, main.V.fold, kind='value', seed=sd)
        kaps = pol.ps * np.concatenate([-np.geomspace(0.5, 1e-3, 40), [0.0], np.geomspace(1e-3, 1.0, 120)])
        SSv = np.stack([fpi.value_path(pol(hp(ev), main.X[ev], p_ev, cost_ev, [th], kaps)[:, 0], p_ev, th, kaps, T) for th in THETAS]).astype(np.int8)
        out['meta|train_time|lsmc'] = np.array([time.time() - t1])
        fam = B.Family(SSv, cost_ev)
        run_family('abl|lsmc', fam, p_ev, yev, cost_ev, T, TG, B3); del fam, pol, SSv
        log('abl lsmc'); save()
        # ---- monotone distortions of the verifier score (the cells and the chain are unchanged; the payoff is not)
        if 'inv' in parts:
            for nm, f in (('sq', lambda v: v ** 2), ('sqrt', lambda v: np.sqrt(v))):
                pf_tr, pf_ev = f(p_tr).astype(np.float32), f(p_ev).astype(np.float32)
                fam, _ = main.family(cost_ev, dc_ret, p=pf_ev, U=B.call_table(pf_tr, main.lat, cells=main.b_tr))
                run_family(f'inv|{nm}|ours', fam, pf_ev, yev, cost_ev, T, TG, B3); del fam
                run_mix(pf_ev, TG, B3, prefix=f'inv|{nm}|')
            log('score distortions'); save()
        # ---- reader state 2 x 2
        bb = Unit(z, H, y, tr, ev, 'blackbox', A_, log)
        fam, _ = bb.family(cost_ev, dc_ret)
        run_family('abl|blackbox', fam, bb.p_ev, yev, cost_ev, T, TG, B3); del fam
        run_mix(bb.p_ev, TG, B3, prefix='abl|blackbox|')
        if main.Hp is not None:
            cb = Unit(z, H, y, tr, ev, 'state', A_, log, base=main, ctrl=bb)
            fam, _ = cb.family(cost_ev, dc_ret)
            run_family('abl|ctrlbb', fam, p_ev, yev, cost_ev, T, TG, B3); del fam, cb
            vb = Unit(z, H, y, tr, ev, 'blackbox', A_, log, base=bb, ctrl=main)
            fam, _ = vb.family(cost_ev, dc_ret)
            run_family('abl|verbb', fam, bb.p_ev, yev, cost_ev, T, TG, B3); del fam, vb
        del bb
        log('ablations'); save()
    # ------------------------------------------------------------------ token budgets ------------------------------
    if 'tok' in parts and base_curves:
        ctok = (tok_all / tokmean).astype(np.float32)
        tmax = float(4096 * T1)
        if np.any(np.diff(tok_all, axis=1, prepend=0) > 4096):
            raise ValueError('Recorded round exceeds the declared 4096-token context bound')
        tbud = [f * tokmean for f in TOKF]
        out['tok|budgets'] = np.array(tbud)
        tB = [(f, Bt, f) for f, Bt in zip(TOKF, tbud)]
        dc_tok = np.diff(ctok[tr].mean(0)).astype(np.float32)                     # expected tokens of each read, budget units
        fam, _ = main.family(tok, dc_tok)
        run_family('tok|ours', fam, p_ev, yev, tok, tmax, TG, tB); del fam
        fixed = np.concatenate([np.repeat(np.arange(T, -1, -1)[None], N, 0), np.zeros((N, 1), int)], 1)
        for k, S in dict(fixed=fixed, **base_curves).items():
            run_fixed_curve(k, S, p_ev, tok, tmax, TG, tB, prefix='tok|')
        log('token budgets'); save()
    # ------------------------------------------------------------------ deadline -----------------------------------
    if 'dl' in parts:
        for Bq in (1.0, 2.0):
            d = int(np.ceil(Bq)) + 1
            if d >= T:
                continue
            fam, _ = main.family(cost_ev, dc_ret, last=d)
            run_family(f'dl{d}', fam, p_ev, yev, cost_ev, d, TG, (Bq,)); del fam
            fam = B.Family(np.minimum(fam_main.SS, d), cost_ev)
            run_family(f'cap{d}', fam, p_ev, yev, cost_ev, d, TG, (Bq,)); del fam
        log('deadline'); save()
    # ================================================================== round 21 ====================================
    G = int(grp.max()) + 1
    out['meta|grp'] = np.bincount(grp, minlength=G)
    tq = ('cs', 0.15)
    # ------------------------------------------------------------------ evaluation workloads of a mixed unit --------
    if 'wl' in parts and base_curves and G > 1:
        for wn, sub in workloads(grp, sd).items():
            w = WL(sub, p_ev, yev, cost_ev, tok, sd, args.Rw)
            out[f'wl|{wn}|n'] = np.bincount(grp[sub], minlength=G)
            fam_w = B.Family(fam_main.SS[:, sub], cost_ev[sub])
            w.family(f'wl|{wn}|ours', fam_w, T, TG, B3, add)
            w.family(f'wl|{wn}|ours1q', fam_w, T, TG, B3, add, strike='th1q')
            for k, S in base_curves.items():
                w.curve(f'wl|{wn}|{k}', S[sub], T, TG, B3, add)
            w.mix(f'wl|{wn}|mix', T, TG, B3, add)
            del fam_w
            log(f'workload {wn} n={len(sub)} ' + ' '.join(f"{k}:{100 * np.mean(np.array(rec[f'wl|{wn}|{k}|cs|0.15|B1.0'])[:, 1]):.1f}" for k in ('ours', 'ours1q', 'mix', 'stoprag', 'ircot')))
            save()
    # ------------------------------------------------------------------ published rules with a depth cap -------------
    if 'cap' in parts and base_curves:
        # every published loop has a maximum number of retrieval steps; here each rule is run with every cap d < T
        # (d = T is the rule of the main part), so that a table can give each rule its best cap
        for d in range(1, T):
            for k, S in base_curves.items():
                Sd = np.minimum(S, d).astype(S.dtype)
                cst = fpi.path_cost(Sd, cost_ev); cenv = C.env_cols(cst)
                for r in range(args.R):
                    _, cal, te, _ = splits(r)
                    for Bq in Bs:
                        for tg in TG:
                            add(f'cap{d}|{k}|{tg[0]}|{tg[1]}|B{Bq}', cert_split(Sd, cst, cenv, d, p_ev, L, cal, te, Bq, tg)[0])
        log('capped rules at q=.15 B=1: ' + ' '.join(f"{k}:" + '/'.join(f"{100 * np.mean(np.array(rec[f'cap{d}|{k}|cs|0.15|B1.0'])[:, 1]):.1f}" for d in range(1, T)) for k in base_curves))
        save()
    # ------------------------------------------------------------------ where the reads go ---------------------------
    if 'flow' in parts and base_curves:
        w = WL(ar, p_ev, yev, cost_ev, tok, sd, min(args.R, args.Rw))
        prep = {k: (S, fpi.path_cost(S, cost_ev)) for k, S in base_curves.items()}
        prep = {k: (S, cst, C.env_cols(cst)) for k, (S, cst) in prep.items()}
        for Bq in (1.0, 2.0):
            for r in range(len(w.perms)):
                Cidx, cal, te = w.splits(r)
                o = solve_all(fam_main, p_ev, yev, Bq, [tq], Cidx)[tq][0]
                for k, (S, cst, cenv) in dict(ours=fam_main.prep(o), **prep).items():
                    _, st, ab, sel = w.cert(S, cst, cenv, T, cal, te, Bq, tq)
                    add(f'flow|{k}|B{Bq}', flow_rows(st[te], sel, L[ar, st][te], grp[te], G))
                st = C.mix_stops(N, Bq, T)
                add(f'flow|mix|B{Bq}', flow_rows(st[te], w.cert_fixed(st, cal, te, tq)[1], L[ar, st][te], grp[te], G))
        log('flow: reads on unanswered questions at B=1 ' + ' '.join(f"{k}:{np.mean(np.array(rec[f'flow|{k}|B1.0'])[:, G, 5]):.2f}" for k in ('ours', 'mix', 'stoprag', 'ircot', 'flare')))
        save()
    # ------------------------------------------------------------------ valid per-round certificates (baselines) -----
    if 'vcert' in parts:
        first = np.where(y[tr].max(1) > 0, y[tr].argmax(1), T1)
        ans_at = np.bincount(first, minlength=T1 + 1)[:T1].astype(float) + 1.0     # questions first answerable at a round
        allocs = {'bonf': np.full(T1, 1.0 / T1), 'micp': ans_at / ans_at.sum()}
        out['vcert|alloc|micp'] = allocs['micp']
        for r in range(args.R):
            _, cal, te, _ = splits(r)
            for Bq in Bs:
                for tg in TG:
                    for an, wv in allocs.items():
                        add(f'vcert|{an}|{tg[0]}|{tg[1]}|B{Bq}', seq_certificate(p_ev, L, cost_ev, Ud, cal, te, Bq, tg, T, wv))
        log('per-round certificates: ' + ' '.join(f"{an} B{Bq}:{100 * np.mean(np.array(rec[f'vcert|{an}|cs|0.15|B{Bq}'])[:, 1]):.1f}" for an in allocs for Bq in Bs[:3]))
        save()
    # ------------------------------------------------------------------ accuracy against reads, no certificate -------
    if 'plain' in parts and base_curves:
        ym = {'acc': yev, 'em': z['yem'][ev].astype(np.float32), 'f1': (z['yf1'][ev] if 'yf1' in z else z['yem'][ev]).astype(np.float32)}
        o0 = fam_main.idx(0.0)
        pols = dict(ours=fam_main.prep(o0))
        for k, S in base_curves.items():
            cst = fpi.path_cost(S, cost_ev); pols[k] = (S, cst, C.env_cols(cst))
        Bp = tuple(Bs) + ((float(T),) if float(T) not in Bs else ())
        for r in range(args.R):
            _, cal, te, _ = splits(r)
            Bh = cal[len(cal) // 2:]
            for Bq in Bp:
                for k, (S, cst, cenv) in pols.items():
                    j, om = C.budget_index_env(S, cenv, Ud, Bh, Bq, T, cost_is_envelope=True)
                    st, ab, _ = C.deployed(S, Ud, j, om, True)
                    add(f'plain|{k}|B{Bq}', [ym['acc'][ar, st][te].mean(), ym['em'][ar, st][te].mean(), ym['f1'][ar, st][te].mean(), st[te].mean(), tok[ar, st][te].mean()])
                st = C.mix_stops(N, Bq, T)
                add(f'plain|mix|B{Bq}', [ym['acc'][ar, st][te].mean(), ym['em'][ar, st][te].mean(), ym['f1'][ar, st][te].mean(), st[te].mean(), tok[ar, st][te].mean()])
        out['plain|depth'] = np.array([[ym[m][:, t].mean() for t in range(T1)] for m in ('acc', 'em', 'f1')])
        log('plain accuracy at B=1: ' + ' '.join(f"{k}:{100 * np.mean(np.array(rec[f'plain|{k}|B1.0'])[:, 0]):.1f}" for k in ['ours', 'mix'] + list(base_curves)))
        save()
    # ------------------------------------------------------------------ fine budget grid -----------------------------
    if 'reach' in parts and base_curves:
        Bf = tuple(float(b) for b in np.round(np.arange(0.5, T + 1e-9, 0.25), 2))
        out['reach|budgets'] = np.array(Bf)
        w = WL(ar, p_ev, yev, cost_ev, tok, sd, min(args.R, args.Rw))
        w.family('reach|ours', fam_main, T, [tq], Bf, add)
        for k, S in base_curves.items():
            w.curve(f'reach|{k}', S, T, [tq], Bf, add)
        w.mix('reach|mix', T, [tq], Bf, add)
        log('reach'); save()
    # ------------------------------------------------------------------ i.i.d. resampling of the workload ------------
    if 'valid' in parts and base_curves:
        vr = np.random.RandomState(7000 + sd)
        pos = np.arange(N)
        Cp, calp, tep = pos[: N // 4], pos[N // 4: N // 2], pos[N // 2:]
        A, Bh = calp[: len(calp) // 2], calp[len(calp) // 2:]
        Sb = {k: (S, C.env_cols(fpi.path_cost(S, cost_ev))) for k, S in base_curves.items()}
        for b in range(args.nvalid):
            idx = vr.randint(0, N, N); U = vr.rand(N); tie = vr.rand(N)
            for Bq in (1.0, 2.0):
                oo = solve_all(fam_main, p_ev, yev, Bq, TG, idx[Cp])
                stm = C.mix_stops(N, Bq, T, seed=int(vr.randint(1 << 30)))
                for tg in TG:
                    S, _, cenv = fam_main.prep(oo[tg][0])
                    for k, pc in dict(ours=(S, cenv), **Sb, mix=None).items():
                        if pc is None:
                            st = stm; s = p_ev[idx, st]
                        else:
                            Sp = pc[0][idx]
                            j, om = C.budget_index_env(Sp, pc[1][idx], U, Bh, Bq, T, cost_is_envelope=True)
                            st, ab, _ = C.deployed(Sp, U, j, om, True)
                            s = np.where(ab, -np.inf, p_ev[idx, st])
                        Lw = L[idx, st]
                        sel = C.conf_select(s, Lw, A, tep, tg[1], tie=tie) if tg[0] == 'cs' else s[tep] >= C.crc_strike(s, Lw, A, SGRID, tg[1])
                        add(f'valid|{k}|{tg[0]}|{tg[1]}|B{Bq}', [(sel * Lw[tep]).sum() / max(sel.sum(), 1), sel.mean(), st[tep].mean(), np.mean(sel * Lw[tep])])
            if b % 100 == 99:
                log(f'valid {b + 1}: ours q=.15 B=1 error {np.mean(np.array(rec["valid|ours|cs|0.15|B1.0"])[:, 0]):.4f} reads {np.mean(np.array(rec["valid|ours|cs|0.15|B1.0"])[:, 2]):.3f}')
                save()
        save()
    save()
    for q in QS:
        for Bq in Bs[:3]:
            ks = ['ours'] + list(BASE) + ['oracle']
            print(ds, args.rtag, args.fresh_tag, f'q{q} B{Bq}', f"th {out.get(f'theta|ours|cs|{q}|B{Bq}', [np.nan])[0]:.2f}",
                  ' '.join(f"{k}:{100 * out[f'{k}|cs|{q}|B{Bq}'][:, 1].mean():.1f}" for k in ks if f'{k}|cs|{q}|B{Bq}' in out), flush=True)
    log('done')


def peek_inflation(p, L, cal_idx, tests, alphas, last, grid=None):
    """RQ1: one fixed-look conformal threshold per round (each valid at its own round), used inside the loop.
    Returns {alpha: [R, 4]} = (stopped wrong-answer risk over all questions, largest single-round risk, coverage,
    error among answered questions)."""
    grid = C.strike_grid() if grid is None else grid
    out = {a: [] for a in alphas}
    for r, c in enumerate(cal_idx):
        te = tests[r]
        for a in alphas:
            th = np.array([C.crc_strike(p[:, t], L[:, t], c, grid, a) for t in range(last + 1)])
            hit = p[te][:, : last + 1] >= th[None]
            st = C.first_true(hit, last)
            ans = hit.any(1)
            risk = float(np.mean(ans * L[te, st]))
            single = max(float(np.mean(hit[:, t] * L[te, t])) for t in range(last + 1))
            cov = float(ans.mean())
            out[a].append([risk, single, cov, risk / max(cov, 1e-9)])
    return {a: np.array(v) for a, v in out.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--traj'); ap.add_argument('--out'); ap.add_argument('--ds')
    ap.add_argument('--layer', type=int, default=18)
    ap.add_argument('--rtag', default='')
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--R', type=int, default=200)
    ap.add_argument('--nboot', type=int, default=200)
    ap.add_argument('--Rw', type=int, default=100)
    ap.add_argument('--nvalid', type=int, default=500)
    ap.add_argument('--K', type=int, default=5)
    ap.add_argument('--M', type=int, default=5)
    ap.add_argument('--pca', type=int, default=128)
    ap.add_argument('--cells', type=int, default=20)
    ap.add_argument('--parts', default='main,boot,peek,oracle,calsize,p1,lab,abl,inv,sdiag,tok,dl,contracts')
    ap.add_argument('--vep', type=int, default=0); ap.add_argument('--fep', type=int, default=0)
    ap.add_argument('--fresh_tag', default='')
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    run(args)


if __name__ == '__main__':
    main()

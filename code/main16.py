"""Round 16 main experiment (results/r16), one reader-benchmark unit per call.

SCOPE (lattice16): the controller FORECASTS the reader's belief at every future round from the reader's state
history and prices the next read by the best forecasted gain in certified slack per unit of resource (the fair
price with deterministic horizons); stop at the first round where the price falls to the budget price.  The
same forecaster serves every reliability target, budget, horizon and cost unit.  Every policy answers with the
SAME cross-fitted compact verifier (reader state + 4 generation/grounding signals) and is certified by the SAME
wrapper (cert14).
  ours     SCOPE forecaster (state + black-box forecast average), gate g = theta = 1 - q (batch) / CRC anchor (risk)
  mix      randomised fixed horizon
  tokconf  token-probability trigger (FLARE/DRAGIN style)
  vtrig    verifier trigger (SeaKR / Self-RAG-critic style)
  router   Adaptive-RAG style complexity router
  stoprag  Stop-RAG Q(lambda) stop/continue values
  halt     HALT-style evidence-coverage stopper (LLM judge scores)
  value    least-squares Monte Carlo value regression (same inputs, same gate, same certificates)
  index    model-free fair-price index trained by counterfactual policy optimisation (round-15 SCOPE), same gate
Parts: main, boot, peek, abl, tok, dl, lab (see main15 for the protocol; the ablations here are
  myopic (one-read horizon), pop (population forecasts from the current cell), stateonly / blindonly forecaster,
  ctrlbb / verbb / blackbox (2 x 2 reader-state information), value, index)."""
import os, sys, json, argparse, time
import numpy as np, torch

sys.path.insert(0, os.path.dirname(__file__))
import fpi, cert14 as C, lattice16 as Lt
from main15 import load_unit, train_router, auroc, norm_ans, NONANS, budgets

QS = (0.1, 0.15, 0.2)
ALPHAS = (0.1, 0.15, 0.2)
TOKF = (0.15, 0.25, 0.35, 0.5)
BASE = ('mix', 'tokconf', 'vtrig', 'router', 'stoprag', 'halt', 'value', 'index')
SIG4 = ['lp_mean', 'p_suff', 'same_prev', 'any_support']


def views16(z, H, mode):
    """compact observations: 4 generation/grounding signals + answer similarity + time (+ state drift when the
    reader state is used).  Returns X [N, T1, F] and whether the state is used."""
    names = list(z['u_names']); u = np.asarray(z['u'], np.float32)
    N, T1 = u.shape[:2]
    parts = [u[:, :, [names.index(c) for c in SIG4]], np.asarray(z['d'], np.float32)[:, :, 1:2],
             np.repeat((np.arange(T1, dtype=np.float32) / max(T1 - 1, 1))[None, :, None], N, 0)]
    use_h = H is not None and mode != 'blackbox'
    if use_h:
        hn = H / (np.linalg.norm(H, axis=-1, keepdims=True) + 1e-6)
        parts.append(np.concatenate([np.zeros((N, 1), np.float32), (hn[:, 1:] * hn[:, :-1]).sum(-1)], 1)[..., None])
    return np.concatenate(parts, -1).astype(np.float32), use_h


class Unit:
    """verifier (cross-fitted), lattice, forecasters and curves for one observation mode.
    base: reuse observations AND verifier of another unit (only the controller differs);
    obs : reuse observations only (new verifier, e.g. a new label);
    ctrl: the unit whose observations the FORECASTER reads (default: its own)."""

    def __init__(self, z, H, y, tr, ev, view, args, log, base=None, obs=None, ctrl=None, kind='fc', insample=False):
        self.view, self.tr, self.ev, self.kind = view, tr, ev, kind
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
            self.p_tr = self.V(hp(tr), self.X[tr]) if insample else self.V.oof
            self.p_ev = self.V(hp(ev), self.X[ev])
            log(f'verifier[{view}] ready')
        self.lat = Lt.Lattice(self.p_tr, y[tr], K=args.cells)
        self.b_tr, self.b_ev = self.lat.binof(self.p_tr), self.lat.binof(self.p_ev)
        c = self if ctrl is None else ctrl
        chp = lambda idx: None if c.Hp is None else c.Hp[idx]
        self.F_tr = None
        if kind == 'fc':
            kw = dict(epochs=args.fep) if args.fep else {}
            fcs = Lt.CrossForecast(chp(tr), c.X[tr], self.p_tr, y[tr], self.lat, self.V.fold, seed=args.seed, **kw)
            self.F_ev = fcs(chp(ev), c.X[ev], self.p_ev)
            self.F_tr = fcs.oof(chp(tr), c.X[tr], self.p_tr)
            if args.ens and c.Hp is not None:
                Xb, _ = views16(z, None, 'blackbox'); Xb = fpi.Std(Xb[tr])(Xb)
                fcb = Lt.CrossForecast(None, Xb[tr], self.p_tr, y[tr], self.lat, self.V.fold, seed=args.seed, **kw)
                self.F_ev = 0.5 * (self.F_ev + fcb(None, Xb[ev], self.p_ev))
                self.F_tr = 0.5 * (self.F_tr + fcb.oof(None, Xb[tr], self.p_tr))
            log(f'forecaster[{view}{"" if ctrl is None else " ctrl=" + ctrl.view}{" insample" if insample else ""}] trained')
        elif kind == 'pop':
            self.F_ev = pop_forecast(self.lat, self.b_ev); self.F_tr = pop_forecast(self.lat, self.b_tr)

    def curve(self, g, th, cost_ev, last=None, myopic=False, payoff=None):
        rho = Lt.fair_price(self.lat, self.F_ev, self.b_ev, self.p_ev, g, th, cost_ev, last=last, myopic=myopic,
                            payoff=payoff or PAYOFF)
        return Lt.index_curve(rho, GRID, self.lat.T if last is None else int(last))


PAYOFF = 'table'      # cell payoff through the verifier's reliability table (set from --payoff)


def pop_forecast(lat, b):
    """population forecasts: F[:, t, tau-1] = row b_t of K_t K_{t+1} ... K_{tau-1}."""
    N, T1 = b.shape; T = T1 - 1; K = lat.K
    F_ = np.zeros((N, T1, T, K), np.float32)
    for t in range(T):
        M = np.eye(K)
        for tau in range(t + 1, T1):
            M = M @ lat.Kt[tau - 1]
            F_[:, t, tau - 1] = M[b[:, t]]
    return F_


GRID = Lt.price_grid(200)


def gate_for(target, p_tr, L_tr, B, T, seed, R=20):
    """gate g (= strike theta): 1 - q for the batch target; for the risk target the median CRC threshold of the
    fixed horizon at this budget on training halvings (the fixed-horizon anchor of the dual variable)."""
    if target[0] == 'cs':
        return 1.0 - target[1]
    rs = np.random.RandomState(31 + seed); N = len(p_tr); ar = np.arange(N); gs = []
    sgrid = C.strike_grid()
    st = C.mix_stops(N, B, T, seed=7)
    for _ in range(R):
        A = rs.permutation(N)[: N // 4]
        gs.append(float(C.crc_strike(p_tr[ar, st], L_tr[ar, st], A, sgrid, target[1])))
    g = float(np.median(gs))
    return min(max(g, 0.05), 0.99)


def run(args):
    t0 = time.time()
    ds, sd = args.ds, args.seed
    log = lambda m: print(ds, args.rtag, m, f'{time.time() - t0:.0f}s', flush=True)
    parts = set(args.parts.split(','))
    z, H, rows, extra = load_unit(args.traj, ds, args.layer, args.fresh_tag)
    y = z['y'].astype(np.float32); split = z['split']
    tr, ev = np.where(split == 0)[0], np.where(split == 1)[0]
    if args.max_train:
        tr = tr[: args.max_train]
    T1 = y.shape[1]; T = T1 - 1
    tok_all = np.cumsum(np.array([[s['n_prompt_tok'] + s['n_gen_tok'] for s in r['steps']] for r in rows], float), 1)
    tokmean = float(tok_all[tr, -1].mean())
    cost_ret = np.repeat(np.arange(T1, dtype=np.float32)[None], len(y), 0)
    tok = tok_all[ev]; L = 1 - y[ev]; N = len(ev); ar = np.arange(N)
    Bs = budgets(T)
    rng = np.random.RandomState(123 + sd)
    perms = [rng.permutation(N) for _ in range(args.R)]
    cal_idx = [pm[: N // 2] for pm in perms]; tests = [pm[N // 2:] for pm in perms]
    Ud = np.random.RandomState(999).rand(N)
    out = {'meta|T': np.array([T]), 'meta|N': np.array([N, len(tr)]), 'meta|tok': np.array([tokmean, float(tok[:, -1].mean())])}
    tgt_main = [('cs', q) for q in QS] + [('risk', 0.15)]
    Lfull = 1 - y

    def save():
        np.savez_compressed(os.path.join(args.out, f'{ds}{args.rtag}{args.fresh_tag}_s{sd}_r16.npz'), **out)

    def certify_ours(unit, tag, targets, Bs_, cost_ev, cmax, last=None, myopic=False, Lev=None, Ltr=None, keep=True, dc_tok=None):
        """our curves per (target, budget): gate from gate_for on training beliefs; certified on evaluation."""
        Lev = L if Lev is None else Lev
        Ltr = Lfull[tr] if Ltr is None else Ltr
        for B in Bs_:
            if isinstance(B, tuple):
                Btr, Bev, bkey = B[0] * T, B[1], f'f{B[2]}'      # token fraction -> equivalent retrieval budget for the gate
            else:
                Btr, Bev, bkey = B, B, f'B{B}'
            for tg in targets:
                g = gate_for(tg, unit.p_tr, Ltr, Btr, T, sd)
                S = unit.curve(g, g, cost_ev if dc_tok is None else dc_tok, last=last, myopic=myopic)
                if keep:
                    out[f'theta|{tag}|{tg[0]}|{tg[1]}|{bkey}'] = np.array([g])
                cst = fpi.path_cost(S, cost_ev)
                r = C.certify_curve(S, cst, cmax, unit.p_ev, Lev, tok, cal_idx, tests, Ud, Bev, [tg])
                out[f'{tag}|{tg[0]}|{tg[1]}|{bkey}'] = r[tg]
                if keep and tg == ('cs', 0.15) and not isinstance(B, tuple):
                    out[f'curve|{tag}|B{B}'] = S.astype(np.int8)

    def certify_base(name, S, p_ev, targets, Bs_, cost_ev, cmax, prefix='', Lev=None):
        Lev = L if Lev is None else Lev
        for B in Bs_:
            r = C.certify_curve(S, fpi.path_cost(S, cost_ev), cmax, p_ev, Lev, tok, cal_idx, tests, Ud, B, targets)
            for tg, v in r.items():
                out[f'{prefix}{name}|{tg[0]}|{tg[1]}|B{B}'] = v

    def certify_mix(p_ev, targets, Bs_, prefix='', Lev=None):
        Lev = L if Lev is None else Lev
        for B in Bs_:
            st = C.mix_stops(N, B, T)
            for tg, v in C.certify_fixed(st, p_ev, Lev, tok, cal_idx, tests, targets).items():
                out[f'{prefix}mix|{tg[0]}|{tg[1]}|B{B}'] = v

    def learned_curves(unit, kind, cost_tr, cost_ev, targets, Bs_, tag, cmax, Lev=None, Ltr=None, tB=None):
        """matched learned stoppers on the same inputs: value (LSMC) or index (model-free fair price), deployed
        at the same gate/strike as ours.  Returns {theta: S_ev} for the bootstrap."""
        Lev = L if Lev is None else Lev; Ltr = Lfull[tr] if Ltr is None else Ltr
        hp = lambda idx: None if unit.Hp is None else unit.Hp[idx]
        pol = fpi.CrossPolicy(hp(tr), unit.X[tr], unit.p_tr, y[tr], cost_tr, unit.V.fold, kind=kind, seed=sd, last=None)
        ths = sorted({round(gate_for(tg, unit.p_tr, Ltr, (B[0] * T if isinstance(B, tuple) else B), T, sd), 2) for tg in targets for B in Bs_})
        curves = {}
        if kind == 'index':
            R_tr = pol.oof(hp(tr), unit.X[tr], unit.p_tr, cost_tr, ths); R_ev = pol(hp(ev), unit.X[ev], unit.p_ev, cost_ev, ths)
            for k, th in enumerate(ths):
                curves[th] = fpi.index_path(R_ev[:, k], fpi.price_grid(R_tr[:, k], T), T)
        else:
            kaps = pol.ps * np.concatenate([-np.geomspace(0.5, 1e-3, 40), [0.0], np.geomspace(1e-3, 1.0, 120)])
            for th in ths:
                C_ev = pol(hp(ev), unit.X[ev], unit.p_ev, cost_ev, [th], kaps)
                curves[th] = fpi.value_path(C_ev[:, 0], unit.p_ev, th, kaps, T)
        for B in Bs_:
            if isinstance(B, tuple):
                Btr, Bev, bkey = B[0] * T, B[1], f'f{B[2]}'
            else:
                Btr, Bev, bkey = B, B, f'B{B}'
            for tg in targets:
                th = round(gate_for(tg, unit.p_tr, Ltr, Btr, T, sd), 2)
                S = curves[th]
                out[f'theta|{tag}|{tg[0]}|{tg[1]}|{bkey}'] = np.array([th])
                r = C.certify_curve(S, fpi.path_cost(S, cost_ev), cmax, unit.p_ev, Lev, tok, cal_idx, tests, Ud, Bev, [tg])
                out[f'{tag}|{tg[0]}|{tg[1]}|{bkey}'] = r[tg]
        log(f'{tag} certified')
        return curves

    def baseline_curves(u_):
        names = list(z['u_names'])
        sc = np.exp(z['u'][:, :, names.index('lp_mean')])
        bc = {'tokconf': C.first_cross_path(sc[ev], C.thr_desc(sc[tr][:, :T]), T),
              'vtrig': C.first_cross_path(u_.p_ev, C.thr_desc(u_.p_tr[:, :T]), T)}
        bc.update(shared)
        return bc

    # ------------------------------------------------------------------ main unit ---------------------------------
    main = Unit(z, H, y, tr, ev, args.view, args, log)
    p_ev = main.p_ev
    auc = [auroc(y[ev][:, t], p_ev[:, t]) for t in range(T1)]
    out['meta|auc'] = np.array(auc); out['meta|acc'] = y[ev].mean(0); out['meta|oracle'] = np.array([y[ev].max(1).mean()])
    log('AUROC per round ' + ' '.join(f'{a:.3f}' for a in auc))
    # forecast quality (eval questions): mean NLL of the realised future cell vs the population forecast
    def fc_nll(Fc, b):
        v = [-np.log(np.take_along_axis(Fc[:, t, tau - 1], b[:, tau:tau + 1], -1)[:, 0] + 1e-9).mean() for t in range(T) for tau in range(t + 1, T1)]
        return float(np.mean(v))
    out['meta|fc_nll'] = np.array([fc_nll(main.F_ev, main.b_ev), fc_nll(pop_forecast(main.lat, main.b_ev), main.b_ev)])
    out['meta|cells'] = np.array([args.cells]); out['meta|edges'] = main.lat.edges; out['meta|acc_table'] = main.lat.acc
    shared, base_curves, value_curves, index_curves = {}, {}, None, None
    if parts & {'main', 'boot', 'lab', 'tok'}:
        need = np.where(y.max(1) > 0, y.argmax(1), T)
        x0 = np.concatenate([main.Hp[:, 0], main.X[:, 0]], -1) if main.Hp is not None else main.X[:, 0]
        routers = [train_router(x0[tr], need[tr], T1, seed=1000 * sd + m) for m in range(args.M)]
        cdf = np.cumsum(np.mean([net(x0[ev]) for net in routers], 0), 1)
        cdf_tr = np.cumsum(np.mean([net(x0[tr]) for net in routers], 0), 1)
        shared['router'] = C.first_cross_path(cdf, C.thr_desc(cdf_tr[:, :T]), T)
        from baselines import train_stoprag, stoprag_index
        hp = lambda idx: None if main.Hp is None else main.Hp[idx]
        srnets = [train_stoprag(hp(tr), main.X[tr], y[tr], seed=1000 * sd + m, dev=fpi.DEV) for m in range(args.M)]
        Isr = np.mean([stoprag_index(net, hp(ev), main.X[ev], dev=fpi.DEV)[0] for net in srnets], 0)
        Isr_tr = np.mean([stoprag_index(net, hp(tr), main.X[tr], dev=fpi.DEV)[0] for net in srnets], 0)
        shared['stoprag'] = C.first_cross_path(Isr, C.thr_desc(Isr_tr[:, :T]), T)
        if extra.get('halt') is not None:
            hs = np.nan_to_num(extra['halt'][ev].astype(float), nan=0.0)
            shared['halt'] = C.first_cross_path(hs, np.r_[1.01, np.linspace(1.0, 0.0, 160)], T)
        base_curves = baseline_curves(main)
        log('baselines trained')
    if parts & {'main', 'boot', 'lab'}:
        certify_ours(main, 'ours', tgt_main, Bs, cost_ret[ev], T)
        out['mono|ours'] = np.array([Lt.nested_check(out[f'curve|ours|B{B}']) for B in Bs])
        log('ours certified')
        for k, S in base_curves.items():
            certify_base(k, S, p_ev, tgt_main, Bs, cost_ret[ev], T)
        certify_mix(p_ev, tgt_main, Bs)
        value_curves = learned_curves(main, 'value', cost_ret[tr], cost_ret[ev], tgt_main, Bs, 'value', T)
        index_curves = learned_curves(main, 'index', cost_ret[tr], cost_ret[ev], tgt_main, Bs, 'index', T)
        log('main certified')
        save()
    # ------------------------------------------------------------------ RQ1: repeated looks ------------------------
    if 'peek' in parts:
        for a, v in C.peek_inflation(p_ev, L, cal_idx, tests, ALPHAS, T).items():
            out[f'peek|{a}'] = v
        log('peek')
    # ------------------------------------------------------------------ paired bootstrap ---------------------------
    if 'boot' in parts and value_curves is not None:
        brng = np.random.RandomState(9000 + sd)
        bt, boot_cost = {}, {}
        ours_q = {q: main.curve(1 - q, 1 - q, cost_ret[ev]) for q in QS}
        for b in range(args.nboot):
            idx = brng.randint(0, N, N)
            c, te = idx[: N // 2], idx[N // 2:]
            A, Bh = c[: len(c) // 2], c[len(c) // 2:]
            for B in Bs:
                for q in QS:
                    pols = {'ours': ours_q[q]}
                    pols.update({k: S for k, S in base_curves.items()})
                    pols['value'] = value_curves[round(1 - q, 2)]; pols['index'] = index_curves[round(1 - q, 2)]
                    for k, S in pols.items():
                        key = id(S)
                        if key not in boot_cost:
                            boot_cost[key] = C.env_cols(fpi.path_cost(S, cost_ret[ev]))
                        st, ab, _ = C.deploy_once(S, boot_cost[key], T, Ud, Bh, B, cost_is_envelope=True)
                        s = np.where(ab, -np.inf, p_ev[ar, st])
                        sel = C.conf_select(s, L[ar, st], A, te, q)
                        bt.setdefault(f'{k}|cs|{q}|B{B}', []).append([sel.mean(), (sel * (1 - L[ar, st][te])).mean()])
                    st = C.mix_stops(N, B, T)
                    sel = C.conf_select(p_ev[ar, st], L[ar, st], A, te, q)
                    bt.setdefault(f'mix|cs|{q}|B{B}', []).append([sel.mean(), (sel * (1 - L[ar, st][te])).mean()])
        for k, v in bt.items():
            out[f'boot|{k}'] = np.array(v)
        log('bootstrap')
        save()
    # ------------------------------------------------------------------ labels -------------------------------------
    if 'lab' in parts:
        import re
        ans = np.array([[norm_ans(s['answer']) for s in r['steps']] for r in rows], dtype=object)
        nonans = np.vectorize(lambda a: (not a) or bool(NONANS.search(a)))(ans)
        labs = {'em': z['yem'].astype(np.float32)}
        if extra.get('sem') is not None:
            sem = extra['sem'].astype(float)
            if sem.shape == y.shape:
                sem = np.where(np.isfinite(sem), sem, y)
                labs['judge'] = ((sem > 0.5) & ~nonans).astype(np.float32)
                out['meta|judge_agree'] = np.array([float(np.mean(labs['judge'][ev] == (y[ev] > 0.5))), float(np.mean(nonans[ev] & (sem[ev] > 0.5)))])
        out['meta|nonans_rate'] = np.array([float(nonans[ev].mean()), float(np.mean(nonans[ev] & (y[ev] > 0.5)))])
        tg = [('cs', 0.1), ('cs', 0.15), ('cs', 0.2), ('risk', 0.15)]
        for ln, yl in labs.items():
            Ll_tr, Ll_ev = 1 - yl[tr], 1 - yl[ev]
            ul = Unit(z, H, yl, tr, ev, args.view, args, log, obs=main)
            out[f'meta|auc_{ln}'] = np.array([auroc(yl[ev][:, t], ul.p_ev[:, t]) for t in range(T1)])
            out[f'meta|acc_{ln}'] = yl[ev].mean(0)
            certify_ours(ul, f'lab|{ln}|ours', tg, Bs[:3], cost_ret[ev], T, Lev=Ll_ev, Ltr=Ll_tr, keep=False)
            for k, S in baseline_curves(ul).items():
                certify_base(k, S, ul.p_ev, tg, Bs[:3], cost_ret[ev], T, prefix=f'lab|{ln}|', Lev=Ll_ev)
            certify_mix(ul.p_ev, tg, Bs[:3], prefix=f'lab|{ln}|', Lev=Ll_ev)
            learned_curves(ul, 'value', cost_ret[tr], cost_ret[ev], tg, Bs[:3], f'lab|{ln}|value', T, Lev=Ll_ev, Ltr=Ll_tr)
            log(f'labels {ln}')
            save()
    # ------------------------------------------------------------------ ablations ----------------------------------
    if 'abl' in parts:
        tg = [('cs', q) for q in QS] + [('risk', 0.15)]
        for B in Bs:
            for t_ in tg:
                for src, dst in (('ours', 'full'), ('value', 'value'), ('index', 'index')):
                    if f'{src}|{t_[0]}|{t_[1]}|B{B}' in out:
                        out[f'abl|{dst}|{t_[0]}|{t_[1]}|B{B}'] = out[f'{src}|{t_[0]}|{t_[1]}|B{B}']
        certify_ours(main, 'abl|myopic', tg, Bs, cost_ret[ev], T, myopic=True, keep=False)
        # payoff through the raw belief (calibration assumed) instead of the reliability table
        for B in Bs:
            for t_ in tg:
                g = gate_for(t_, main.p_tr, Lfull[tr], B, T, sd)
                S = main.curve(g, g, cost_ret[ev], payoff='belief' if PAYOFF == 'table' else 'table')
                r = C.certify_curve(S, fpi.path_cost(S, cost_ret[ev]), T, p_ev, L, tok, cal_idx, tests, Ud, B, [t_])
                out[f'abl|payoff|{t_[0]}|{t_[1]}|B{B}'] = r[t_]
        pop = Unit(z, H, y, tr, ev, args.view, args, log, base=main, kind='pop')
        certify_ours(pop, 'abl|pop', tg, Bs, cost_ret[ev], T, keep=False)
        if args.ens:
            a2 = argparse.Namespace(**vars(args)); a2.ens = 0
            so = Unit(z, H, y, tr, ev, args.view, a2, log, base=main)
            certify_ours(so, 'abl|stateonly', tg, Bs, cost_ret[ev], T, keep=False)
        log('abl controller')
        save()
        vunits = {}
        for vw in args.abl_views.split(','):
            if not vw or (vw != 'blackbox' and H is None):
                continue
            u_ = Unit(z, H, y, tr, ev, vw, args, log)
            vunits[vw] = u_
            certify_ours(u_, f'abl|{vw}', tg, Bs, cost_ret[ev], T, keep=False)
            certify_mix(u_.p_ev, tg, Bs, prefix=f'abl|{vw}|')
            save()
        if 'blackbox' in vunits and main.Hp is not None:
            bb = vunits['blackbox']
            cb = Unit(z, H, y, tr, ev, args.view, args, log, base=main, ctrl=bb)          # forecaster black-box
            certify_ours(cb, 'abl|ctrlbb', tg, Bs, cost_ret[ev], T, keep=False)
            vb = Unit(z, H, y, tr, ev, 'blackbox', args, log, base=bb, ctrl=main)         # verifier black-box
            certify_ours(vb, 'abl|verbb', tg, Bs, cost_ret[ev], T, keep=False)
            save()
        log('ablations')
    # ------------------------------------------------------------------ token budgets ------------------------------
    if 'tok' in parts:
        ctok = (tok_all / tokmean).astype(np.float32)
        tmax = float(4096 * T1)
        if np.any(np.diff(tok_all, axis=1, prepend=0) > 4096):
            raise ValueError('Recorded round exceeds the declared 4096-token context bound')
        tbud = [f * tokmean for f in TOKF]
        out['tok|budgets'] = np.array(tbud)
        tg = [('cs', 0.1), ('cs', 0.15), ('cs', 0.2), ('risk', 0.15)]
        tB = [(f, Bt, f) for f, Bt in zip(TOKF, tbud)]
        # our planner prices reads in expected tokens: the training-mean cumulative token cost per round
        dc_tok = np.repeat(ctok[tr].mean(0)[None], N, 0)
        certify_ours(main, 'tok|ours', tg, tB, tok, tmax, keep=False, dc_tok=dc_tok)
        learned_curves(main, 'value', ctok[tr], tok, tg, tB, 'tok|value', tmax)
        fixed = np.concatenate([np.repeat(np.arange(T, -1, -1)[None], N, 0), np.zeros((N, 1), int)], 1)
        bcs = dict(fixed=fixed, **{k: v for k, v in base_curves.items()})
        for k, S in bcs.items():
            for f, Bt in zip(TOKF, tbud):
                for t_, v in C.certify_curve(S, fpi.path_cost(S, tok), tmax, p_ev, L, tok, cal_idx, tests, Ud, Bt, tg).items():
                    out[f'tok|{k}|{t_[0]}|{t_[1]}|f{f}'] = v
        log('token budgets')
        save()
    # ------------------------------------------------------------------ deadline -----------------------------------
    if 'dl' in parts:
        tg = [('cs', 0.1), ('cs', 0.15), ('cs', 0.2), ('risk', 0.15)]
        for B in (1.0, 2.0):
            d = int(np.ceil(B)) + 1
            if d >= T:
                continue
            certify_ours(main, f'dl{d}', tg, (B,), cost_ret[ev], d, last=d, keep=False)
            for t_ in tg:
                g = gate_for(t_, main.p_tr, Lfull[tr], B, T, sd)
                S = np.minimum(main.curve(g, g, cost_ret[ev]), d)
                r = C.certify_curve(S, fpi.path_cost(S, cost_ret[ev]), d, p_ev, L, tok, cal_idx, tests, Ud, B, [t_])
                out[f'cap{d}|{t_[0]}|{t_[1]}|B{B}'] = r[t_]
        log('deadline')
        save()
    save()
    for q in QS:
        for B in Bs[:3]:
            ks = ['ours'] + list(BASE)
            print(ds, args.rtag, f'q{q} B{B}', ' '.join(f"{k}:{100 * out[f'{k}|cs|{q}|B{B}'][:, 1].mean():.1f}" for k in ks if f'{k}|cs|{q}|B{B}' in out), flush=True)
    log('done')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--traj'); ap.add_argument('--out'); ap.add_argument('--ds')
    ap.add_argument('--layer', type=int, default=18)
    ap.add_argument('--rtag', default='')
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--R', type=int, default=200)
    ap.add_argument('--nboot', type=int, default=200)
    ap.add_argument('--K', type=int, default=5)
    ap.add_argument('--M', type=int, default=5)
    ap.add_argument('--pca', type=int, default=128)
    ap.add_argument('--cells', type=int, default=20)
    ap.add_argument('--ens', type=int, default=1, help='average the reader-state forecaster with a black-box one')
    ap.add_argument('--view', default='state', choices=('state', 'blackbox'))
    ap.add_argument('--abl_views', default='blackbox')
    ap.add_argument('--parts', default='main,boot,peek,lab,abl,tok,dl')
    ap.add_argument('--vep', type=int, default=0, help='verifier epochs (0 = default); smoke tests')
    ap.add_argument('--fep', type=int, default=0, help='forecaster epochs (0 = default); smoke tests')
    ap.add_argument('--max_train', type=int, default=0)
    ap.add_argument('--fresh_tag', default='')
    ap.add_argument('--payoff', default='table', choices=('table', 'belief'))
    args = ap.parse_args()
    global PAYOFF
    PAYOFF = args.payoff
    os.makedirs(args.out, exist_ok=True)
    run(args)


if __name__ == '__main__':
    main()

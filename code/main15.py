"""Round 13 main experiment (results/r15), one reader-benchmark unit per call (single training seed).

Every policy answers with the SAME cross-fitted multi-view verifier and is certified by the SAME wrapper (cert14):
  ours     SCOPE: fair-price index, counterfactual policy optimisation, GATE-CONSISTENT strike (fixed point of the
           strike -> certificate-gate map on out-of-fold training questions; fpi.gate_theta)
  mix      randomised fixed horizon (floor(B) or ceil(B) rounds, mean exactly B)
  tokconf  FLARE/DRAGIN-style trigger on the answer-token probability
  vtrig    verifier trigger (SeaKR / Self-RAG-critic style): stop once the verifier is confident
  router   Adaptive-RAG-style complexity router (number of retrievals predicted before retrieving)
  stoprag  Stop-RAG: Q(lambda) stop/continue values
  halt     HALT-style: fraction of reasoning steps supported by the evidence (LLM judge scores)
  value    matched scalar value network trained by least-squares Monte Carlo (same inputs, same gate-consistent
           strike, same certificates: the 'scalar value + same certification' baseline)
Parts (--parts, comma separated; default all):
  main   targets: batch error among answers q (conformal selection) and risk alpha, retrieval budgets
  boot   paired question-level bootstrap of ours against every baseline (batch targets)
  peek   RQ1: fixed-look thresholds deployed inside the loop
  abl    ablations.  Strike: grid (max certified coverage over the strikes, the round-12 rule), th1q (1 - q).
         Controller: myopic (one-step index, no policy optimisation), value (LSMC), insample (beliefs not
         cross-fitted).  Information, 2 x 2: ctrlbb (verifier with reader state, controller black-box), verbb
         (verifier black-box, controller with reader state), blackbox (both), plus noground; Mix is re-run with each
         verifier, so the verifier's and the controller's share of the reader-state gain are separated.
  tok    reader-token budgets: index AND value network retrained with token cost; fixed horizon as a dithered curve
  dl     deadline-aware index (forced stop at d = ceil(B) + 1, answer at the deadline) for bounded tail latency
  lab    label definitions (exact match; LLM judge with non-answer filter): verifier, index and strike RETRAINED on
         the label that is certified (the label is part of the service specification), every baseline answering
         with the label's verifier
Output {out}/{ds}{rtag}{fresh_tag}_s{seed}_r15.npz"""
import os, sys, json, argparse, time, re
import numpy as np, torch

sys.path.insert(0, os.path.dirname(__file__))
import fpi
import cert14 as C

QS = (0.1, 0.15, 0.2)
ALPHAS = (0.1, 0.15, 0.2)
QABL = (0.1, 0.15, 0.2)
THS = fpi.STRIKES
TOKF = (0.15, 0.25, 0.35, 0.5)
BASE = ('mix', 'tokconf', 'vtrig', 'router', 'stoprag', 'halt', 'value')
NONANS = re.compile(r"\b(unknown|not (?:mentioned|provided|specified|stated|given|available|known|clear|enough)|"
                    r"cannot be determined|can ?not determine|no (?:information|answer|data)|insufficient|"
                    r"unanswerable|none|n/?a|i do not know|i don't know|unclear)\b")


def budgets(T):
    return (1.0, 1.5, 2.0, 2.5, 3.0) if T <= 4 else (1.0, 1.5, 2.0, 3.0, 4.0)


def norm_ans(a):
    return re.sub(r'\s+', ' ', re.sub(r'[^a-z0-9 ]', ' ', (a or '').lower())).strip()


def train_router(x, need, ncls, seed, dev=fpi.DEV, epochs=200):
    torch.manual_seed(seed)
    net = torch.nn.Sequential(torch.nn.Linear(x.shape[1], 128), torch.nn.ReLU(), torch.nn.Dropout(0.2),
                              torch.nn.Linear(128, ncls)).to(dev)
    opt = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=1e-2)
    X = torch.as_tensor(x, dtype=torch.float32, device=dev); Y = torch.as_tensor(need, device=dev)
    for _ in range(epochs):
        loss = torch.nn.functional.cross_entropy(net(X), Y)
        opt.zero_grad(); loss.backward(); opt.step()
    net.eval()
    return lambda z: torch.softmax(net(torch.as_tensor(z, dtype=torch.float32, device=dev)), -1).detach().cpu().numpy()


def auroc(y, s):
    y = np.asarray(y) > 0.5
    if y.all() or not y.any():
        return np.nan
    r = np.argsort(np.argsort(s, kind='mergesort'), kind='mergesort') + 1.0
    n1 = y.sum(); n0 = len(y) - n1
    return float((r[y].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def load_unit(traj, ds, layer, fresh_tag=''):
    """Features, hidden states, trajectories and judge/HALT side files.  split: 0 train, 1 evaluation.  With
    fresh_tag the evaluation questions are REPLACED by the fresh questions of {ds}{fresh_tag}_* (independent test):
    the original evaluation questions get split 2 and are not used at all."""
    def one(tag):
        z = dict(np.load(os.path.join(traj, f'{ds}{tag}_feat.npz'), allow_pickle=True))
        hp = os.path.join(traj, f'{ds}{tag}_h{layer}.npy')
        H = np.load(hp).astype(np.float32) if os.path.exists(hp) else None
        rows = [json.loads(l) for l in open(os.path.join(traj, f'{ds}{tag}.jsonl'))]
        ex = {}
        for k in ('halt', 'sem'):
            f = os.path.join(traj, f'{ds}{tag}_{k}.npy')
            ex[k] = np.load(f) if os.path.exists(f) else None
        return z, H, rows, ex
    z, H, rows, ex = one('')
    if not fresh_tag:
        return z, H, rows, ex
    zf, Hf, rowsf, exf = one(fresh_tag)
    keep = {k: np.concatenate([z[k], zf[k]]) for k in ('u', 'd', 'y', 'yem')}
    keep['u_names'] = z['u_names']
    keep['split'] = np.concatenate([np.where(z['split'] == 0, 0, 2), np.ones(len(zf['y']), np.int8)]).astype(np.int8)
    H2 = None if (H is None or Hf is None) else np.concatenate([H, Hf])
    ex2 = {k: (np.concatenate([ex[k], exf[k]]) if ex[k] is not None and exf[k] is not None else None) for k in ex}
    return keep, H2, rows + rowsf, ex2


class Unit:
    """Verifier, cross-fitted beliefs, policy networks and curves.
    obs : a Unit whose observations (X, reader-state PCA) are reused; a NEW verifier is trained on y (label studies)
    base: a Unit whose observations AND verifier are reused (only the policy is new)
    ctrl: a Unit whose observations the CONTROLLER reads (default: its own); the verifier is unaffected."""

    def __init__(self, z, H, y, tr, ev, cost_tr, view, args, log, insample=False, kind='index', last=None, base=None,
                 epochs=None, obs=None, ctrl=None):
        self.view, self.trI = view, tr
        src_ = base if base is not None else obs
        if src_ is not None:
            self.X, self.Hp, self.htr, self.hev = src_.X, src_.Hp, src_.htr, src_.hev
        else:
            X, use_h = fpi.views(z, H, view)
            X = fpi.Std(X[tr])(X)
            if use_h:
                dv = os.environ.get('PCA_DEVICE', 'cpu')
                Hp = fpi.pca_apply(H, fpi.pca_fit(H[tr], args.pca, dv, seed=args.seed), dv)
            else:
                Hp = None
            self.X, self.Hp = X, Hp
            self.htr, self.hev = (None, None) if Hp is None else (Hp[tr], Hp[ev])
        if base is not None:
            self.V = base.V
        else:
            kw = dict(epochs=args.vep) if args.vep else {}
            self.V = fpi.CrossFit(self.htr, self.X[tr], y[tr], K=args.K, seed=args.seed, **kw)
            log(f'verifier[{view}] ready')
        self.p_tr = self.V(self.htr, self.X[tr]) if insample else self.V.oof
        self.p_ev = base.p_ev if base is not None else self.V(self.hev, self.X[ev])
        c = self if ctrl is None else ctrl
        self.cX, self.chtr, self.chev = c.X, c.htr, c.hev
        self.cost_tr = cost_tr
        ep = epochs if epochs is not None else args.pep
        pk = dict(epochs=ep) if ep is not None else {}
        self.pol = fpi.CrossPolicy(self.chtr, self.cX[tr], self.p_tr, y[tr], cost_tr, self.V.fold, kind=kind,
                                   seed=args.seed, last=last, **pk)
        log(f'{kind}[{view}{"" if ctrl is None else " ctrl=" + ctrl.view}{" insample" if insample else ""}'
            f'{"" if last is None else " last=" + str(last)}{" myopic" if ep == 0 else ""}] trained')

    def index_curves(self, ev, cost_ev, ths=THS):
        """theta -> (S_tr out-of-fold, S_ev) along price grids fixed from the out-of-fold training indices."""
        last = self.pol.last
        R_tr = self.pol.oof(self.chtr, self.cX[self.trI], self.p_tr, self.cost_tr, list(ths))
        R_ev = self.pol(self.chev, self.cX[ev], self.p_ev, cost_ev, list(ths))
        out = {}
        for k, th in enumerate(ths):
            grid = fpi.price_grid(R_tr[:, k], last)
            out[th] = (fpi.index_path(R_tr[:, k], grid, last), fpi.index_path(R_ev[:, k], grid, last))
        return out

    def value_curves(self, ev, cost_ev, ths=THS):
        last, ps = self.pol.last, self.pol.ps
        kaps = ps * np.concatenate([-np.geomspace(0.5, 1e-3, 40), [0.0], np.geomspace(1e-3, 1.0, 120)])
        out = {}
        for th in ths:           # one strike at a time: bounded memory with the finer strike set
            C_tr = self.pol.oof(self.chtr, self.cX[self.trI], self.p_tr, self.cost_tr, [th], kaps)
            C_ev = self.pol(self.chev, self.cX[ev], self.p_ev, cost_ev, [th], kaps)
            out[th] = (fpi.value_path(C_tr[:, 0], self.p_tr, th, kaps, last), fpi.value_path(C_ev[:, 0], self.p_ev, th, kaps, last))
        return out


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
        np.savez_compressed(os.path.join(args.out, f'{ds}{args.rtag}{args.fresh_tag}_s{sd}_r15.npz'), **out)

    def certify_index(unit, curves, tag, targets, Bs_, cost_tr, cost_ev, cmax, rule='gate', fixed_theta=None,
                      theta_from=None, Ltr=None, Lev=None, cmax_tr=None, keep=True):
        """Strike per (target, budget) -- by default the gate-consistent fixed point on the out-of-fold training
        curves (rule='grid': max certified training coverage, the round-12 rule) -- then certified on the evaluation
        questions.  fixed_theta: target -> strike; theta_from: (target, B) -> strike.  A budget may be a tuple
        (training budget, evaluation budget, key) when training and evaluation costs are in different units."""
        Ltr = Lfull[tr] if Ltr is None else Ltr
        Lev = L if Lev is None else Lev
        cmax_tr = cmax if cmax_tr is None else cmax_tr
        res = {}
        for B in Bs_:
            if isinstance(B, tuple):
                Btr, Bev, bkey = B[0], B[1], f'f{B[2]}'
            else:
                Btr, Bev, bkey = B, B, f'B{B}'
            for tg in targets:
                if fixed_theta is not None:
                    th = fixed_theta(tg)
                elif theta_from is not None:
                    th = theta_from[(tg, B)]
                elif rule == 'grid':
                    th, _ = fpi.select_theta({t: c[0] for t, c in curves.items()}, unit.p_tr, Ltr, cost_tr, cmax_tr, Btr,
                                             tg, R=args.Rsel, seed=sd)
                else:
                    th, G = fpi.gate_theta({t: c[0] for t, c in curves.items()}, unit.p_tr, Ltr, cost_tr, cmax_tr, Btr,
                                           tg, R=args.Rsel, seed=sd)
                    if keep:
                        out[f'gate|{tag}|{tg[0]}|{tg[1]}|{bkey}'] = np.array([[t, G[t]] for t in sorted(G)], np.float32)
                if keep:
                    out[f'theta|{tag}|{tg[0]}|{tg[1]}|{bkey}'] = np.array([th])
                res[(tg, B)] = th
                S = curves[th][1]
                r = C.certify_curve(S, fpi.path_cost(S, cost_ev), cmax, unit.p_ev, Lev, tok, cal_idx, tests, Ud, Bev, [tg])
                out[f'{tag}|{tg[0]}|{tg[1]}|{bkey}'] = r[tg]
        return res

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

    def baseline_curves(u_):
        """Every baseline curve except the value network; the verifier trigger reads the verifier of u_."""
        names = list(z['u_names'])
        sc = np.exp(z['u'][:, :, names.index('lp_mean')])
        # Grids are frozen on training questions: evaluation quantiles would make the deployment family depend on
        # the questions it certifies.
        bc = {'tokconf': C.first_cross_path(sc[ev], C.thr_desc(sc[tr][:, :T]), T),
              'vtrig': C.first_cross_path(u_.p_ev, C.thr_desc(u_.p_tr[:, :T]), T)}
        bc.update(shared)
        return bc

    # ------------------------------------------------------------------ the main unit: multi-view, cross-fitted -----
    main = Unit(z, H, y, tr, ev, cost_ret[tr], args.view, args, log)
    p_ev = main.p_ev
    auc = [auroc(y[ev][:, t], p_ev[:, t]) for t in range(T1)]
    out['meta|auc'] = np.array(auc); out['meta|acc'] = y[ev].mean(0); out['meta|oracle'] = np.array([y[ev].max(1).mean()])
    log('AUROC per round ' + ' '.join(f'{a:.3f}' for a in auc))
    curves = main.index_curves(ev, cost_ret[ev])
    out['mono|ours'] = np.array([np.mean(np.all(np.diff(curves[t][1], axis=1) <= 0, 1)) for t in THS])
    theta_main = None
    base_curves, shared = {}, {}
    if parts & {'main', 'boot', 'lab', 'tok'}:
        # baselines whose stopping curves do not read the verifier
        need = np.where(y.max(1) > 0, y.argmax(1), T)
        x0 = np.concatenate([main.Hp[:, 0], main.X[:, 0]], -1) if main.Hp is not None else main.X[:, 0]
        routers = [train_router(x0[tr], need[tr], T1, seed=1000 * sd + m) for m in range(args.M)]
        cdf = np.cumsum(np.mean([net(x0[ev]) for net in routers], 0), 1)
        cdf_tr = np.cumsum(np.mean([net(x0[tr]) for net in routers], 0), 1)
        shared['router'] = C.first_cross_path(cdf, C.thr_desc(cdf_tr[:, :T]), T)
        from baselines import train_stoprag, stoprag_index
        srnets = [train_stoprag(main.htr, main.X[tr], y[tr], seed=1000 * sd + m, dev=fpi.DEV) for m in range(args.M)]
        Isr = np.mean([stoprag_index(net, main.hev, main.X[ev], dev=fpi.DEV)[0] for net in srnets], 0)
        Isr_tr = np.mean([stoprag_index(net, main.htr, main.X[tr], dev=fpi.DEV)[0] for net in srnets], 0)
        shared['stoprag'] = C.first_cross_path(Isr, C.thr_desc(Isr_tr[:, :T]), T)
        if extra.get('halt') is not None:
            hs = np.nan_to_num(extra['halt'][ev].astype(float), nan=0.0)
            shared['halt'] = C.first_cross_path(hs, np.r_[1.01, np.linspace(1.0, 0.0, 160)], T)
        base_curves = baseline_curves(main)
        log('baselines trained')
    if parts & {'main', 'boot', 'lab'}:
        theta_main = certify_index(main, curves, 'ours', tgt_main, Bs, cost_ret[tr], cost_ret[ev], T)
        log('ours certified')
        for k, S in base_curves.items():
            certify_base(k, S, p_ev, tgt_main, Bs, cost_ret[ev], T)
        certify_mix(p_ev, tgt_main, Bs)
        vunit = Unit(z, H, y, tr, ev, cost_ret[tr], args.view, args, log, kind='value', base=main)
        vcurves = vunit.value_curves(ev, cost_ret[ev])
        out['mono|value'] = np.array([np.mean(np.all(np.diff(vcurves[t][1], axis=1) <= 0, 1)) for t in THS])
        certify_index(vunit, vcurves, 'value', tgt_main, Bs, cost_ret[tr], cost_ret[ev], T)
        base_curves['value'] = vcurves
        log('main certified')
        save()
    # ------------------------------------------------------------------ RQ1: repeated looks ------------------------
    if 'peek' in parts:
        for a, v in C.peek_inflation(p_ev, L, cal_idx, tests, ALPHAS, T).items():
            out[f'peek|{a}'] = v
        log('peek')
    # ------------------------------------------------------------------ paired bootstrap ---------------------------
    if 'boot' in parts and theta_main is not None:
        brng = np.random.RandomState(9000 + sd)
        qb = (0.1, 0.15, 0.2)
        bt = {}
        boot_cost = {}          # the path and its cost envelope do not change across bootstrap draws
        for b in range(args.nboot):
            idx = brng.randint(0, N, N)
            c, te = idx[: N // 2], idx[N // 2:]
            A, Bh = c[: len(c) // 2], c[len(c) // 2:]
            for B in Bs:
                for q in qb:
                    pols = {'ours': curves[theta_main[(('cs', q), B)]][1]}
                    pols.update({k: S for k, S in base_curves.items() if k != 'value'})
                    pols['value'] = base_curves['value'][out[f'theta|value|cs|{q}|B{B}'][0]][1]
                    for k, S in pols.items():
                        if id(S) not in boot_cost:
                            boot_cost[id(S)] = C.env_cols(fpi.path_cost(S, cost_ret[ev]))
                        st, ab, _ = C.deploy_once(S, boot_cost[id(S)], T, Ud, Bh, B, cost_is_envelope=True)
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
        ans = np.array([[norm_ans(s['answer']) for s in r['steps']] for r in rows], dtype=object)
        nonans = np.vectorize(lambda a: (not a) or bool(NONANS.search(a)))(ans)
        labs = {'em': z['yem'].astype(np.float32)}
        if extra.get('sem') is not None:
            sem = extra['sem'].astype(float)
            if sem.shape == y.shape:
                sem = np.where(np.isfinite(sem), sem, y)
                labs['judge'] = ((sem > 0.5) & ~nonans).astype(np.float32)
                out['meta|judge_agree'] = np.array([float(np.mean(labs['judge'][ev] == (y[ev] > 0.5))),
                                                    float(np.mean(nonans[ev] & (sem[ev] > 0.5)))])
        out['meta|nonans_rate'] = np.array([float(nonans[ev].mean()), float(np.mean(nonans[ev] & (y[ev] > 0.5)))])
        tg = [('cs', 0.1), ('cs', 0.15), ('cs', 0.2), ('risk', 0.15)]
        for ln, yl in labs.items():
            Ll_tr, Ll_ev = 1 - yl[tr], 1 - yl[ev]
            ul = Unit(z, H, yl, tr, ev, cost_ret[tr], args.view, args, log, obs=main)      # retrained on the label
            out[f'meta|auc_{ln}'] = np.array([auroc(yl[ev][:, t], ul.p_ev[:, t]) for t in range(T1)])
            out[f'meta|acc_{ln}'] = yl[ev].mean(0)
            certify_index(ul, ul.index_curves(ev, cost_ret[ev]), f'lab|{ln}|ours', tg, Bs[:3], cost_ret[tr], cost_ret[ev],
                          T, Ltr=Ll_tr, Lev=Ll_ev)
            for k, S in baseline_curves(ul).items():
                certify_base(k, S, ul.p_ev, tg, Bs[:3], cost_ret[ev], T, prefix=f'lab|{ln}|', Lev=Ll_ev)
            certify_mix(ul.p_ev, tg, Bs[:3], prefix=f'lab|{ln}|', Lev=Ll_ev)
            log(f'labels {ln}')
            save()
    # ------------------------------------------------------------------ ablations ----------------------------------
    if 'abl' in parts:
        tg = [('cs', q) for q in QABL] + [('risk', 0.15)]
        if theta_main is None:
            certify_index(main, curves, 'abl|full', tg, Bs, cost_ret[tr], cost_ret[ev], T)
        else:
            for B in Bs:
                for t_ in tg:
                    out[f'abl|full|{t_[0]}|{t_[1]}|B{B}'] = out[f'ours|{t_[0]}|{t_[1]}|B{B}']
                    if f'value|{t_[0]}|{t_[1]}|B{B}' in out:
                        out[f'abl|value|{t_[0]}|{t_[1]}|B{B}'] = out[f'value|{t_[0]}|{t_[1]}|B{B}']
        # strike: max certified training coverage over the same strikes (the round-12 rule), and theta = 1 - q
        certify_index(main, curves, 'abl|grid', tg, Bs, cost_ret[tr], cost_ret[ev], T, rule='grid')
        certify_index(main, curves, 'abl|th1q', [t for t in tg if t[0] == 'cs'], Bs, cost_ret[tr], cost_ret[ev], T,
                      fixed_theta=lambda t: min(THS, key=lambda s: abs(s - (1 - t[1]))))
        log('abl strikes')
        # controller: one-step (myopic) index without policy optimisation; beliefs not cross-fitted
        myo = Unit(z, H, y, tr, ev, cost_ret[tr], args.view, args, log, base=main, epochs=0)
        certify_index(myo, myo.index_curves(ev, cost_ret[ev]), 'abl|myopic', tg, Bs, cost_ret[tr], cost_ret[ev], T)
        ins = Unit(z, H, y, tr, ev, cost_ret[tr], args.view, args, log, insample=True, base=main)
        certify_index(ins, ins.index_curves(ev, cost_ret[ev]), 'abl|insample', tg, Bs, cost_ret[tr], cost_ret[ev], T)
        if theta_main is None:     # value regression when the main part did not run
            vunit = Unit(z, H, y, tr, ev, cost_ret[tr], args.view, args, log, kind='value', base=main)
            certify_index(vunit, vunit.value_curves(ev, cost_ret[ev]), 'abl|value', tg, Bs, cost_ret[tr], cost_ret[ev], T)
        save()
        # information, 2 x 2: (verifier, controller) x (with reader state, black-box), and the grounding view
        vunits = {}
        for vw in args.abl_views.split(','):
            if not vw or (vw != 'blackbox' and H is None):
                continue
            u_ = Unit(z, H, y, tr, ev, cost_ret[tr], vw, args, log)
            vunits[vw] = u_
            certify_index(u_, u_.index_curves(ev, cost_ret[ev]), f'abl|{vw}', tg, Bs, cost_ret[tr], cost_ret[ev], T)
            certify_mix(u_.p_ev, tg, Bs, prefix=f'abl|{vw}|')
            save()
        if 'blackbox' in vunits and main.Hp is not None:
            bb = vunits['blackbox']
            cb = Unit(z, H, y, tr, ev, cost_ret[tr], args.view, args, log, base=main, ctrl=bb)       # controller black-box
            certify_index(cb, cb.index_curves(ev, cost_ret[ev]), 'abl|ctrlbb', tg, Bs, cost_ret[tr], cost_ret[ev], T)
            vb = Unit(z, H, y, tr, ev, cost_ret[tr], 'blackbox', args, log, base=bb, ctrl=main)      # verifier black-box
            certify_index(vb, vb.index_curves(ev, cost_ret[ev]), 'abl|verbb', tg, Bs, cost_ret[tr], cost_ret[ev], T)
            save()
        log('ablations')
    # ------------------------------------------------------------------ token budgets ------------------------------
    if 'tok' in parts:
        ctok = (tok_all / tokmean).astype(np.float32)
        # rollout.py fixes max_model_len=4096; use an a-priori bound, not a test-set maximum.
        tmax = float(4096 * T1)
        if np.any(np.diff(tok_all, axis=1, prepend=0) > 4096):
            raise ValueError('Recorded round exceeds the declared 4096-token context bound')
        tbud = [f * tokmean for f in TOKF]
        out['tok|budgets'] = np.array(tbud)
        tg = [('cs', 0.1), ('cs', 0.15), ('cs', 0.2), ('risk', 0.15)]
        tB = [(f, Bt, f) for f, Bt in zip(TOKF, tbud)]   # (training budget in mean-token units, evaluation budget, key)
        tu = Unit(z, H, y, tr, ev, ctok[tr], args.view, args, log, base=main)
        certify_index(tu, tu.index_curves(ev, ctok[ev]), 'tok|ours', tg, tB, ctok[tr], tok, tmax, cmax_tr=tmax / tokmean)
        tv = Unit(z, H, y, tr, ev, ctok[tr], args.view, args, log, kind='value', base=main)
        certify_index(tv, tv.value_curves(ev, ctok[ev]), 'tok|value', tg, tB, ctok[tr], tok, tmax, cmax_tr=tmax / tokmean)
        fixed = np.concatenate([np.repeat(np.arange(T, -1, -1)[None], N, 0), np.zeros((N, 1), int)], 1)
        bcs = dict(fixed=fixed, **{k: v for k, v in base_curves.items() if k != 'value'})
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
            du = Unit(z, H, y, tr, ev, cost_ret[tr], args.view, args, log, last=d, base=main)
            certify_index(du, du.index_curves(ev, cost_ret[ev]), f'dl{d}', tg, (B,), cost_ret[tr], cost_ret[ev], d)
            if theta_main is not None:     # the unconstrained index capped at d (answer at the cap)
                for t_ in tg:
                    S = np.minimum(curves[theta_main[(t_, B)]][1], d)
                    r = C.certify_curve(S, fpi.path_cost(S, cost_ret[ev]), d, p_ev, L, tok, cal_idx, tests, Ud, B, [t_])
                    out[f'cap{d}|{t_[0]}|{t_[1]}|B{B}'] = r[t_]
        log('deadline')
        save()
    save()
    for q in (0.1, 0.15, 0.2):
        for B in Bs[:3]:
            ks = ['ours'] + list(BASE)
            th_ = out.get(f'theta|ours|cs|{q}|B{B}', np.array([np.nan]))[0]
            print(ds, args.rtag, f'q{q} B{B} theta {th_}',
                  ' '.join(f"{k}:{100 * out[f'{k}|cs|{q}|B{B}'][:, 1].mean():.1f}" for k in ks if f'{k}|cs|{q}|B{B}' in out),
                  flush=True)
    log('done')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--traj'); ap.add_argument('--out'); ap.add_argument('--ds')
    ap.add_argument('--layer', type=int, default=18)
    ap.add_argument('--rtag', default='')
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--R', type=int, default=200)
    ap.add_argument('--Rsel', type=int, default=40)
    ap.add_argument('--nboot', type=int, default=200)
    ap.add_argument('--K', type=int, default=5)
    ap.add_argument('--M', type=int, default=5)
    ap.add_argument('--pca', type=int, default=128)
    ap.add_argument('--view', default='multi', choices=fpi.VIEWS)
    ap.add_argument('--abl_views', default='blackbox,noground')
    ap.add_argument('--parts', default='main,boot,peek,lab,abl,tok,dl')
    ap.add_argument('--vep', type=int, default=0, help='verifier epochs (0 = default); for smoke tests')
    ap.add_argument('--pep', type=int, default=None, help='policy epochs (None = default); for smoke tests')
    ap.add_argument('--max_train', type=int, default=0)
    ap.add_argument('--fresh_tag', default='', help="independent test: certify and test on {ds}<fresh_tag>_* only")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    run(args)


if __name__ == '__main__':
    main()

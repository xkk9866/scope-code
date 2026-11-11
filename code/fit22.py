"""Round 22, stage 1 (needs a GPU): train every network of one unit and one seed, plan every deployment curve, and
store the beliefs, curves and scores that stage 2 (main22.py, numpy only) certifies and analyses.

A unit is a reader on the mixed workload `mix3` (or on one benchmark).  Two resources are planned for: reads (one unit
per retrieval round) and tokens (prompt + generated tokens of the reader, in units of the mean full-depth cost).

SCOPE: cross-fitted verifier (belief p_t), kernel field (per-question Markov chain of the future belief cells, fitted
on belief paths alone), call payoff (p - theta)^+ at the strike theta = 1 - q of the target, exact backward induction
for every price of a grid.  Curves stored as S|<name>|<resource> [strike, question, price column] (int8).
  ours                the method
  pop                 population chain instead of the kernel field
  ctrlp / ctrlbb      the kernel field reads the belief path only / the scalar signals without the reader state
  so                  the reader state (and the round index) as the only observation of both networks
  bb / verbb          no reader state for both networks / for the verifier only
  digit, strike 0     digital payoff; expected belief (strike 0 is one of the stored strikes)
  myopic / dh / marg  one read ahead; deterministic horizons on the same chain; multi-horizon marginal forecaster
  admit               the plan fixed at admission: the horizon chosen from the round-0 kernels and never revised
  lsmc / rel          Longstaff-Schwartz value regression; label reliabilities in the payoff (reads only)
Published stopping rules: curve S|<rule> [question, column] and the rule's own confidence score own|<rule> [question,
round] (for certification by the rule's own score).  Triage: tri [depth, question] = a classifier on the round-0
observation predicts whether the answer after d reads is correct."""
import os, sys, argparse, time, copy
from types import SimpleNamespace
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F

sys.path.insert(0, os.path.dirname(__file__))
import fpi, cert14 as C, lattice16 as Lt, bpm20 as B
import baselines as BL
from main15 import load_unit, auroc, train_router
from main21 import Unit

DEV = fpi.DEV
TH = np.array([0.0, 0.5, 0.8, 0.85, 0.9])          # strikes: expected belief, the risk target, 1 - q for q = .2, .15, .1
OI = [int(round(100 * t)) for t in TH]


def fit_triage(x_tr, y_tr, x_ev, seed=0, M=3, epochs=80, H=128):
    """P(answer after d reads is correct | round-0 observation), an ensemble of M small networks."""
    out = []
    X = torch.as_tensor(x_tr, dtype=torch.float32, device=DEV); Y = torch.as_tensor(y_tr, dtype=torch.float32, device=DEV)
    Xe = torch.as_tensor(x_ev, dtype=torch.float32, device=DEV)
    for m in range(M):
        rng = np.random.RandomState(100 * seed + m); torch.manual_seed(100 * seed + m)
        ti, vi = fpi._split(len(y_tr), 0.15, rng)
        VI = torch.as_tensor(vi, device=DEV)
        net = nn.Sequential(nn.Dropout(0.25), nn.Linear(x_tr.shape[1], H), nn.GELU(), nn.Dropout(0.1), nn.Linear(H, H), nn.GELU(), nn.Linear(H, 1)).to(DEV)
        opt = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=1e-2)
        best, state, bad = np.inf, None, 0
        for ep in range(epochs):
            net.train(); perm = ti[rng.permutation(len(ti))]
            for k in range(0, len(perm), 128):
                b = torch.as_tensor(perm[k:k + 128], device=DEV)
                loss = F.binary_cross_entropy_with_logits(net(X[b]).squeeze(-1), Y[b]); opt.zero_grad(); loss.backward(); opt.step()
            net.eval()
            with torch.no_grad():
                vl = float(F.binary_cross_entropy_with_logits(net(X[VI]).squeeze(-1), Y[VI]))
            if vl < best - 1e-4:
                best, state, bad = vl, copy.deepcopy(net.state_dict()), 0
            else:
                bad += 1
                if bad >= 10:
                    break
        net.load_state_dict(state); net.eval()
        with torch.no_grad():
            out.append(torch.sigmoid(net(Xe).squeeze(-1)).cpu().numpy())
    return np.mean(out, 0)


def admit_curves(P0, b0, p0, U, kap, cum):
    """The plan fixed at admission.  P0 [N, T, K, K] round-0 kernels, U [Theta, K], cum[d] = resource of d reads.
    A question reads d*(kappa) = argmax_d E_0[U(cell at round d)] - kappa cum[d] times (d = 0: its payoff now)."""
    N, T, K, _ = P0.shape; Th = len(U); J = len(kap)
    m = np.zeros((N, K), np.float32); m[np.arange(N), b0] = 1
    val = np.zeros((Th, N, T + 1), np.float32)
    val[:, :, 0] = np.maximum(p0[None, :] - TH[:, None], 0)
    for d in range(1, T + 1):
        m = np.einsum('nk,nkl->nl', m, P0[:, d - 1].astype(np.float32))
        val[:, :, d] = (m @ U.T).T
    out = np.zeros((Th, N, J + 3), np.int8); out[:, :, 0] = T
    for j in range(J):
        out[:, :, j + 1] = (val - kap[j] * cum[None, None, :]).argmax(-1)
    return out


def run(args):
    t0 = time.time()
    ds, sd = args.ds, args.seed
    log = lambda m: print(ds, args.rtag, args.fresh_tag, m, f'{time.time() - t0:.0f}s', flush=True)
    z, H, rows, extra = load_unit(args.traj, ds, args.layer, args.fresh_tag)
    y = z['y'].astype(np.float32); split = z['split']
    tr, ev = np.where(split == 0)[0], np.where(split == 1)[0]
    N = len(ev); T1 = y.shape[1]; T = T1 - 1; yev = y[ev]
    grp = (np.asarray(z['grp']) if 'grp' in z and len(z['grp']) == len(y) else np.zeros(len(y), np.int8))[ev]
    tok_all = np.cumsum(np.array([[s['n_prompt_tok'] + s['n_gen_tok'] for s in r['steps']] for r in rows], float), 1)
    if np.any(np.diff(tok_all, axis=1, prepend=0) > 4096):
        raise ValueError('a recorded round exceeds the declared 4096-token context bound')
    tokmean = float(tok_all[tr, -1].mean())
    ctok = (tok_all / tokmean).astype(np.float32)
    cost_ret = np.repeat(np.arange(T1, dtype=np.float32)[None], len(y), 0)
    dc_r = np.ones(T, np.float32); dc_t = np.diff(ctok[tr].mean(0)).astype(np.float32)
    RES = {'r': (cost_ret, dc_r, B.price_grid(dc_r)), 't': (ctok, dc_t, B.price_grid(dc_t))}
    A_ = argparse.Namespace(pca=args.pca, seed=sd, K=args.K, vep=0, fep=0, cells=args.cells)
    out = {'meta|T': np.array([T]), 'meta|N': np.array([N, len(tr)]), 'meta|tokmean': np.array([tokmean]),
           'y': yev.astype(np.float32), 'yem': z['yem'][ev].astype(np.float32), 'grp': grp.astype(np.int8), 'tok': tok_all[ev].astype(np.float32),
           'ctok_tr_mean': ctok[tr].mean(0), 'dc|t': dc_t, 'kap|r': RES['r'][2], 'kap|t': RES['t'][2], 'TH': TH}
    if 'yf1' in z:
        out['yf1'] = z['yf1'][ev].astype(np.float32)
    if 'types' in z and len(z['types']) == len(y):
        out['types'] = np.asarray(z['types'])[ev].astype(str)
    path = os.path.join(args.out, f'{ds}{args.rtag}{args.fresh_tag}_s{sd}.npz')

    def save():
        np.savez_compressed(path, **out)

    def plans(tag, P, b, p, U, **kw):
        for c, (_, dc, kap) in RES.items():
            out[f'S|{tag}|{c}'] = B.plan(P, b, p, U, TH, kap, dc, **kw)

    # ------------------------------------------------------------------ the method -----------------------------------
    t1 = time.time()
    main = Unit(z, H, y, tr, ev, 'state', A_, log)
    out['meta|train_time'] = np.array([time.time() - t1])
    p_ev, p_tr, b_ev = main.p_ev, main.p_tr, main.b_ev
    hp = lambda idx: None if main.Hp is None else main.Hp[idx]
    U, Fq = main.call[OI], main.frac[OI]
    out.update({'p': p_ev.astype(np.float32), 'b': b_ev.astype(np.int8), 'call': U, 'frac': Fq, 'edges': main.lat.edges, 'mid': main.lat.mid,
                'Kt': np.asarray(main.lat.Kt, np.float32), 'P0': main.P_ev[:, 0].astype(np.float16),
                'meta|auc': np.array([auroc(yev[:, t], p_ev[:, t]) for t in range(T1)]), 'meta|acc': yev.mean(0),
                'meta|any': np.array([yev.max(1).mean()])})
    Pp = B.pop_chain(main.lat, N)
    out['meta|path_nll'] = np.array(list(B.path_nll(main.P_ev, b_ev)) + list(B.path_nll(Pp, b_ev)))
    log('AUROC per round ' + ' '.join(f'{a:.3f}' for a in out['meta|auc']) + ' | path NLL learned %.4f %.4f population %.4f %.4f' % tuple(out['meta|path_nll']))
    t1 = time.time()
    plans('ours', main.P_ev, b_ev, p_ev, U)
    out['meta|plan_time'] = np.array([(time.time() - t1) / (2 * len(TH))])
    out['meta|nested'] = np.array([min(B.nested_check(S) for S in out['S|ours|r']), min(B.nested_check(S) for S in out['S|ours|t'])])
    log(f'curves planned, nested {out["meta|nested"]}')
    plans('pop', Pp, b_ev, p_ev, U); del Pp
    plans('digit', main.P_ev, b_ev, p_ev, Fq, payoff='digital')
    plans('myopic', main.P_ev, b_ev, p_ev, U, myopic=True)
    for c, (cost, dc, kap) in RES.items():
        out[f'S|dh|{c}'] = B.plan_dh(main.P_ev, b_ev, p_ev, U, TH, kap, cost[ev])
        out[f'S|admit|{c}'] = admit_curves(main.P_ev[:, 0], b_ev[:, 0], p_ev[:, 0], U, kap, np.r_[0.0, np.cumsum(dc)])
    log('pop / digital / myopic / dh / admit'); save()
    # ------------------------------------------------------------------ published rules, triage ----------------------
    names = list(z['u_names']); col = lambda n: np.asarray(z['u'][:, :, names.index(n)], np.float32)
    curve = lambda s_ev, s_tr: C.first_cross_path(s_ev, C.thr_desc(s_tr[:, :T]), T).astype(np.int8)
    for k, s in (('flare', np.exp(col('lp_min'))), ('dragin', -col('ent_mean')), ('seakr', col('cons_soft')), ('ircot', col('p_suff'))):
        out[f'S|{k}'] = curve(s[ev], s[tr]); out[f'own|{k}'] = s[ev].astype(np.float32)
    x0 = np.concatenate([main.Hp[:, 0], main.X[:, 0]], -1)
    need = np.where(y.max(1) > 0, y.argmax(1), T)
    out['S|adapt'] = BL.adaptive_rag_curve(x0[tr], need[tr], x0[ev], T, sd, args.M, T1).astype(np.int8)
    fold = main.V.fold
    oof = np.zeros(y[tr].shape, np.float32); nets = []
    for k in range(int(fold.max()) + 1):
        net = BL._fit_probe(hp(tr)[fold != k], y[tr][fold != k], 1000 * sd + k, 30, DEV); nets.append(net)
        oof[fold == k] = BL._probe_pred(net, hp(tr)[fold == k], DEV)
    pe = np.mean([BL._probe_pred(n, hp(ev), DEV) for n in nets], 0)
    out['S|probe'] = curve(pe, oof); out['own|probe'] = pe.astype(np.float32)
    nets = [BL.train_stoprag(hp(tr), main.X[tr], y[tr], seed=1000 * sd + m, dev=DEV) for m in range(args.M)]
    ie = [BL.stoprag_index(n, hp(ev), main.X[ev], dev=DEV) for n in nets]
    it = np.mean([BL.stoprag_index(n, hp(tr), main.X[tr], dev=DEV)[0] for n in nets], 0)
    out['S|stoprag'] = curve(np.mean([a for a, _ in ie], 0), it); out['own|stoprag'] = np.mean([q for _, q in ie], 0).astype(np.float32)
    if extra.get('halt') is not None:
        hs = np.nan_to_num(extra['halt'][ev].astype(float), nan=0.0)
        out['S|halt'] = C.first_cross_path(hs, np.r_[1.01, np.linspace(1.0, 0.0, 160)], T).astype(np.int8); out['own|halt'] = hs.astype(np.float32)
    del nets
    tri = np.stack([fit_triage(x0[tr], y[tr][:, d], x0[ev], seed=sd) for d in range(1, T + 1)])
    out['tri'] = tri.astype(np.float32)
    out['meta|tri_auc'] = np.array([auroc(yev[:, d], tri[d - 1]) for d in range(1, T + 1)])
    log('published rules and triage; triage AUROC ' + ' '.join(f'{a:.3f}' for a in out['meta|tri_auc'])); save()
    if args.light:
        log('done (light)'); return
    # ------------------------------------------------------------------ the law ---------------------------------------
    cpn = B.CrossPath(None, main.X[tr][..., :0], p_tr, main.lat, fold, seed=sd)
    Pn = cpn(None, main.X[ev][..., :0], p_ev)
    out['meta|path_nll|ctrlp'] = np.array(list(B.path_nll(Pn, b_ev)))
    plans('ctrlp', Pn, b_ev, p_ev, U); del cpn, Pn
    log('belief-only law'); save()
    # ------------------------------------------------------------------ the observation -------------------------------
    so = Unit(z, H, y, tr, ev, 'state', A_, log, obs=SimpleNamespace(X=main.X[..., 5:6], Hp=main.Hp))
    out['p|so'] = so.p_ev.astype(np.float32); out['meta|auc|so'] = np.array([auroc(yev[:, t], so.p_ev[:, t]) for t in range(T1)])
    plans('so', so.P_ev, so.b_ev, so.p_ev, so.call[OI]); del so
    log('state only'); save()
    bb = Unit(z, H, y, tr, ev, 'blackbox', A_, log)
    out['p|bb'] = bb.p_ev.astype(np.float32); out['meta|auc|bb'] = np.array([auroc(yev[:, t], bb.p_ev[:, t]) for t in range(T1)])
    plans('bb', bb.P_ev, bb.b_ev, bb.p_ev, bb.call[OI])
    cb = Unit(z, H, y, tr, ev, 'state', A_, log, base=main, ctrl=bb)
    plans('ctrlbb', cb.P_ev, b_ev, p_ev, U); del cb
    vb = Unit(z, H, y, tr, ev, 'blackbox', A_, log, base=bb, ctrl=main)
    plans('verbb', vb.P_ev, bb.b_ev, bb.p_ev, bb.call[OI]); del vb, bb
    log('reader state 2 x 2'); save()
    # ------------------------------------------------------------------ the price -------------------------------------
    fcs = Lt.CrossForecast(hp(tr), main.X[tr], p_tr, y[tr], main.lat, fold, seed=sd)
    F_ev = fcs(hp(ev), main.X[ev], p_ev)
    for c, (cost, dc, kap) in RES.items():
        out[f'S|marg|{c}'] = np.stack([Lt.index_curve(Lt.fair_price(main.lat, F_ev, b_ev, p_ev, th, th, cost[ev], payoff='belief'), kap, T) for th in TH]).astype(np.int8)
    del fcs, F_ev
    log('marginal forecaster'); save()
    pol = fpi.CrossPolicy(hp(tr), main.X[tr], p_tr, y[tr], cost_ret[tr], fold, kind='value', seed=sd)
    kaps = pol.ps * np.concatenate([-np.geomspace(0.5, 1e-3, 40), [0.0], np.geomspace(1e-3, 1.0, 120)])
    out['S|lsmc|r'] = np.stack([fpi.value_path(pol(hp(ev), main.X[ev], p_ev, cost_ret[ev], [th], kaps)[:, 0], p_ev, th, kaps, T) for th in TH]).astype(np.int8)
    del pol
    log('value regression'); save()
    import bpm20rel as BR
    cpr = BR.CrossPath(hp(tr), main.X[tr], p_tr, y[tr], main.lat, fold, seed=sd)
    Pr, Ar = cpr(hp(ev), main.X[ev], p_ev)
    out['S|rel|r'] = BR.plan(Pr, Ar, b_ev, p_ev, Fq, TH, RES['r'][2], dc_r)
    del cpr, Pr, Ar
    log('label reliabilities'); save()
    log('done')


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--traj'); ap.add_argument('--out'); ap.add_argument('--ds'); ap.add_argument('--layer', type=int, default=18)
    ap.add_argument('--rtag', default=''); ap.add_argument('--fresh_tag', default=''); ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--K', type=int, default=5); ap.add_argument('--M', type=int, default=5)
    ap.add_argument('--pca', type=int, default=128); ap.add_argument('--cells', type=int, default=20)
    ap.add_argument('--light', type=int, default=0)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    run(args)

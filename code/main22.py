"""Round 22, stage 2 (numpy; no training): certify and analyse the curves stored by fit22.py for one unit and seed.

Protocol of every random split of an evaluation pool: calibration half (first half A: answer certificate, second half
B: budget certificate), test half.  Resources: 'r' reads, 't' tokens (budget T<d> = mean tokens of the fixed horizon
with d reads on the training pool, in units of the mean full-depth cost).

The strike of the call is the second multiplier of the Lagrangian and is calibrated like the price.  A priced policy
(SCOPE and every control that prices a payoff) is a family of curves, one per strike of the declared grid TH.  B is cut
in two parts.  On the first, every strike is deployed (price certified on that part) and the strike with the most
answers admitted by the certificate of the target, calibrated on that same part, is kept; the price of the kept strike
is certified on the second part; the answers are certified on A.  A policy with no strike to choose (the fixed horizon,
the published rules, triage, a variant with a fixed strike) certifies its price on all of B.  `ours1q` is SCOPE with
the strike fixed at 1 - q (0.5 for the risk target).

Row of a certified policy on a split:
  [0 risk E[answered and wrong], 1 certified coverage, 2 mean reads, 3 mean tokens, 4 error among the answered,
   5 resource spent (budget units), 6 coverage of correct answers]
Parts
  main  : SCOPE, the fixed horizon, the eight published rules (certified by the shared verifier and by their own score),
          triage at every depth and the admission planner; batch error targets, the risk target, the PAC certificate
  abl   : controls of the law, the observation, the payoff and the price
  oracle: label hindsight
  wl    : evaluation workloads of a mixed unit (every benchmark alone, compositions), own splits
  cm    : the kernel field as a cost model (spend forecast from round 0; price moved across workloads)
  boot  : bootstrap over questions of the split-averaged coverage (differences between policies)
  perq  : per-question certification frequency and reads (flows, question types)
  plain : accuracy against the resource with no certificate"""
import os, sys, argparse, time
import numpy as np, torch

sys.path.insert(0, os.path.dirname(__file__))
import fpi, cert14 as C, bpm20 as B
import main21
from main21 import workloads

PARTS = ('main', 'cap', 'perq', 'abl', 'oracle', 'wl', 'cm', 'boot', 'reach', 'calsize', 'peek', 'vcert', 'plain')


def stamp(args):
    """Version of the code (this file and the modules it certifies with) and the arguments that fix the counts a
    table states.  Written per part, so that a result file assembled from several runs shows it."""
    import hashlib
    h = hashlib.md5()
    for f in (__file__, fpi.__file__, C.__file__, B.__file__, main21.__file__):
        h.update(open(os.path.abspath(f), 'rb').read())
    return (f'code={h.hexdigest()[:10]};R={args.R};Rw={args.Rw};Rb={args.Rb};nboot={args.nboot};Rp={min(args.R, RPLAIN)}'
            f';delta={DELTA};pgrid={len(PGRID)};pgrid_lo={PGRID[0]};pgrid_hi={PGRID[-1]};nth={len(TH)}')


QS = (0.1, 0.15, 0.2)
SGRID = C.strike_grid()
PGRID = np.round(np.arange(0.50, 0.995, 0.01), 2)          # thresholds of the PAC certificate (declared in advance)
DELTA = 0.1
RPLAIN = 50                                                # splits of the accuracy curves with no certificate
BASE = ('ircot', 'flare', 'adapt', 'dragin', 'seakr', 'probe', 'stoprag', 'halt')
DEPTHS = (1.0, 1.5, 2.0, 2.5, 3.0)
MAINB = ('1.0', '1.5', '2.0')                              # budgets of the tables (the fine grid of `reach` spans 0.5 to 3)
OS = {0.0: 0, 0.5: 1, 0.8: 2, 0.85: 3, 0.9: 4}
TH = tuple(OS)                                             # the strike grid of the fit caches (asserted against D['TH'])


def strike_of(tg):
    return OS[0.5] if tg[0] == 'risk' else OS[round(1 - tg[1], 2)]


def pac_threshold(s, Lw, A, q, delta=DELTA, grid=PGRID):
    """Selective risk with high probability: P(error among answers with score >= threshold  <=  q) >= 1 - delta.
    Exact binomial tail test of every threshold of a grid declared in advance, Bonferroni over the grid; the lowest
    certified threshold is released (inf when none)."""
    from scipy.stats import binom
    sA, wA = s[A], Lw[A] > 0
    ge = sA[:, None] >= grid[None, :]
    n = ge.sum(0); k = (ge & wA[:, None]).sum(0)
    ok = (binom.cdf(k, n, q) <= delta / len(grid)) & (n > 0)
    return float(grid[ok].min()) if ok.any() else np.inf


def answer(tg, s, Lw, A, te):
    if tg[0] == 'cs':
        return C.conf_select(s, Lw, A, te, tg[1])
    if tg[0] == 'risk':
        return s[te] >= C.crc_strike(s, Lw, A, SGRID, tg[1])
    if tg[0] == 'pac':
        return s[te] >= pac_threshold(s, Lw, A, tg[1])
    raise ValueError(tg)


def fixed_curve(N, T):
    return np.concatenate([np.repeat(np.arange(T, -1, -1)[None], N, 0), np.zeros((N, 1), int)], 1).astype(np.int8)


def triage_curve(s, d):
    thr = np.r_[-np.inf, np.unique(np.quantile(s, np.linspace(0, 1, 200))), np.inf]
    return np.concatenate([np.where(s[:, None] >= thr[None, :], d, 0), np.zeros((len(s), 1), int)], 1).astype(np.int8)


def planner_curve(val, kap, cum, T):
    """Admission planner on label-supervised forecasts val [N, T + 1] (probability that the answer after d reads is
    correct): d*(kappa) = argmax_d val_d - kappa cum_d; column 0 = full depth, last columns stop at once / abstain."""
    N = len(val)
    S = np.zeros((N, len(kap) + 3), np.int8); S[:, 0] = T
    for j, k in enumerate(kap):
        S[:, j + 1] = (val - k * cum[None, :]).argmax(1)
    return S


class Pool:
    """An evaluation pool (or a workload cut from it) with its own random splits and dither."""

    def __init__(self, D, sub, seed, R):
        self.N = len(sub); self.sub = sub; self.ar = np.arange(self.N)
        self.y = D['y'][sub]; self.L = 1 - self.y; self.p = D['p'][sub]; self.tok = D['tok'][sub]
        T1 = self.y.shape[1]; self.T = T1 - 1
        tokmean = float(D['meta|tokmean'][0])
        self.cost = {'r': np.repeat(np.arange(T1, dtype=np.float32)[None], self.N, 0), 't': (self.tok / tokmean).astype(np.float32)}
        self.cmax = {'r': float(self.T), 't': 4096.0 * T1 / tokmean}
        self.bud = {'r': {f'B{d}': d for d in DEPTHS}, 't': {f'T{d}': float(np.interp(d, np.arange(T1), D['ctok_tr_mean'])) for d in DEPTHS}}
        rng = np.random.RandomState(123 + seed)
        self.perms = [rng.permutation(self.N) for _ in range(R)]
        self.Ud = np.random.RandomState(999).rand(self.N)
        self.R = R
        self._dep = {}

    def split(self, r):
        pm = self.perms[r]; cal = pm[: self.N // 2]
        return cal[: len(cal) // 2], cal[len(cal) // 2:], pm[self.N // 2:]

    def prep(self, S, c):
        cst = fpi.path_cost(S, self.cost[c]).astype(np.float32)
        return S, cst, C.env_cols(cst)

    def deploy(self, pr, c, Bq, Bh):
        S, cst, cenv = pr
        j, om = C.budget_index_env(S, cenv, self.Ud, Bh, Bq, self.cmax[c], cost_is_envelope=True)
        st, ab, col = C.deployed(S, self.Ud, j, om, True)
        return st.astype(np.int64), ab, cst[self.ar, col]

    def row(self, st, ab, spent, score, tg, A, te):
        s = np.where(ab, -np.inf, score[self.ar, st]); Lw = self.L[self.ar, st]
        sel = answer(tg, s, Lw, A, te)
        n = max(sel.sum(), 1); w = Lw[te]
        return [float(np.mean(sel * w)), float(sel.mean()), float(st[te].mean()), float(np.where(ab, 0.0, self.tok[self.ar, st])[te].mean()),
                float((sel * w).sum() / n), float(spent[te].mean()), float(np.mean(sel * (1 - w)))], sel

    def stopped(self, pr, c, Bq, idx, score):
        """stopped scores and losses of a curve whose price is certified on idx."""
        st, ab, _ = self.deploy(pr, c, Bq, idx)
        return np.where(ab, -np.inf, score[self.ar, st]), self.L[self.ar, st]

    def pick(self, stage, tg, idx):
        """index of the strike whose policy has the most answers admitted on idx by the certificate of target tg,
        calibrated on idx itself (the labels of idx and of no other question); ties go to the highest strike."""
        n = [int(answer(tg, s, Lw, idx, idx).sum()) for s, Lw in stage]
        return len(n) - 1 - int(np.argmax(n[::-1]))

    def deployer(self, pr, c, Bq, Bh, score=None):
        """target -> (st, ab, spent, index of the strike or -1) of a policy on a split with budget part Bh.
        A curve: price certified on Bh.  Fam: strike calibrated on the first half of Bh, price of that strike certified
        on the second half.  At1q: the curve of the strike 1 - q of the target, price certified on Bh."""
        memo = {}
        if isinstance(pr, At1q):
            def get(tg):
                o = strike_of(tg)
                if o not in memo:
                    memo[o] = self.deploy(pr[o], c, Bq, Bh)
                return memo[o] + (o,)
            return get
        if isinstance(pr, Fam):
            Bs, Bk = Bh[: len(Bh) // 2], Bh[len(Bh) // 2:]
            stage = [self.stopped(q_, c, Bq, Bs, self.p if score is None else score) for q_ in pr]

            def get(tg):
                o = self.pick(stage, tg, Bs)
                if o not in memo:
                    memo[o] = self.deploy(pr[o], c, Bq, Bk)
                return memo[o] + (o,)
            return get
        d = self.deploy(pr, c, Bq, Bh) + (-1,)
        return lambda tg: d


class Fam(list):
    """the prepared curves of one priced policy over the strike grid TH: the strike is calibrated."""


class At1q(list):
    """the same curves read at the strike 1 - q of the target (0.5 for the risk target): nothing is calibrated."""


def simulate0(P0, b0, p0, U, theta, kap, dc, pol=None, bs=512, dev='cpu'):
    """Forward simulation of the priced policy from round 0 alone.  P0 [N, T, K, K]: the kernels of the steps 0->1, ...,
    T-1->T as the field emits them at round 0 (the dynamics); pol: the kernels that define the stopping regions
    (default P0).  Returns the expected number of reads at every price [N, J] and, per step, the probability that the
    question is still reading [N, J, T]."""
    N, T, K, _ = P0.shape; J = len(kap)
    Ut = torch.as_tensor(U, dtype=torch.float32, device=dev)
    ka = torch.as_tensor(np.asarray(kap), dtype=torch.float32, device=dev)
    dct = torch.as_tensor(np.asarray(dc), dtype=torch.float32, device=dev)
    Rd = np.zeros((N, J, T), np.float32)
    with torch.no_grad():
        for i0 in range(0, N, bs):
            Pd = torch.as_tensor(P0[i0:i0 + bs].astype(np.float32), device=dev)
            Pp = Pd if pol is None else torch.as_tensor(pol[i0:i0 + bs].astype(np.float32), device=dev)
            bb = torch.as_tensor(b0[i0:i0 + bs].astype(np.int64), device=dev)
            pb = torch.as_tensor(p0[i0:i0 + bs], dtype=torch.float32, device=dev)
            n = len(pb); arn = torch.arange(n, device=dev)
            Ue = Ut[None, None, :].expand(n, J, K)
            V = Ue; stop = [None] * (T + 1)
            for tau in range(T - 1, 0, -1):
                cont = -ka[None, :, None] * dct[tau] + torch.einsum('nkl,njl->njk', Pp[:, tau], V)
                stop[tau] = ~(cont > Ue)
                V = torch.maximum(Ue, cont)
            cont0 = -ka[None, :] * dct[0] + torch.einsum('nl,njl->nj', Pp[:, 0][arn, bb], V)
            read0 = (cont0 > (pb - theta).clamp(min=0)[:, None]).float()
            Rd[i0:i0 + n, :, 0] = read0.cpu().numpy()
            alive = Pd[:, 0][arn, bb][:, None, :] * read0[:, :, None]
            for tau in range(1, T):
                cm = alive * (~stop[tau]).float()
                Rd[i0:i0 + n, :, tau] = cm.sum(-1).cpu().numpy()
                alive = torch.einsum('njk,nkl->njl', cm, Pd[:, tau])
    return Rd.sum(-1), Rd


def run(args):
    t0 = time.time()
    name = f'{args.ds}{args.rtag}{args.fresh_tag}_s{args.seed}'
    log = lambda m: print(name, m, f'{time.time() - t0:.0f}s', flush=True)
    parts = set(args.parts.split(','))
    if parts - set(PARTS):
        raise ValueError(f'unknown parts {sorted(parts - set(PARTS))}')
    run_stamp = stamp(args)                       # (st is the name of the stop times below)
    D = dict(np.load(os.path.join(args.cache, name + '.npz'), allow_pickle=False))
    N, T = int(D['meta|N'][0]), int(D['meta|T'][0]); T1 = T + 1; sd = args.seed
    assert np.allclose(D['TH'], TH), D['TH']
    grp = D['grp'].astype(int); G = int(grp.max()) + 1
    kap = {'r': D['kap|r'], 't': D['kap|t']}
    dcs = {'r': np.ones(T, np.float32), 't': D['dc|t']}
    TG = [('cs', q) for q in QS] + [('risk', 0.15), ('pac', 0.15), ('pac', 0.2)]
    path = os.path.join(args.out, name + '_r22.npz')
    out = dict(np.load(path, allow_pickle=False)) if (args.merge and os.path.exists(path)) else {}
    out.update({k: v for k, v in D.items() if k.startswith('meta|')})
    out['meta|grp'] = np.bincount(grp, minlength=G)
    rec = {}

    def add(key, row):
        rec.setdefault(key, []).append(row)

    def save():
        for k, v in rec.items():
            out[k] = np.array(v, np.float32)
        np.savez_compressed(path, **out)

    def tkey(tg):
        return f'{tg[0]}|{tg[1]}'

    def family(Dd, pool, k, c):
        assert len(Dd[f'S|{k}|{c}']) == len(TH), (k, c, Dd[f'S|{k}|{c}'].shape)
        return [pool.prep(Dd[f'S|{k}|{c}'][o][pool.sub], c) for o in range(len(TH))]

    def policies(Dd, pool, c, base=True, own=True, tri=True, cap=False):
        """{name: (policy, score or None)} of every policy compared on resource c.  policy: a prepared curve, a Fam
        (strike calibrated) or an At1q (strike 1 - q).  own: the published rules certified by their own confidence
        score; cap: the published rules with a depth cap."""
        sub = pool.sub; pols = {}
        ours = family(Dd, pool, 'ours', c)
        pols['ours'] = (Fam(ours), None)
        pols['ours1q'] = (At1q(ours), None)
        pols['fixed'] = (pool.prep(fixed_curve(pool.N, T), c), None)
        if base:
            for k in BASE:
                if f'S|{k}' in Dd:
                    pr = pool.prep(Dd[f'S|{k}'][sub], c)
                    pols[k] = (pr, None)
                    if own and f'own|{k}' in Dd:
                        pols[f'{k}@own'] = (pr, Dd[f'own|{k}'][sub])
                    if cap:
                        for d in range(1, T):
                            pols[f'{k}@cap{d}'] = (pool.prep(np.minimum(Dd[f'S|{k}'][sub], d), c), None)
        if tri:
            cum = np.r_[0.0, np.cumsum(dcs[c])]
            for d in range(1, T + 1):
                pols[f'tri{d}'] = (pool.prep(triage_curve(Dd['tri'][d - 1][sub], d), c), None)
            val = np.concatenate([Dd['p'][sub][:, :1], Dd['tri'][:, sub].T], 1)
            pols['triP'] = (pool.prep(planner_curve(val, kap[c], cum, T), c), None)
            pols['admit'] = (Fam(family(Dd, pool, 'admit', c)), None)
        return pols

    def certify(tag, pool, pols, c, targets, buds=None, keep=None):
        """rows of every policy for every split, budget and target; keep: {(policy, budget key, target key): [sum of
        certified indicator, sum of reads, count in test] per question} filled when given.  strike|...: the strike
        that a family calibrated on the split."""
        for bk, Bq in pool.bud[c].items():
            if buds is not None and bk not in buds:
                continue
            for r in range(pool.R):
                A, Bh, te = pool.split(r)
                for k, (pr, score) in pols.items():
                    sc = pool.p if score is None else score
                    dep = pool.deployer(pr, c, Bq, Bh, sc)
                    for tg in targets:
                        if (score is not None or '@cap' in k) and tg[0] != 'cs':
                            continue
                        st, ab, spent, o = dep(tg)
                        row, sel = pool.row(st, ab, spent, sc, tg, A, te)
                        add(f'{tag}{k}|{tkey(tg)}|{bk}', row)
                        if isinstance(pr, Fam):
                            add(f'strike|{tag}{k}|{tkey(tg)}|{bk}', TH[o])
                        kk = (k, bk, tkey(tg))
                        if keep is not None and kk in keep:
                            keep[kk][te, 0] += sel; keep[kk][te, 1] += st[te]; keep[kk][te, 2] += 1

    pool = Pool(D, np.arange(N), sd, args.R)

    def hindsight_curve(th, c):
        """Label hindsight at strike th.  A budget is a cap: beyond what its lowest price spends the hindsight reads no
        further (column 0 is the lowest-price plan, not full depth)."""
        Sh = B.hindsight(D['p'], D['y'], pool.cost[c], th, kap[c], T).astype(np.int8)
        Sh[:, 0] = Sh[:, 1]
        return Sh

    out['meta|budgets_t'] = np.array([pool.bud['t'][f'T{d}'] for d in DEPTHS]); out['meta|depths'] = np.array(DEPTHS)
    out['meta|tok_depth'] = pool.tok.mean(0)
    # ------------------------------------------------------------------ main -----------------------------------------
    if 'main' in parts:
        keep = {(k, bk, 'cs|0.15'): np.zeros((N, 3)) for k in ('ours', 'fixed', 'stoprag', 'tri2') for bk in ('B1.0', 'B2.0', 'T1.0', 'T2.0')} if 'perq' in parts else None
        for c in ('t', 'r'):
            mb = [b for b in pool.bud[c] if b[1:] in MAINB]
            certify('', pool, policies(D, pool, c, cap='cap' in parts), c, TG, buds=mb, keep=keep)
            cv = lambda k, bk: 100 * np.mean(np.array(rec[f'{k}|cs|0.15|{bk}'])[:, 1])
            log(f'main [{c}]: q=.15 ours/ours1q/fixed (mean strike) ' + ' '.join(
                f"{bk} {cv('ours', bk):.1f}/{cv('ours1q', bk):.1f}/{cv('fixed', bk):.1f} ({np.mean(rec[f'strike|ours|cs|0.15|{bk}']):.2f})" for bk in mb))
            save()
        if keep is not None:
            for (k, bk, tk), v in keep.items():
                out[f'perq|{k}|{tk}|{bk}'] = v.astype(np.float32)
            if 'types' in D:
                out['perq|types'] = D['types']
            out['perq|grp'] = grp.astype(np.int8); out['perq|y'] = D['y']
            save()
    # ------------------------------------------------------------------ ablations ------------------------------------
    if 'abl' in parts:
        # every control keeps the protocol of SCOPE (strike calibrated on the first part of B, price on the second); a
        # control scored by another verifier calibrates and certifies with that score.  th<100 strike>: SCOPE with the
        # strike fixed and the price certified on all of B (th0: the expected accuracy as the payoff).
        tg = ('cs', 0.15)
        for c in ('t', 'r'):
            pols = {}
            for k in ('pop', 'ctrlp', 'ctrlbb', 'so', 'bb', 'verbb', 'digit', 'myopic', 'dh', 'marg', 'lsmc', 'rel'):
                if f'S|{k}|{c}' in D:
                    sc = D['p|so'] if k == 'so' else D['p|bb'] if k in ('bb', 'verbb') else None
                    pols[k] = (Fam(family(D, pool, k, c)), sc)
            for th_ in TH:
                pols[f'th{int(round(100 * th_))}'] = (pool.prep(D[f'S|ours|{c}'][OS[th_]], c), None)
            for k in ('so', 'bb'):
                if f'p|{k}' in D:
                    pols[f'fixed@{k}'] = (pool.prep(fixed_curve(N, T), c), D[f'p|{k}'])
            certify('abl|', pool, pols, c, [tg], buds=[b for b in pool.bud[c] if b[1:] in ('1.0', '1.5', '2.0')])
            log(f'ablations [{c}]'); save()
    # ------------------------------------------------------------------ label hindsight ------------------------------
    if 'oracle' in parts:
        for c in ('t', 'r'):
            for tg in [('cs', q) for q in QS]:
                th = round(1 - tg[1], 2)
                certify('', pool, {'oracle': (pool.prep(hindsight_curve(th, c), c), None)}, c, [tg])
        log('hindsight'); save()
    # ------------------------------------------------------------------ workloads ------------------------------------
    if 'wl' in parts and G > 1:
        tg = ('cs', 0.15)
        for wn, sub in workloads(grp, sd).items():
            w = Pool(D, sub, sd, args.Rw)
            out[f'wl|{wn}|n'] = np.bincount(grp[sub], minlength=G)
            for c in ('t', 'r'):
                pols = policies(D, w, c, own=False, tri=False)
                pols['tri2'] = (w.prep(triage_curve(D['tri'][1][sub], 2), c), None)
                certify(f'wl|{wn}|', w, pols, c, [tg], buds=('B1.0', 'B1.5', 'B2.0', 'T1.0', 'T1.5', 'T2.0'))
        log('workloads'); save()
    # ------------------------------------------------------------------ the kernel field as a cost model ------------
    if 'cm' in parts:
        o = OS[0.85]; U = D['call'][o]; P0 = D['P0']; b0 = D['b'][:, 0].astype(int); p0 = D['p'][:, 0]
        Pp = np.repeat(D['Kt'][None], N, 0)
        wls = workloads(grp, sd) if G > 1 else {}
        wn = [k for k in wls if k.startswith('h')] + [k for k in wls if k.startswith('g')]
        for c in ('r', 't'):
            J = len(kap[c]); S = D[f'S|ours|{c}'][o][:, 1:J + 1].astype(int)
            base = float(D['ctok_tr_mean'][0]) if c == 't' else 0.0
            real = pool.cost[c][pool.ar[:, None], S]                                               # realised spend [N, J]
            preds = {}
            for nm, kw in (('field', {}), ('pop', dict(pol=P0))):
                _, Rd = simulate0(P0 if nm == 'field' else Pp, b0, p0, U, 0.85, kap[c], dcs[c], **kw)
                preds[nm] = base + (Rd * dcs[c][None, None, :]).sum(-1)
            mr = real.mean(0)
            rng_ = (mr >= pool.bud[c][('B' if c == 'r' else 'T') + '1.0'] * 0.25) & (mr <= pool.bud[c][('B' if c == 'r' else 'T') + '3.0'])
            for nm, pr in preds.items():
                out[f'cm|curve|{nm}|{c}'] = np.stack([mr, pr.mean(0)])
                out[f'cm|gap|{nm}|{c}'] = np.array([np.abs(pr.mean(0) - mr)[rng_].mean(), np.abs(pr.mean(0) - mr)[rng_].max(), np.mean(np.abs(pr.mean(0) - mr)[rng_] / mr[rng_])])
            col_at = lambda curve, Bq: int(np.where(curve <= Bq)[0][0]) if (curve <= Bq).any() else len(curve) - 1
            for bk in (('B1.0', 'B1.5', 'B2.0') if c == 'r' else ('T1.0', 'T1.5', 'T2.0')):
                Bq = pool.bud[c][bk]; j = col_at(mr, Bq)
                out[f'cm|point|{c}|{bk}'] = np.array([mr[j]] + [preds[nm][:, j].mean() for nm in preds] + [np.corrcoef(preds[nm][:, j], real[:, j])[0, 1] for nm in preds])
                out[f'cm|grp|{c}|{bk}'] = np.array([[real[grp == g, j].mean()] + [preds[nm][grp == g, j].mean() for nm in preds] for g in range(G)])
                if wn:
                    res = {k: [] for k in ('static', 'field', 'pop', 'own')}
                    for a in wn:
                        ja = col_at(real[wls[a]].mean(0), Bq)
                        for b_ in wn:
                            if a == b_:
                                continue
                            sb = wls[b_]
                            res['static'].append(real[sb, ja].mean() / Bq - 1)
                            res['own'].append(real[sb, col_at(real[sb].mean(0), Bq)].mean() / Bq - 1)
                            for nm in preds:
                                res[nm].append(real[sb, col_at(preds[nm][sb].mean(0), Bq)].mean() / Bq - 1)
                    for k, v in res.items():
                        out[f'cm|move|{k}|{c}|{bk}'] = np.array(v, np.float32)
                    # the same shift for the certified coverage: coverage on the target workload at the moved price (labels of
                    # the target are used for the answer certificate only)
        log('cost model: ' + ' | '.join(f"{c} gap field {out[f'cm|gap|field|{c}'][2]:.3f} pop {out[f'cm|gap|pop|{c}'][2]:.3f}" for c in ('r', 't'))
            + (' ; move B1 static %.3f field %.3f pop %.3f own %.3f' % tuple(np.abs(out[f'cm|move|{k}|r|B1.0']).mean() for k in ('static', 'field', 'pop', 'own')) if wn else ''))
        save()
    # ------------------------------------------------------------------ bootstrap over questions ---------------------
    if 'boot' in parts:
        tg = ('cs', 0.15)
        brng = np.random.RandomState(9000 + sd)
        names = ('ours', 'ours1q', 'fixed', 'stoprag', 'halt', 'ircot', 'tri1', 'tri2', 'tri3', 'triP', 'admit')
        P = {c: {k: v for k, v in policies(D, pool, c, own=False).items() if k in names} for c in ('t', 'r')}
        keys = [(c, bk) for c in ('t', 'r') for bk in list(pool.bud[c])[:3]]
        acc = {(c, bk): np.zeros((args.nboot, len(names))) for c, bk in keys}
        for (c, bk), v in acc.items():           # a policy that this unit does not have is nan, not a coverage of zero
            v[:, [i for i, k in enumerate(names) if k not in P[c]]] = np.nan
        for b in range(args.nboot):
            idx = brng.randint(0, N, N)
            for r in range(args.Rb):
                pm = idx[brng.permutation(N)]
                A, Bh, te = pm[: N // 4], pm[N // 4: N // 2], pm[N // 2:]
                for c, bk in keys:
                    for i, k in enumerate(names):
                        if k not in P[c]:
                            continue
                        st, ab, _, _ = pool.deployer(P[c][k][0], c, pool.bud[c][bk], Bh)(tg)
                        s = np.where(ab, -np.inf, pool.p[pool.ar, st])
                        acc[(c, bk)][b, i] += C.conf_select(s, pool.L[pool.ar, st], A, te, tg[1]).mean() / args.Rb
            if b % 50 == 49:
                log(f'bootstrap {b + 1}')
        for (c, bk), v in acc.items():
            out[f'boot|{bk}'] = v.astype(np.float32)
        out['boot|names'] = np.array(names)
        save()
    # ------------------------------------------------------------------ fine budget grid ----------------------------
    if 'reach' in parts:
        tg = ('cs', 0.15)
        fine = [float(d) for d in np.round(np.arange(0.5, 3.01, 0.25), 2)]
        out['reach|depths'] = np.array(fine)
        for c in ('t', 'r'):
            pols = {k: v for k, v in policies(D, pool, c, own=False).items() if not k.startswith('tri') or k in ('tri1', 'tri2', 'tri3')}
            pols['oracle'] = (pool.prep(hindsight_curve(0.85, c), c), None)
            for d in fine:
                Bq = float(np.interp(d, np.arange(T1), D['ctok_tr_mean'])) if c == 't' else d
                bk = f"{'T' if c == 't' else 'B'}{d}"
                for r in range(min(pool.R, args.Rw)):
                    A, Bh, te = pool.split(r)
                    for k, (pr, _) in pols.items():
                        st, ab, spent, _ = pool.deployer(pr, c, Bq, Bh)(tg)
                        add(f'reach|{k}|{tkey(tg)}|{bk}', pool.row(st, ab, spent, pool.p, tg, A, te)[0])
        log('fine budget grid'); save()
    # ------------------------------------------------------------------ size of the calibration halves --------------
    if 'calsize' in parts:
        tg = ('cs', 0.15)
        pols = {k: v for k, v in policies(D, pool, 't', own=False, tri=False).items() if k in ('ours', 'ours1q', 'fixed', 'stoprag')}
        Bq = pool.bud['t']['T1.0']
        for n in (100, 200, 400, 800):          # questions kept in A and in B (N / 4 in full: 1125 or more)
            assert n < pool.N // 4, (n, pool.N)
            for r in range(pool.R):
                A, Bh, te = pool.split(r)
                for k, (pr, _) in pols.items():
                    st, ab, spent, _ = pool.deployer(pr, 't', Bq, Bh[:n])(tg)
                    add(f'cal{n}|{k}|{tkey(tg)}|T1.0', pool.row(st, ab, spent, pool.p, tg, A[:n], te)[0])
        log('calibration size'); save()
    # ------------------------------------------------------------------ certificates of single rounds ---------------
    if 'peek' in parts:
        # one certificate per round, each valid for the candidates of its round, used inside the loop: a question is
        # answered at the first round whose certificate admits it.  Realised quantity of the answers so released.
        p, L = pool.p, pool.L
        for r in range(pool.R):
            A, Bh, te = pool.split(r)
            for a in (0.1, 0.15, 0.2):
                th = np.array([C.crc_strike(p[:, t], L[:, t], A, SGRID, a) for t in range(T1)])
                hit = p[te] >= th[None]
                st = C.first_true(hit, T); ans = hit.any(1)
                add(f'peek|risk|{a}', [float(np.mean(ans * L[te, st])), max(float(np.mean(hit[:, t] * L[te, t])) for t in range(T1)), float(ans.mean())])
                hit = np.stack([C.conf_select(p[:, t], L[:, t], A, te, a) for t in range(T1)], 1)
                st = C.first_true(hit, T); ans = hit.any(1)
                fdp = lambda m, w: float((m * w).sum() / max(m.sum(), 1))
                add(f'peek|cs|{a}', [fdp(ans, L[te, st]), max(fdp(hit[:, t], L[te, t]) for t in range(T1)), float(ans.mean())])
        log('single-round certificates in the loop: risk %.3f (target .15), batch error %.3f (target .15)' %
            (np.mean(np.array(rec['peek|risk|0.15'])[:, 0]), np.mean(np.array(rec['peek|cs|0.15'])[:, 0])))
        save()
    if 'vcert' in parts:
        # valid per-round certificates: the level is split across the rounds (uniformly, or in proportion to the questions
        # first answerable at a round, counted on the budget half, whose labels no certificate uses)
        from main21 import seq_certificate
        for r in range(pool.R):
            A, Bh, te = pool.split(r)
            yb = pool.y[Bh]; first = np.where(yb.max(1) > 0, yb.argmax(1), T1)
            ans_at = np.bincount(first, minlength=T1 + 1)[:T1].astype(float) + 1.0
            for an, wv in (('bonf', np.full(T1, 1.0 / T1)), ('alloc', ans_at / ans_at.sum())):
                for Bq in (1.0, 1.5, 2.0):
                    for tg in (('cs', 0.15), ('risk', 0.15)):
                        add(f'vcert|{an}|{tkey(tg)}|B{Bq}', seq_certificate(pool.p, pool.L, pool.cost['r'], pool.Ud, np.concatenate([A, Bh]), te, Bq, tg, T, wv))
        log('per-round certificates with a split level: ' + ' '.join(f"{an} {100 * np.mean(np.array(rec[f'vcert|{an}|cs|0.15|B1.0'])[:, 1]):.1f}" for an in ('bonf', 'alloc')))
        save()
    # ------------------------------------------------------------------ accuracy with no certificate ----------------
    if 'plain' in parts:
        ym = {'acc': D['y'], 'em': D['yem']}
        if 'yf1' in D:
            ym['f1'] = D['yf1']
        for c in ('t', 'r'):
            # no certificate, hence no target to calibrate a strike for: SCOPE prices the expected belief (strike 0)
            pols = {k: v for k, v in policies(D, pool, c, own=False).items() if not isinstance(v[0], (Fam, At1q))}
            pols['ours0'] = (pool.prep(D[f'S|ours|{c}'][OS[0.0]], c), None)
            for bk, Bq in pool.bud[c].items():
                if bk[1:] not in MAINB:
                    continue
                for r in range(min(pool.R, RPLAIN)):
                    A, Bh, te = pool.split(r)
                    for k, (pr, _) in pols.items():
                        st, ab, spent = pool.deploy(pr, c, Bq, Bh)
                        add(f'plain|{k}|{bk}', [ym['acc'][pool.ar, st][te].mean(), ym['em'][pool.ar, st][te].mean(),
                                                 ym['f1'][pool.ar, st][te].mean() if 'f1' in ym else np.nan, st[te].mean(), pool.tok[pool.ar, st][te].mean(), spent[te].mean()])
        log('plain accuracy'); save()
    for pt in sorted(parts):                      # reached only when every requested part has run to its end
        if pt in ('cap', 'perq') and 'main' not in parts:
            continue
        out[f'stamp|{pt}'] = np.array(run_stamp)
    save()
    log('done ' + run_stamp)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--cache'); ap.add_argument('--out'); ap.add_argument('--ds'); ap.add_argument('--rtag', default='')
    ap.add_argument('--fresh_tag', default=''); ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--R', type=int, default=200); ap.add_argument('--Rw', type=int, default=100)
    ap.add_argument('--nboot', type=int, default=200); ap.add_argument('--Rb', type=int, default=8)
    ap.add_argument('--parts', default='main,cap,perq,abl,oracle,wl,cm,boot,reach,calsize,peek,vcert,plain')
    ap.add_argument('--merge', type=int, default=0)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    run(args)

"""Neural Conditional SPRT (NC-SPRT) and certificate-aligned training (CCM).

I_t = gamma_t * I_{t-1} + Delta_t,
  Delta_t = f_Delta(x_t, g_{t-1})        conditional evidence increment (history-conditioned LLR)
  gamma_t = sigmoid(f_gamma(d_t, g_{t-1})) hypothesis-persistence gate (answer revisions reset evidence)
  g_t     = GRU(x_t, g_{t-1})            filtration summary
'innov' (main model): I_t = I_{t-1} + kappa(nu_t) (gamma_t I_{t-1} + Delta_t - I_{t-1}), kappa(0) = 0, so evidence
without innovation cannot move the index (redundancy invariance).
Variants used in ablations: 'indep' (Delta_t = f(x_t), gamma = 1: Wald-style sum of independent LLRs),
'nogate' (gamma = 1), 'direct' (I_t = head(g_t): recurrent classifier), 'step' (I_t = f(x_t): per-step probe).
"""
import math, numpy as np, torch, torch.nn as nn, torch.nn.functional as F
import os as _os
DEV = _os.environ.get('STOPRAG_DEV', 'cuda')   # 'cpu' when the shared GPU has no memory left


def mlp(i, h, o, p=0.1):
    return nn.Sequential(nn.Linear(i, h), nn.GELU(), nn.Dropout(p), nn.Linear(h, o))


class NCSPRT(nn.Module):
    def __init__(self, dh, du, dd, H=128, mode='full', p=0.2):
        super().__init__()
        self.mode, self.H = mode, H
        self.hproj = nn.Sequential(nn.Dropout(p), nn.Linear(dh, H), nn.LayerNorm(H)) if dh > 0 else None
        self.uproj = nn.Sequential(nn.Linear(du, H), nn.GELU(), nn.Linear(H, H), nn.LayerNorm(H))
        dx = 2 * H if dh > 0 else H
        self.gru = nn.GRUCell(dx, H)
        self.delta = mlp(dx + (0 if mode in ('indep', 'step') else H), H, 1)
        self.gate = mlp(dd + H, H // 2, 1)
        self.head = mlp(H, H, 1)
        self.fut = mlp(H + 3, H, 1)
        self.b0 = nn.Parameter(torch.zeros(1))
        self.a_gain = nn.Parameter(torch.tensor(5.0))  # innovation gain kappa(nu) = 1 - exp(-softplus(a) nu)

    def encode(self, h, u):
        xs = [self.uproj(u)]
        if self.hproj is not None:
            xs = [self.hproj(h)] + xs
        return torch.cat(xs, -1)

    def forward(self, h, u, d):
        x = self.encode(h, u)
        B, T1, _ = x.shape
        g = x.new_zeros(B, self.H)
        I_prev = self.b0.expand(B)
        Is, gs = [], []
        for t in range(T1):
            xt = x[:, t]
            if self.mode in ('indep', 'step'):
                delta = self.delta(xt).squeeze(-1)
            else:
                delta = self.delta(torch.cat([xt, g], -1)).squeeze(-1)
            g_new = self.gru(xt, g)
            if self.mode == 'direct':
                I = self.head(g_new).squeeze(-1)
            elif self.mode == 'step':
                I = delta + self.b0
            elif self.mode in ('indep', 'nogate'):
                I = I_prev + delta
            elif self.mode == 'innov':
                # innovation-gain update: the index moves towards the candidate posterior only in proportion
                # to the novelty of the new evidence; nu_t = 0 (fully redundant evidence) leaves I unchanged
                if t == 0:
                    I = delta + self.b0
                else:
                    gam = torch.sigmoid(self.gate(torch.cat([d[:, t], g], -1)).squeeze(-1) + 2.0)
                    cand = gam * I_prev + delta
                    kg = 1 - torch.exp(-F.softplus(self.a_gain) * d[:, t, 1].clamp(min=0))
                    I = I_prev + kg * (cand - I_prev)
            else:
                gam = torch.sigmoid(self.gate(torch.cat([d[:, t], g], -1)).squeeze(-1) + 2.0) if t > 0 else torch.ones_like(delta)
                I = gam * I_prev + delta
            Is.append(I); gs.append(g_new)
            g, I_prev = g_new, I
        return torch.stack(Is, 1), torch.stack(gs, 1)

    def futility(self, g, I, mu):
        """P(process crosses mu later | F_t, no crossing so far); inputs detached."""
        runmax = torch.cummax(I, 1).values
        z = torch.stack([I - mu, runmax - mu, torch.full_like(I, 1.0) * mu], -1)
        return self.fut(torch.cat([g, z], -1)).squeeze(-1)


# ------------------------------- certificate-aligned objective ---------------------------------

def soft_envelope(I, L, mu, kap=0.5, kap_r=0.5):
    """Smooth record-envelope loss and smooth coverage / expected cost of first-crossing at mu."""
    B, T1 = I.shape
    diff = I.unsqueeze(2) - I.unsqueeze(1)  # [B, t, s] = I_t - I_s
    mask = torch.tril(torch.ones(T1, T1, device=I.device), -1).bool()  # s < t
    logsig = F.logsigmoid(diff / kap_r).masked_fill(~mask, 0.0)
    rho = torch.exp(logsig.sum(2))  # soft "I_t is a running-max record"
    cross = torch.sigmoid((I - mu.unsqueeze(-1)) / kap)
    env = 1 - torch.prod(1 - L * rho * cross, 1)
    cov = 1 - torch.prod(1 - cross, 1)
    surv = torch.cumprod(torch.cat([torch.ones_like(cross[:, :1]), 1 - cross[:, :-1]], 1), 1)
    pi = surv * cross
    steps = torch.arange(T1, device=I.device, dtype=I.dtype)
    cost = (pi * steps).sum(1) + (1 - pi.sum(1)) * (T1 - 1)
    return env, cov, cost


def ccm_loss(I, L, alpha, beta=0.1, kap=0.5, iters=40):
    """Certified Coverage Maximisation: maximise coverage (minus beta*cost) at the soft-CRC threshold.

    mu_hat solves G(mu) = mean(env(mu)) - alpha = 0; its dependence on the network parameters is
    obtained with the implicit-function theorem through a single Newton re-parameterisation
    mu(theta) = mu0 - G(mu0, theta) / dG/dmu(mu0)."""
    B, T1 = I.shape
    with torch.no_grad():
        lo, hi = I.min() - 6 * kap, I.max() + 6 * kap
        for _ in range(iters):
            mid = (lo + hi) / 2
            env, _, _ = soft_envelope(I, L, mid.expand(B), kap)
            if env.mean() > alpha:
                lo = mid
            else:
                hi = mid
        mu0 = (lo + hi) / 2
    mu_var = mu0.clone().requires_grad_(True)
    env0, _, _ = soft_envelope(I.detach(), L, mu_var.expand(B), kap)
    gmu = torch.autograd.grad(env0.mean(), mu_var)[0].detach()
    env, _, _ = soft_envelope(I, L, mu0.expand(B), kap)
    G = env.mean() - alpha
    mu = mu0 - G / (gmu - 1e-6)
    _, cov, cost = soft_envelope(I, L, mu.expand(B), kap)
    return -(cov.mean() - beta * cost.mean() / (T1 - 1)), mu0.item()


# ----------------------------------------- training --------------------------------------------

def to_t(*a, dev=DEV):
    return [torch.as_tensor(x, dtype=torch.float32, device=dev) for x in a]


def train_index(h, u, d, y, mode='full', ccm=1.0, bce=1.0, beta=0.1, epochs=80, lr=1e-3, wd=1e-2, seed=0,
                val_frac=0.2, H=128, bs=256, alphas=(0.05, 0.3), verbose=False, dev=DEV):
    """Train an index network on training trajectories; model selection on a held-out slice by the
    certified coverage of the hard record certificate (calibrate on one half, evaluate on the other)."""
    from certify import cert_record, stopped_outcome
    torch.manual_seed(seed); np.random.seed(seed)
    N = len(y)
    perm = np.random.permutation(N)
    nv = int(N * val_frac)
    va, tr = perm[:nv], perm[nv:]
    net = NCSPRT(h.shape[-1] if h is not None else 0, u.shape[-1], d.shape[-1], H=H, mode=mode).to(dev)
    opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=wd)
    hT = torch.as_tensor(h, dtype=torch.float32, device=dev) if h is not None else torch.zeros(N, y.shape[1], 0, device=dev)
    uT, dT, yT = to_t(u, d, y, dev=dev)
    LT = 1 - yT
    best, best_state, bad = -1e9, None, 0
    for ep in range(epochs):
        net.train()
        rp = np.random.permutation(tr)
        for k in range(0, len(rp), bs):
            b = torch.as_tensor(rp[k:k + bs], device=dev)
            I, _ = net(hT[b], uT[b], dT[b])
            loss = bce * F.binary_cross_entropy_with_logits(I, yT[b])
            if ccm > 0:
                a = float(np.random.uniform(*alphas))
                lc, _ = ccm_loss(I, LT[b], a, beta=beta)
                loss = loss + ccm * lc
            opt.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            opt.step()
        net.eval()
        with torch.no_grad():
            Iv = net(hT[va], uT[va], dT[va])[0].cpu().numpy()
        Lv = 1 - y[va]
        h1, h2 = np.arange(nv) < nv // 2, np.arange(nv) >= nv // 2
        score = 0.0
        for a in (0.1, 0.15, 0.2):
            for c_, e_ in ((h1, h2), (h2, h1)):
                mu = cert_record(Iv[c_], Lv[c_], a)
                _, ans, cost = stopped_outcome(Iv[e_], Lv[e_], mu)
                score += ans.mean() - beta * cost.mean() / (y.shape[1] - 1)
        if verbose:
            print(ep, round(score / 6, 4), flush=True)
        if score > best:
            best, bad = score, 0
            best_state = {k: v.detach().clone() for k, v in net.state_dict().items()}
        else:
            bad += 1
            if bad >= 15:
                break
    net.load_state_dict(best_state)
    net.eval()
    return net


@torch.no_grad()
def apply_index(net, h, u, d, dev=DEV):
    N = len(u)
    hT = torch.as_tensor(h, dtype=torch.float32, device=dev) if h is not None else torch.zeros(N, u.shape[1], 0, device=dev)
    uT, dT = to_t(u, d, dev=dev)
    I, g = net(hT, uT, dT)
    return I.cpu().numpy(), g


def train_futility(net, h, u, d, epochs=30, lr=1e-3, dev=DEV, seed=0):
    """Fit the futility head: predicts whether the (frozen) index will cross mu at a later step."""
    torch.manual_seed(seed)
    I, g = apply_index(net, h, u, d, dev)
    I_t = torch.as_tensor(I, device=dev); g = g.detach()
    params = list(net.fut.parameters())
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=1e-2)
    qs = torch.as_tensor(np.quantile(I.max(1), np.linspace(0.05, 0.95, 64)), dtype=torch.float32, device=dev)
    T1 = I.shape[1]
    future_max = torch.flip(torch.cummax(torch.flip(I_t, [1]), 1).values, [1])  # max_{s>=t}
    nxt = torch.cat([future_max[:, 1:], torch.full_like(future_max[:, :1], -1e9)], 1)  # max_{s>t}
    runmax = torch.cummax(I_t, 1).values
    for ep in range(epochs):
        for _ in range(20):
            mu = qs[torch.randint(len(qs), (I_t.shape[0],), device=dev)].unsqueeze(1).expand(-1, T1)
            target = (nxt >= mu).float()
            active = (runmax < mu).float()  # not yet crossed at t
            logit = net.futility(g, I_t, mu)
            loss = (F.binary_cross_entropy_with_logits(logit, target, reduction='none') * active).sum() / active.sum().clamp(min=1)
            opt.zero_grad(); loss.backward(); opt.step()
    return net


@torch.no_grad()
def futility_prob(net, h, u, d, mu, dev=DEV):
    I, g = apply_index(net, h, u, d, dev)
    I_t = torch.as_tensor(I, device=dev)
    m = torch.full_like(I_t, float(mu))
    return torch.sigmoid(net.futility(g, I_t, m)).cpu().numpy()

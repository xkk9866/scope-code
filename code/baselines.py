"""Learned baselines on the same observations: Stop-RAG (Q(lambda) value-based stopping)."""
import math, numpy as np, torch, torch.nn as nn, torch.nn.functional as F
from model import mlp, to_t
import os as _os
DEV = _os.environ.get('STOPRAG_DEV', 'cuda')   # 'cpu' when the shared GPU has no memory left


class QNet(nn.Module):
    def __init__(self, dh, du, H=128, p=0.2):
        super().__init__()
        self.hproj = nn.Sequential(nn.Dropout(p), nn.Linear(dh, H), nn.LayerNorm(H)) if dh > 0 else None
        self.uproj = nn.Sequential(nn.Linear(du, H), nn.GELU(), nn.Linear(H, H), nn.LayerNorm(H))
        self.gru = nn.GRU(2 * H if dh > 0 else H, H, batch_first=True)
        self.qs, self.qc = mlp(H, H, 1), mlp(H, H, 1)

    def forward(self, h, u):
        x = self.uproj(u)
        if self.hproj is not None:
            x = torch.cat([self.hproj(h), x], -1)
        g, _ = self.gru(x)
        return self.qs(g).squeeze(-1), self.qc(g).squeeze(-1)


def train_stoprag(h, u, y, epochs=60, lr=1e-3, seed=0, bs=256, dev=DEV):
    """Full-width forward-view Q(lambda) targets with lambda annealed 1.0 -> 0.1 (Stop-RAG).
    Reward: 1 for stopping with a correct answer, 0 otherwise; continuing past the budget yields 0."""
    torch.manual_seed(seed); np.random.seed(seed)
    N, T1 = y.shape
    net = QNet(h.shape[-1] if h is not None else 0, u.shape[-1]).to(dev)
    opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=1e-2)
    hT = torch.as_tensor(h, dtype=torch.float32, device=dev) if h is not None else torch.zeros(N, T1, 0, device=dev)
    uT, yT = to_t(u, y, dev=dev)
    for ep in range(epochs):
        lam = 0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * ep / max(epochs - 1, 1)))
        net.eval()
        with torch.no_grad():
            qs, qc = net(hT, uT)
            # full-width backup (as in the official code): the STOP reward of a visited state is observed exactly
            V = torch.maximum(yT, torch.maximum(torch.sigmoid(qs), torch.sigmoid(qc)))
            V[:, -1] = yT[:, -1]  # at the budget only STOP is meaningful
        # forward-view lambda-return for CONT at t: mix of n-step bootstraps V(s_{t+n})
        tgt_c = torch.zeros_like(yT)
        for t in range(T1 - 1):
            ret, w = 0.0, 1.0
            for n in range(1, T1 - t):
                if t + n < T1 - 1:
                    ret = ret + w * (1 - lam) * V[:, t + n]
                    w = w * lam
                else:
                    ret = ret + w * V[:, t + n]
            tgt_c[:, t] = ret
        net.train()
        perm = np.random.permutation(N)
        for k in range(0, N, bs):
            b = torch.as_tensor(perm[k:k + bs], device=dev)
            qs, qc = net(hT[b], uT[b])
            loss = F.binary_cross_entropy_with_logits(qs, yT[b]) + F.binary_cross_entropy_with_logits(qc, tgt_c[b])
            opt.zero_grad(); loss.backward(); opt.step()
    net.eval()
    return net


@torch.no_grad()
def stoprag_index(net, h, u, dev=DEV):
    N, T1 = u.shape[:2]
    hT = torch.as_tensor(h, dtype=torch.float32, device=dev) if h is not None else torch.zeros(N, T1, 0, device=dev)
    uT, = to_t(u, dev=dev)
    qs, qc = net(hT, uT)
    I = torch.sigmoid(qs) - torch.sigmoid(qc)
    I[:, -1] = torch.sigmoid(qs[:, -1])  # last step: stop is the only option; answer value = Q_stop
    return I.cpu().numpy(), torch.sigmoid(qs).cpu().numpy()


# Published adaptive-retrieval stopping rules, 2023-2025. Each returns a score that rises with confidence;
# the certified policy stops at the first round whose score clears a threshold (cert14.first_cross_path),
# so every rule is compared at the same mean budget. None of these rules is a model we designed.

def _col(z, name):
    names = list(z['u_names'])
    return np.asarray(z['u'][:, :, names.index(name)], np.float32)


def _curve(score, tr, ev, T):
    import cert14 as C
    return C.first_cross_path(score[ev], C.thr_desc(score[tr][:, :T]), T)


def confidence_curves(z, tr, ev, T):
    """Rules whose score is logged by the reader and needs no training.
    FLARE stops when the least probable generated token is confident (Jiang et al., EMNLP 2023).
    DRAGIN stops when the generated tokens are no longer high-entropy (Su et al., ACL 2024).
    SeaKR stops when sampled reasoning paths agree (Yao et al., ACL 2025).
    IRCoT stops when the reader judges the evidence sufficient (Trivedi et al., ACL 2023)."""
    return {
        'flare': _curve(np.exp(_col(z, 'lp_min')), tr, ev, T),
        'dragin': _curve(-_col(z, 'ent_mean'), tr, ev, T),
        'seakr': _curve(_col(z, 'cons_soft'), tr, ev, T),
        'ircot': _curve(_col(z, 'p_suff'), tr, ev, T),
    }


class _Probe(nn.Module):
    """Probing-RAG's prober: one linear layer on the reader state, nothing else (Baek et al., NAACL 2025)."""

    def __init__(self, d):
        super().__init__()
        self.w = nn.Linear(d, 1)

    def forward(self, h):
        return self.w(h).squeeze(-1)


def _fit_probe(h, y, seed, epochs, dev):
    torch.manual_seed(seed)
    net = _Probe(h.shape[-1]).to(dev)
    opt = torch.optim.AdamW(net.parameters(), lr=1e-2, weight_decay=1e-3)
    H = torch.as_tensor(h, dtype=torch.float32, device=dev)
    Y = torch.as_tensor(y, dtype=torch.float32, device=dev)
    for _ in range(epochs):
        loss = F.binary_cross_entropy_with_logits(net(H), Y)
        opt.zero_grad(); loss.backward(); opt.step()
    net.eval()
    return net


@torch.no_grad()
def _probe_pred(net, h, dev):
    return torch.sigmoid(net(torch.as_tensor(h, dtype=torch.float32, device=dev))).cpu().numpy()


def probing_curve(h_tr, y_tr, h_ev, fold, T, seed=0, epochs=30, dev=DEV):
    """Cross-fitted prober: stop when the reader state predicts that the current answer is correct."""
    oof = np.zeros(y_tr.shape, np.float32)
    nets = []
    for k in range(int(fold.max()) + 1):
        net = _fit_probe(h_tr[fold != k], y_tr[fold != k], 1000 * seed + k, epochs, dev)
        nets.append(net)
        oof[fold == k] = _probe_pred(net, h_tr[fold == k], dev)
    ev = np.mean([_probe_pred(n, h_ev, dev) for n in nets], 0)
    return _curve_parts(ev, oof, T)


def _curve_parts(ev, tr, T):
    import cert14 as C
    return C.first_cross_path(ev, C.thr_desc(tr[:, :T]), T)


def adaptive_rag_curve(x0_tr, need, x0_ev, T, seed, M, ncls):
    """Adaptive-RAG: a classifier predicts how many reads the question needs before any retrieval
    (Jeong et al., NAACL 2024). Supervision is the first round with a correct answer."""
    from main15 import train_router
    import cert14 as C
    nets = [train_router(x0_tr, need, ncls, seed=1000 * seed + m) for m in range(M)]
    cdf = np.cumsum(np.mean([n(x0_ev) for n in nets], 0), 1)
    cdf_tr = np.cumsum(np.mean([n(x0_tr) for n in nets], 0), 1)
    return C.first_cross_path(cdf, C.thr_desc(cdf_tr[:, :T]), T)


def stoprag_curve(h_tr, u_tr, y_tr, h_ev, u_ev, T, seed, M, dev=DEV):
    """Stop-RAG: Q(lambda) stop-versus-continue values (Park et al., 2025)."""
    import cert14 as C
    nets = [train_stoprag(h_tr, u_tr, y_tr, seed=1000 * seed + m, dev=dev) for m in range(M)]
    ev = np.mean([stoprag_index(n, h_ev, u_ev, dev=dev)[0] for n in nets], 0)
    tr = np.mean([stoprag_index(n, h_tr, u_tr, dev=dev)[0] for n in nets], 0)
    return C.first_cross_path(ev, C.thr_desc(tr[:, :T]), T)

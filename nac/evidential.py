"""Evidential heads and losses.

One linear head (raw logits) + pluggable evidence functions and losses covering both
formulations compared in the notebook:
  - Sensoy et al. (NeurIPS 2018): capped-softplus evidence, MSE-form loss,
    KL-to-uniform regularizer with a linear ramp to `lambda_ceiling` (Section 6/7).
  - DEAR (Bao et al., ICCV 2021): exp evidence, log-form loss, optional KL and/or
    EUC/AvU calibration with exponential annealing 0.01 -> 1.0 (Section 8; ported
    from Cogito2012/DEAR mmaction/models/losses/edl_loss.py).
Every component is a switch so ablations are one-flag changes.
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class LinearHead(nn.Module):
    def __init__(self, in_dim, num_classes):
        super().__init__()
        self.fc = nn.Linear(in_dim, num_classes)

    def forward(self, x):
        return self.fc(x)


def evidence_fn(logits, kind='exp', cap=10.0):
    if kind == 'exp':
        return torch.exp(torch.clamp(logits, -10, 10))
    if kind == 'softplus_capped':
        return torch.clamp(F.softplus(logits), max=cap)
    raise ValueError(kind)


def kl_to_uniform(alpha, num_classes):
    beta = torch.ones([1, num_classes], device=alpha.device)
    S_alpha = alpha.sum(dim=1, keepdim=True)
    lnB = torch.lgamma(S_alpha) - torch.lgamma(alpha).sum(dim=1, keepdim=True)
    lnB_uni = torch.lgamma(beta).sum(dim=1, keepdim=True) - torch.lgamma(beta.sum(dim=1, keepdim=True))
    return ((alpha - beta) * (torch.digamma(alpha) - torch.digamma(S_alpha))).sum(dim=1, keepdim=True) + lnB + lnB_uni


def make_edl_loss(num_classes, loss_form='log', evidence='exp',
                  with_kldiv=False, with_avuloss=True,
                  annealing='exp', annealing_start=0.01, annealing_step=40,
                  lambda_ceiling=1.0, total_epoch=75, eps=1e-10):
    """Returns loss(logits, target, epoch). Defaults = DEAR best config.
    Sensoy tuned config (Section 7): loss_form='mse', evidence='softplus_capped',
    with_kldiv=True, with_avuloss=False, annealing='linear', lambda_ceiling=0.02."""

    def coef_at(epoch):
        if annealing == 'exp':
            return float(annealing_start * np.exp(-np.log(annealing_start) / total_epoch * epoch))
        if annealing == 'linear':
            return min(lambda_ceiling, lambda_ceiling * epoch / annealing_step)
        raise ValueError(annealing)

    def loss(logits, target, epoch):
        ev = evidence_fn(logits, evidence)
        alpha = ev + 1
        y = F.one_hot(target, num_classes).float()
        S = alpha.sum(dim=1, keepdim=True)
        coef = coef_at(epoch)

        if loss_form == 'log':
            out = (y * (torch.log(S) - torch.log(alpha))).sum(dim=1, keepdim=True)
        elif loss_form == 'mse':
            err = ((y - alpha / S) ** 2).sum(dim=1, keepdim=True)
            var = (alpha * (S - alpha) / (S * S * (S + 1))).sum(dim=1, keepdim=True)
            out = err + var
        else:
            raise ValueError(loss_form)

        if with_kldiv:
            kl_alpha = (alpha - 1) * (1 - y) + 1
            out = out + coef * kl_to_uniform(kl_alpha, num_classes)

        if with_avuloss:
            ps, pc = torch.max(alpha / S, 1, keepdim=True)
            u = num_classes / S
            match = torch.eq(pc, target.unsqueeze(1)).float()
            acc_uncertain = -ps * torch.log(1 - u + eps)
            inacc_certain = -(1 - ps) * torch.log(u + eps)
            out = out + coef * match * acc_uncertain + (1 - coef) * (1 - match) * inacc_certain

        return out.mean()

    return loss


def train_head(X, y, num_classes, loss, epochs=75, lr=1e-3, weight_decay=1e-4,
               batch_size=256, seed=0, device=None):
    device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
    torch.manual_seed(seed)
    head = LinearHead(X.shape[1], num_classes).to(device)
    opt = torch.optim.Adam(head.parameters(), lr=lr, weight_decay=weight_decay)
    X_t = torch.tensor(X, dtype=torch.float32)
    y_t = torch.tensor(y, dtype=torch.long)
    n = len(X_t)
    for epoch in range(epochs):
        perm = torch.randperm(n)
        for i in range(0, n, batch_size):
            idx = perm[i:i + batch_size]
            opt.zero_grad()
            loss(head(X_t[idx].to(device)), y_t[idx].to(device), epoch).backward()
            opt.step()
    head.eval()
    return head


@torch.no_grad()
def predict_alpha(head, X, evidence='exp', batch_size=8192, device=None):
    device = device or next(head.parameters()).device
    out = []
    for i in range(0, len(X), batch_size):
        logits = head(torch.tensor(X[i:i + batch_size], dtype=torch.float32).to(device))
        out.append(evidence_fn(logits, evidence) + 1)
    alpha = torch.cat(out)
    return alpha


def vacuity(alpha, num_classes):
    """Evidential vacuity uncertainty u = K/S — the novelty score."""
    return (num_classes / alpha.sum(dim=1)).cpu().numpy()

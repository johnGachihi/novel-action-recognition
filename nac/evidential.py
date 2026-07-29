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
import os
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
               batch_size=256, seed=0, device=None, checkpoint_path=None,
               X_val_loss=None, y_val_loss=None, X_val_auroc=None, is_novel_val=None,
               evidence='exp', resume=False):
    device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
    torch.manual_seed(seed)
    head = LinearHead(X.shape[1], num_classes).to(device)
    opt = torch.optim.Adam(head.parameters(), lr=lr, weight_decay=weight_decay)

    history = {'train_loss': [], 'val_loss': [], 'val_auroc': []}
    start_epoch = 0
    if resume and checkpoint_path is not None and os.path.exists(checkpoint_path):
        print(f"Loading checkpoint from {checkpoint_path}...")
        try:
            ckpt = torch.load(checkpoint_path, map_location=device)
            state_dict = ckpt['head_state_dict']
            state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
            head.load_state_dict(state_dict)
            opt.load_state_dict(ckpt['opt_state_dict'])
            start_epoch = ckpt['epoch'] + 1
            if 'history' in ckpt:
                history = ckpt['history']
            print(f"Resuming training from epoch {start_epoch}")
        except Exception as e:
            print(f"Failed to load checkpoint: {e}. Starting from scratch.")

    X_t = torch.tensor(X, dtype=torch.float32).to(device)
    y_t = torch.tensor(y, dtype=torch.long).to(device)
    n = len(X_t)

    # Pre-copy validation sets to GPU once to eliminate per-epoch overhead
    X_vl = None
    y_vl = None
    if X_val_loss is not None and y_val_loss is not None:
        X_vl = torch.tensor(X_val_loss, dtype=torch.float32).to(device)
        y_vl = torch.tensor(y_val_loss, dtype=torch.long).to(device)

    X_va = None
    if X_val_auroc is not None:
        X_va = torch.tensor(X_val_auroc, dtype=torch.float32).to(device)
    
    train_losses = history.setdefault('train_loss', [])
    val_losses = history.setdefault('val_loss', [])
    val_aurocs = history.setdefault('val_auroc', [])

    if start_epoch < epochs:
        for epoch in range(start_epoch, epochs):
            head.train()
            perm = torch.randperm(n, device=device)
            epoch_loss = 0.0
            num_batches = 0
            for i in range(0, n, batch_size):
                idx = perm[i:i + batch_size]
                opt.zero_grad()
                l = loss(head(X_t[idx]), y_t[idx], epoch)
                l.backward()
                opt.step()
                epoch_loss += l.item()
                num_batches += 1
            train_losses.append(epoch_loss / max(1, num_batches))
            
            # Calculate validation loss if provided
            if X_vl is not None and y_vl is not None:
                head.eval()
                with torch.no_grad():
                    val_l = loss(head(X_vl), y_vl, epoch).item()
                val_losses.append(val_l)
            
            # Calculate validation AUROC if provided
            if X_va is not None and is_novel_val is not None:
                from sklearn.metrics import roc_auc_score
                head.eval()
                with torch.no_grad():
                    alpha_val = predict_alpha(head, X_va, evidence=evidence, device=device)
                    val_vacuity = (num_classes / alpha_val.sum(dim=1)).cpu().numpy()
                    if len(is_novel_val) > 0 and len(np.unique(is_novel_val)) > 1 and np.isfinite(val_vacuity).all():
                        val_auroc = float(roc_auc_score(is_novel_val, val_vacuity))
                    else:
                        val_auroc = float('nan')
                val_aurocs.append(val_auroc)
            
            if checkpoint_path is not None:
                torch.save({
                    'head_state_dict': head.state_dict(),
                    'opt_state_dict': opt.state_dict(),
                    'epoch': epoch,
                    'history': history
                }, checkpoint_path)
    else:
        print("Model already fully trained. Loaded from checkpoint.")

    head.eval()
    return head, history


@torch.no_grad()
def predict_alpha(head, X, evidence='exp', batch_size=8192, device=None):
    device = device or next(head.parameters()).device
    if len(X) == 0:
        out_dim = None
        for p in head.parameters():
            if p.dim() == 2:
                out_dim = p.shape[0]
                break
        return torch.empty(0, out_dim or 1, device=device)

    X_t = X if isinstance(X, torch.Tensor) else torch.tensor(X, dtype=torch.float32)
    X_t = X_t.to(device)

    out = []
    for i in range(0, len(X_t), batch_size):
        logits = head(X_t[i:i + batch_size])
        if evidence == 'msp':
            probs = torch.softmax(logits, dim=1)
            anomaly_score = 1.0 - probs.max(dim=1)[0]
            val_col = 1.0 / torch.clamp(anomaly_score, min=1e-8)
            alpha_row = val_col.unsqueeze(1).repeat(1, logits.shape[1])
            out.append(alpha_row)
        else:
            out.append(evidence_fn(logits, evidence) + 1)
    alpha = torch.cat(out)
    return alpha


def vacuity(alpha, num_classes):
    """Evidential vacuity uncertainty u = K/S — the novelty score."""
    return (num_classes / alpha.sum(dim=1)).cpu().numpy()

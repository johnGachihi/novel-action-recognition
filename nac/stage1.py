"""Stage 1: novelty detection benchmark (notebook Sections 6/8/9).

Group-aware 80/20 split on the subset's known classes; novelty eval on held-out
known + the subset's novel samples. Methods are registry entries so ablations are
selectable by name; the evidential entries expose their loss components as kwargs.
"""
import numpy as np

from .baselines import cosine_prototype, mahalanobis
from .data import load_features, subset_mask, known_train_heldout
from .evidential import make_edl_loss, predict_alpha, train_head, vacuity
from .metrics import roc_auc_score

EVIDENTIAL_CONFIGS = {
    'sensoy_tuned': dict(loss_form='mse', evidence='softplus_capped', with_kldiv=True,
                         with_avuloss=False, annealing='linear', lambda_ceiling=0.02,
                         annealing_step=40),
    'dear':         dict(loss_form='log', evidence='exp', with_kldiv=False, with_avuloss=True),
    'dear_noreg':   dict(loss_form='log', evidence='exp', with_kldiv=False, with_avuloss=False),
    'dear_kl':      dict(loss_form='log', evidence='exp', with_kldiv=True, with_avuloss=False),
}

ALL_METHODS = ['cosine', 'mahalanobis'] + list(EVIDENTIAL_CONFIGS)


def run_stage1(subset=None, methods=ALL_METHODS, epochs=75, seed=0,
               features_path='features_videomae.npz', overrides=None, verbose=True):
    """overrides: dict of extra kwargs merged into every evidential config (ablations)."""
    data = load_features(features_path)
    feats, class_labels, paths = data['features'], data['class_labels'], data['paths']
    known = (data['status'] == 'known') & subset_mask(class_labels, subset)
    novel = (data['status'] == 'novel') & subset_mask(class_labels, subset)
    class_names = np.unique(class_labels[known])
    cls_to_idx = {c: i for i, c in enumerate(class_names)}
    K = len(class_names)

    rng = np.random.default_rng(seed)
    train_idx, heldout_idx = known_train_heldout(class_names, class_labels, paths, rng)
    eval_idx = np.concatenate([heldout_idx, np.where(novel)[0]])
    is_novel = np.concatenate([np.zeros(len(heldout_idx)), np.ones(novel.sum())])
    true_idx = np.array([cls_to_idx[c] for c in class_labels[heldout_idx]])
    nh = len(heldout_idx)
    if verbose:
        print(f"[{subset or 'combined'}] K={K} train={len(train_idx)} heldout={nh} novel={int(novel.sum())}")

    mu, sigma = feats[train_idx].mean(0), feats[train_idx].std(0) + 1e-6
    X_train = (feats[train_idx] - mu) / sigma
    y_train = np.array([cls_to_idx[c] for c in class_labels[train_idx]])
    X_eval = (feats[eval_idx] - mu) / sigma

    results = {}

    def record(name, novelty, pred):
        results[name] = {
            'auroc': float(roc_auc_score(is_novel, novelty)),
            'closed_set_acc': float((true_idx == pred[:nh]).mean()),
        }
        if verbose:
            r = results[name]
            print(f"  {name:16s}  AUROC={r['auroc']:.3f}   closed_set_acc={r['closed_set_acc']:.3f}")

    for name in methods:
        if name == 'cosine':
            record(name, *cosine_prototype(feats, train_idx, eval_idx, class_labels, class_names))
        elif name == 'mahalanobis':
            record(name, *mahalanobis(feats, train_idx, eval_idx, class_labels, class_names))
        elif name in EVIDENTIAL_CONFIGS:
            cfg = {**EVIDENTIAL_CONFIGS[name], **(overrides or {}), 'total_epoch': epochs}
            loss = make_edl_loss(K, **cfg)
            head = train_head(X_train, y_train, K, loss, epochs=epochs, seed=seed)
            alpha = predict_alpha(head, X_eval, evidence=cfg['evidence'])
            record(name, vacuity(alpha, K), alpha.argmax(1).cpu().numpy())
        else:
            raise ValueError(f"unknown method {name}")
    return results

"""Stage 1 novelty-detection benchmark, generalized to EPIC-KITCHENS-100's two
independent label spaces (verb, noun) in place of UCF101/HMDB51's single
action-class label. Same protocol as nac/stage1.py: group-aware 80/20 split on
known classes, novelty eval on held-out known + the label space's novel
samples. Reuses nac.baselines and nac.evidential unchanged (both are already
label-dtype-agnostic); only the data loading/splitting is EPIC-specific
(video_id grouping instead of UCF101/HMDB51 clip grouping).
"""
import json
import math
import os
import numpy as np

from .baselines import cosine_prototype, mahalanobis
from .epic_data import known_three_way, load_epic_manifest
from .evidential import make_edl_loss, predict_alpha, train_head, vacuity
from .metrics import roc_auc_score
from .stage1 import EVIDENTIAL_CONFIGS

ALL_METHODS = ['cosine', 'mahalanobis', 'standard_classifier'] + list(EVIDENTIAL_CONFIGS)


def run_epic_stage1(label_space, methods=ALL_METHODS, epochs=75, seed=0,
                    features_path='features_epic.npz', split_path='class_split_epic.json',
                    overrides=None, verbose=True):
    """label_space: 'verb' or 'noun'."""
    assert label_space in ('verb', 'noun')
    d = np.load(features_path, allow_pickle=True)
    feats = d['features']
    labels = d[f'{label_space}_class']
    keep = d[f'{label_space}_keep']
    video_ids = d['video_id']

    split = json.load(open(split_path))[label_space]
    known_classes = np.array(split['known'])
    novel_classes = np.array(split['novel'])
    cls_to_idx = {c: i for i, c in enumerate(known_classes)}
    K = len(known_classes)

    known = keep & np.isin(labels, known_classes)
    novel = keep & np.isin(labels, novel_classes)

    rng = np.random.default_rng(seed)
    
    # 3-way split for known classes: 80% train, 10% validation, 10% test
    train_idx, val_heldout_idx, test_heldout_idx = known_three_way(known_classes, labels, video_ids, rng)
    
    # Split novel classes/samples 50/50 into validation and test sets
    novel_idx = np.where(novel)[0]
    rng.shuffle(novel_idx)
    half = len(novel_idx) // 2
    val_novel_idx = novel_idx[:half]
    test_novel_idx = novel_idx[half:]

    # Construct Val & Test sets
    val_idx = np.concatenate([val_heldout_idx, val_novel_idx])
    is_novel_val = np.concatenate([np.zeros(len(val_heldout_idx)), np.ones(len(val_novel_idx))])
    y_val_heldout = np.array([cls_to_idx[c] for c in labels[val_heldout_idx]])

    test_idx = np.concatenate([test_heldout_idx, test_novel_idx])
    is_novel_test = np.concatenate([np.zeros(len(test_heldout_idx)), np.ones(len(test_novel_idx))])
    true_idx_test = np.array([cls_to_idx[c] for c in labels[test_heldout_idx]])
    
    nh_val = len(val_heldout_idx)
    nh_test = len(test_heldout_idx)
    novel_val_count = len(val_novel_idx)
    novel_test_count = len(test_novel_idx)

    if verbose:
        print(f"[EPIC:{label_space}] K={K} train={len(train_idx)} val_heldout={nh_val} val_novel={novel_val_count} test_heldout={nh_test} test_novel={novel_test_count}")

    mu, sigma = feats[train_idx].mean(0), feats[train_idx].std(0) + 1e-6
    X_train = (feats[train_idx] - mu) / sigma
    y_train = np.array([cls_to_idx[c] for c in labels[train_idx]])
    
    # Standardize val and test sets
    X_val = (feats[val_idx] - mu) / sigma
    X_val_heldout = (feats[val_heldout_idx] - mu) / sigma
    X_test = (feats[test_idx] - mu) / sigma

    results = {}
    novelty_scores = {}

    def _safe_auroc(gt, scores):
        if len(gt) and np.isfinite(scores).all() and gt.min() < gt.max():
            try:
                return float(roc_auc_score(gt, scores))
            except Exception:
                return float('nan')
        return float('nan')
    histories = {}
    for name in methods:
        if name in ('cosine', 'mahalanobis'):
            novelty, pred = cosine_prototype(
                feats, train_idx, test_idx, labels, known_classes
            ) if name == 'cosine' else mahalanobis(
                feats, train_idx, test_idx, labels, known_classes
            )
            history = None
        elif name == 'standard_classifier':
            import torch.nn as nn
            loss_fn = lambda logits, target, epoch: nn.CrossEntropyLoss()(logits, target)
            ckpt_path = f"checkpoint_stage1_{label_space}_{name}_seed{seed}.pt"
            
            # Train the head with validation tracking
            head, history = train_head(
                X_train, y_train, K, loss_fn, epochs=epochs, seed=seed, checkpoint_path=ckpt_path,
                X_val_loss=X_val_heldout, y_val_loss=y_val_heldout,
                X_val_auroc=X_val, is_novel_val=is_novel_val,
                evidence='msp'
            )
            histories[name] = history
            
            # Predict on test set
            head.eval()
            import torch
            with torch.no_grad():
                device = next(head.parameters()).device
                logits = head(torch.tensor(X_test, dtype=torch.float32).to(device))
                probs = torch.softmax(logits, dim=1)
                novelty = (1.0 - probs.max(1)[0]).cpu().numpy()
                pred = logits.argmax(1).cpu().numpy()
        elif name in EVIDENTIAL_CONFIGS:
            cfg = {**EVIDENTIAL_CONFIGS[name], **(overrides or {}), 'total_epoch': epochs}
            loss = make_edl_loss(K, **cfg)
            ckpt_path = f"checkpoint_stage1_{label_space}_{name}_seed{seed}.pt"
            
            # Train the head with validation tracking
            head, history = train_head(
                X_train, y_train, K, loss, epochs=epochs, seed=seed, checkpoint_path=ckpt_path,
                X_val_loss=X_val_heldout, y_val_loss=y_val_heldout,
                X_val_auroc=X_val, is_novel_val=is_novel_val,
                evidence=cfg['evidence']
            )
            histories[name] = history
            
            # Predict on test set
            alpha = predict_alpha(head, X_test, evidence=cfg['evidence'])
            novelty = vacuity(alpha, K)
            pred = alpha.argmax(1).cpu().numpy()
        else:
            raise ValueError(f"unknown method {name}")

        novelty_scores[name] = novelty
        metrics = {
            'auroc': _safe_auroc(is_novel_test, novelty),
            'closed_set_acc': float((true_idx_test == pred[:nh_test]).mean()) if nh_test else float('nan'),
        }
        results[name] = metrics

        if verbose:
            auroc = f"{metrics['auroc']:.3f}" if math.isfinite(metrics['auroc']) else "N/A"
            cacc = f"{metrics['closed_set_acc']:.3f}" if math.isfinite(metrics['closed_set_acc']) else "N/A"
            print(f"  {name:16s}  AUROC={auroc:>7}   closed_set_acc={cacc}")

    return {
        'metrics': results,
        'test_idx': test_idx.tolist(),
        'is_novel_test': is_novel_test.tolist(),
        'test_heldout_idx': test_heldout_idx.tolist(),
        'true_idx_test': true_idx_test.tolist(),
        'novelty_scores': {name: scores.tolist() for name, scores in novelty_scores.items()},
        'histories': histories
    }

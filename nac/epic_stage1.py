"""Stage 1 novelty-detection benchmark, generalized to EPIC-KITCHENS-100's two
independent label spaces (verb, noun) in place of UCF101/HMDB51's single
action-class label. Same protocol as nac/stage1.py: group-aware 80/20 split on
known classes, novelty eval on held-out known + the label space's novel
samples. Reuses nac.baselines and nac.evidential unchanged (both are already
label-dtype-agnostic); only the data loading/splitting is EPIC-specific
(video_id grouping instead of UCF101/HMDB51 clip grouping).
"""
import json

import numpy as np

from .baselines import cosine_prototype, mahalanobis
from .epic_data import known_train_heldout, load_epic_manifest
from .evidential import make_edl_loss, predict_alpha, train_head, vacuity
from .metrics import roc_auc_score
from .stage1 import EVIDENTIAL_CONFIGS

ALL_METHODS = ['cosine', 'mahalanobis'] + list(EVIDENTIAL_CONFIGS)


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
    known_idx_pool = np.where(known)[0]
    train_idx, heldout_idx = known_train_heldout(known_classes, labels, video_ids, rng)
    eval_idx = np.concatenate([heldout_idx, np.where(novel)[0]])
    is_novel = np.concatenate([np.zeros(len(heldout_idx)), np.ones(novel.sum())])
    true_idx = np.array([cls_to_idx[c] for c in labels[heldout_idx]])
    nh = len(heldout_idx)
    if verbose:
        print(f"[EPIC:{label_space}] K={K} train={len(train_idx)} heldout={nh} novel={int(novel.sum())}")

    mu, sigma = feats[train_idx].mean(0), feats[train_idx].std(0) + 1e-6
    X_train = (feats[train_idx] - mu) / sigma
    y_train = np.array([cls_to_idx[c] for c in labels[train_idx]])
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
            record(name, *cosine_prototype(feats, train_idx, eval_idx, labels, known_classes))
        elif name == 'mahalanobis':
            record(name, *mahalanobis(feats, train_idx, eval_idx, labels, known_classes))
        elif name in EVIDENTIAL_CONFIGS:
            cfg = {**EVIDENTIAL_CONFIGS[name], **(overrides or {}), 'total_epoch': epochs}
            loss = make_edl_loss(K, **cfg)
            ckpt_path = f"checkpoint_stage1_{label_space}_{name}_seed{seed}.pt"
            head = train_head(X_train, y_train, K, loss, epochs=epochs, seed=seed, checkpoint_path=ckpt_path)
            alpha = predict_alpha(head, X_eval, evidence=cfg['evidence'])
            record(name, vacuity(alpha, K), alpha.argmax(1).cpu().numpy())
        else:
            raise ValueError(f"unknown method {name}")
    return results

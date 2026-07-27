#!/usr/bin/env python3
"""Happy-adapted continual GCD on frozen VideoMAE features, with a trainable
projector fed by two augmented views (see nac/happy.py module docstring for the
full architecture rationale and what's kept/adapted from the paper).

Runs Happy's own ablation ladder (self-train only -> +entropy-reg -> +cluster-init
-> +hardness-replay), plus one more rung: +projector, which adds the two-view
contrastive losses and feature-level KD that only become meaningful once something
is actually trainable between the frozen backbone and the classifier. The first
four rungs reproduce the fully-frozen, no-projector architecture (previously
diagnosed as bistable/collapsing once cluster-init is added); the fifth tests
whether restoring that representation-learning capacity fixes it.

Evaluation per stage: 'Old' accuracy is direct index match (old-class head indices
are fixed once assigned); 'New' accuracy is Hungarian-matched among this stage's new
heads only; 'All' is the sample-weighted combination. M_f = max forgetting on
stage-0 classes across all later stages; M_d = final 'All' accuracy.

Usage:
  python3 run_happy.py                          # full ablation ladder, combined
  python3 run_happy.py --subset UCF101 --stages 5
  python3 run_happy.py --config full+projector  # only the complete method
"""
import argparse
import copy
import json

import numpy as np
import torch
import torch.nn.functional as F

from nac.data import load_features, subset_mask
from nac.geometry import fit_whitener
from nac.happy import (GrowingClassifier, Projector, build_multistage_splits,
                       cluster_guided_init, train_stage, train_stage0, update_prototypes)
from nac.metrics import hungarian_acc

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

CONFIGS = {
    'self_train_only':        dict(entropy=False, cluster_init=False, hardness=False, projector=False),
    '+entropy_reg':            dict(entropy=True,  cluster_init=False, hardness=False, projector=False),
    '+cluster_init':           dict(entropy=True,  cluster_init=True,  hardness=False, projector=False),
    'full (+hardness_replay)': dict(entropy=True,  cluster_init=True,  hardness=True,  projector=False),
    'full+projector':          dict(entropy=True,  cluster_init=True,  hardness=True,  projector=True),
}


def load_two_views(features_path, view2_path):
    data = load_features(features_path)
    feats_a = data['features']
    d2 = np.load(view2_path, allow_pickle=True)
    order = np.argsort(d2['idx'])
    feats_b = d2['features'][order]
    assert len(feats_b) == len(feats_a), \
        f"view2 has {len(feats_b)} videos, view A has {len(feats_a)} -- extraction incomplete?"
    return data, feats_a, feats_b


def run_config(subset, cfg, T, seed, feats_a, feats_b, cls, status,
              epochs, old_per_class, stage0_epochs, proj_hidden=512, proj_layers=1, proj_residual=True):
    m = subset_mask(cls, subset)
    known_classes = np.unique(cls[(status == 'known') & m])
    novel_classes = np.unique(cls[(status == 'novel') & m])
    rng = np.random.default_rng(seed)

    stage0_idx, stage0_labels, stages = build_multistage_splits(
        known_classes, novel_classes, cls, rng, T=T, old_per_class=old_per_class)
    n_old0 = len(known_classes)

    tr_labels = np.array([stage0_labels[c] for c in cls[stage0_idx]])
    W, _ = fit_whitener(feats_a[stage0_idx], tr_labels, n_old0)
    fA, fB = feats_a @ W, feats_b @ W  # same whitener applied to both views

    torch.manual_seed(seed)
    clf = GrowingClassifier(fA.shape[1], n_old0).to(DEVICE)
    projector = Projector(fA.shape[1], hidden=proj_hidden, n_layers=proj_layers,
                          residual=proj_residual).to(DEVICE) if cfg['projector'] else None
    Z0_a = torch.tensor(fA[stage0_idx], dtype=torch.float32, device=DEVICE)
    Z0_b = torch.tensor(fB[stage0_idx], dtype=torch.float32, device=DEVICE)
    y0 = torch.tensor(tr_labels, dtype=torch.long, device=DEVICE)
    with torch.no_grad():
        # Warm-start heads from class means IN THE SPACE THE CLASSIFIER WILL ACTUALLY
        # SEE: projected, if a projector exists. Using raw-feature means here while
        # immediately evaluating through a (randomly-initialized) projector creates an
        # inconsistent starting point that joint SGD then has to violently reconcile
        # -- confirmed: this alone (with the temperature bug also present) was enough
        # to make stage-0 training destroy a 0.99-accuracy init down to ~0.02.
        basis = projector(Z0_a) if projector is not None else Z0_a
        for i in range(n_old0):
            clf.weight[i] = F.normalize(basis[y0 == i].mean(0), dim=0)
    if cfg['projector']:
        # Also needs a much lower LR than the closed-form (no-projector) path: at the
        # default lr=0.01, training diverges steadily even from a consistent, already-
        # optimal init (0.990 -> 0.093 over 10 epochs, confirmed by direct sweep).
        clf, projector = train_stage0(clf, projector, Z0_a, Z0_b, y0, epochs=stage0_epochs,
                                      lr=0.001, seed=seed)

    class_to_stable_id = dict(stage0_labels)
    all_classes_ordered = list(known_classes)
    mu, shared_var, _ = update_prototypes(Z0_a, y0, n_old0)  # always raw (pre-projector) space

    per_stage_results = []
    prev_projector = None
    for t, stage in enumerate(stages):
        Zu_a = torch.tensor(fA[stage['unlabeled_idx']], dtype=torch.float32, device=DEVICE)
        Zu_b = torch.tensor(fB[stage['unlabeled_idx']], dtype=torch.float32, device=DEVICE)
        n_new = len(stage['new_classes'])
        n_old = clf.n_classes

        contam = None
        if cfg['cluster_init']:
            # cluster in the space the classifier actually operates in: projected,
            # if a projector exists, since clf.weight lives in that space, not raw
            with torch.no_grad():
                Zu_for_init = projector(Zu_a) if projector is not None else Zu_a
            new_w, is_novel = cluster_guided_init(Zu_for_init, clf.weight.data.clone(), n_new, seed=seed + t)
            # Contamination check: is_novel is a per-CLUSTER decision (see cluster_guided_init's
            # docstring) applied to every member, so it can disagree with ground truth per sample --
            # measure that directly here since this is a controlled experiment with real labels.
            true_is_new = np.isin(cls[stage['unlabeled_idx']], stage['new_classes'])
            contam = {
                'false_novel_rate': float(is_novel[~true_is_new].mean()) if (~true_is_new).any() else float('nan'),
                'missed_novel_rate': float((~is_novel[true_is_new]).mean()) if true_is_new.any() else float('nan'),
                'novel_precision': float(true_is_new[is_novel].mean()) if is_novel.any() else float('nan'),
            }
        else:
            new_w = torch.randn(n_new, fA.shape[1], device=DEVICE) * 0.01
        clf.grow(new_w)

        if cfg['projector']:
            prev_projector = copy.deepcopy(projector)
            for p in prev_projector.parameters():
                p.requires_grad_(False)

        clf = train_stage(clf, Zu_a, n_old, n_new, use_entropy_reg=cfg['entropy'],
                          use_hardness_replay=cfg['hardness'], mu=mu, shared_var=shared_var,
                          epochs=epochs, seed=seed + t,
                          lr=(0.001 if cfg['projector'] else 0.01), projector=projector,
                          Z_view2=Zu_b if cfg['projector'] else None,
                          prev_projector=prev_projector if cfg['projector'] else None)

        with torch.no_grad():
            zu_in = projector(Zu_a) if projector is not None else Zu_a
            logits = clf(zu_in)
            pseudo = logits.argmax(1)
        new_ids = list(range(n_old, n_old + n_new))
        for i, c in enumerate(stage['new_classes']):
            class_to_stable_id[c] = new_ids[i]  # provisional; remapped by Hungarian below
        all_classes_ordered += stage['new_classes']

        mu_all, shared_var, _ = update_prototypes(Zu_a, pseudo, clf.n_classes, shared_var=None)
        mu = torch.where((mu_all.abs().sum(1, keepdim=True) > 0), mu_all, torch.cat(
            [mu, torch.zeros(n_new, mu.shape[1], device=mu.device)], dim=0))

        Zt_a = torch.tensor(fA[stage['test_idx']], dtype=torch.float32, device=DEVICE)
        t_true = stage['test_true']
        with torch.no_grad():
            zt_in = projector(Zt_a) if projector is not None else Zt_a
            pred = clf(zt_in).argmax(1).cpu().numpy()

        old_mask = np.isin(t_true, all_classes_ordered[:n_old0] +
                           [c for s in stages[:t] for c in s['new_classes']])
        new_mask_this_stage = np.isin(t_true, stage['new_classes'])
        old_acc = np.mean([pred[i] == class_to_stable_id[t_true[i]] for i in np.where(old_mask)[0]]) \
            if old_mask.any() else float('nan')
        if new_mask_this_stage.any():
            true_new = t_true[new_mask_this_stage]
            pred_new = pred[new_mask_this_stage]
            new_acc, hmap = hungarian_acc(true_new, pred_new, clf.n_classes)
            for c in stage['new_classes']:
                if c in hmap:
                    class_to_stable_id[c] = hmap[c]
        else:
            new_acc = float('nan')
        all_acc = np.mean([pred[i] == class_to_stable_id.get(t_true[i], -999) for i in range(len(pred))])

        r = {'stage': t + 1, 'all': float(all_acc), 'old': float(old_acc), 'new': float(new_acc)}
        if contam is not None:
            r['contamination'] = contam
        per_stage_results.append(r)

    stage0_all = per_stage_results[0]['old']
    m_f = max(stage0_all - r['old'] for r in per_stage_results) if len(per_stage_results) > 1 else 0.0
    m_d = per_stage_results[-1]['all']
    return {'per_stage': per_stage_results, 'M_f': float(m_f), 'M_d': float(m_d)}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--subset', choices=['combined', 'UCF101', 'HMDB51'], default='combined')
    ap.add_argument('--stages', type=int, default=5)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--epochs', type=int, default=30)
    ap.add_argument('--stage0-epochs', type=int, default=30)
    ap.add_argument('--old-per-class', type=int, default=10)
    ap.add_argument('--config', choices=list(CONFIGS) + ['ladder'], default='ladder')
    ap.add_argument('--projector-hidden', type=int, default=512,
                    help='0 = single Linear residual, no nonlinearity/hidden layer')
    ap.add_argument('--projector-layers', type=int, default=1, help='number of hidden ReLU-Linear blocks')
    ap.add_argument('--no-projector-residual', dest='projector_residual', action='store_false',
                    help='plain feedforward projector (no skip connection), standard init')
    ap.set_defaults(projector_residual=True)
    ap.add_argument('--features', default='features_videomae.npz')
    ap.add_argument('--view2', default='features_videomae_view2.npz')
    ap.add_argument('--out', default=None)
    args = ap.parse_args()
    sub = None if args.subset == 'combined' else args.subset

    data, feats_a, feats_b = load_two_views(args.features, args.view2)
    cls, status = data['class_labels'], data['status']

    configs = CONFIGS if args.config == 'ladder' else {args.config: CONFIGS[args.config]}
    results = {}
    for name, cfg in configs.items():
        print(f"\n===== {name} =====")
        r = run_config(sub, cfg, args.stages, args.seed, feats_a, feats_b, cls, status,
                       args.epochs, args.old_per_class, args.stage0_epochs,
                       proj_hidden=args.projector_hidden, proj_layers=args.projector_layers,
                       proj_residual=args.projector_residual)
        for s in r['per_stage']:
            print(f"  stage {s['stage']}: all={s['all']:.3f} old={s['old']:.3f} new={s['new']:.3f}")
        print(f"  M_f={r['M_f']:.3f}  M_d={r['M_d']:.3f}")
        results[name] = r

    if args.out:
        with open(args.out, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\nsaved {args.out}")


if __name__ == '__main__':
    main()

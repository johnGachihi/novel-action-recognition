#!/usr/bin/env python3
"""Continual-stage (stage-1) learning-rate sweep. Stage-0 is trained ONCE with
lr=1e-2 (validated by sweep_stage0_lr.py: 0.957 final train_acc, far better than
the original 1e-5's 0.58 plateau) and its state is snapshotted, then reloaded
fresh for each continual-LR candidate -- this isolates the continual-stage LR as
the only variable, and avoids retraining stage-0 redundantly (also removes any
worker-shuffling nondeterminism between candidates that separate stage-0 runs
would introduce).

Small increments first, same reasoning as the original stage-0 sweep: this is
specifically the phase (self-training + entropy-reg + hardness-replay on
UNLABELED pseudo-labels) that showed real instability in the earlier frozen-
feature Projector experiments, so start close to the current default (1e-5)
rather than jumping straight to the large values that worked well for stage-0's
very different (supervised, true-label) optimization problem.
"""
import copy
import json

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import VideoMAEImageProcessor

from nac.happy import (GrowingClassifier, cluster_guided_init, group_entropy_reg,
                       sample_replay, supervised_contrastive_loss, unsupervised_contrastive_loss,
                       update_prototypes)
from nac.live_videomae import (PartiallyFrozenVideoMAE, TwoViewVideoDataset, build_manifest,
                               collate_two_view)
from nac.metrics import hungarian_acc
from run_happy_live import build_stage_splits, embed_batch, embed_paths, make_class_subset

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
NUM_WORKERS = 12
BATCH_SIZE = 32
STAGE0_LR = 1e-2       # validated: far better than 1e-5 for this (supervised) phase
STAGE0_EPOCHS = 10
CONTINUAL_EPOCHS = 5
SEED = 0
CONTINUAL_LR_CANDIDATES = [1e-5, 2e-5, 3e-5, 5e-5, 7e-5, 1e-4]


def train_stage0_once(stage0_paths, y0, n_old0, processor):
    torch.manual_seed(SEED)
    model = PartiallyFrozenVideoMAE().to(DEVICE)
    clf = GrowingClassifier(768, n_old0).to(DEVICE)
    ds0 = TwoViewVideoDataset(stage0_paths, processor)
    loader0 = DataLoader(ds0, batch_size=BATCH_SIZE, collate_fn=collate_two_view,
                        num_workers=NUM_WORKERS, shuffle=True, pin_memory=True, persistent_workers=True)

    with torch.no_grad():
        model.eval()
        Z0_init, _ = embed_paths(model, processor, stage0_paths, BATCH_SIZE, DEVICE, two_view=False)
        for i in range(n_old0):
            m = y0 == i
            if m.any():
                clf.weight[i] = F.normalize(Z0_init[m].mean(0), dim=0)

    model.train()
    opt = torch.optim.SGD(list(clf.parameters()) + model.trainable_parameters(), lr=STAGE0_LR, momentum=0.9)
    for ep in range(STAGE0_EPOCHS):
        correct, total = 0, 0
        for pa, pb, idxs in loader0:
            if pa is None:
                continue
            yb = y0[list(idxs)]
            za = embed_batch(model, pa, DEVICE)
            zb = embed_batch(model, pb, DEVICE)
            logits = clf(za)
            loss = F.cross_entropy(logits / 0.1, yb) + supervised_contrastive_loss(za, zb, yb)
            opt.zero_grad(); loss.backward(); opt.step()
            correct += (logits.argmax(1) == yb).sum().item()
            total += len(yb)
        print(f"  [stage0, lr={STAGE0_LR:.0e}] epoch {ep}: train_acc={correct/max(total,1):.3f}")

    model.eval()
    with torch.no_grad():
        Z0_a, _ = embed_paths(model, processor, stage0_paths, BATCH_SIZE, DEVICE, two_view=False)
    mu, shared_var, _ = update_prototypes(Z0_a, y0, n_old0)
    return {'model_state': copy.deepcopy(model.state_dict()),
           'clf_state': copy.deepcopy(clf.state_dict()),
           'mu': mu.clone(), 'shared_var': shared_var}


def main():
    split = json.load(open('class_split_even_odd.json'))
    rng = np.random.default_rng(SEED)
    known_classes, seen_novel, unseen_novel = make_class_subset(split, 15, 8, 7, SEED)
    records = build_manifest()
    records = [(p, c) for p, c in records if c in set(known_classes) | set(seen_novel) | set(unseen_novel)]
    stage0_paths, y0_np, stage0_labels, stages = build_stage_splits(
        records, known_classes, seen_novel, unseen_novel, T=3, test_frac=0.15, old_per_class=5, rng=rng)
    n_old0 = len(known_classes)
    y0 = torch.tensor(y0_np, dtype=torch.long, device=DEVICE)
    stage1 = stages[0]
    print(f"stage0: {len(stage0_paths)} videos, {n_old0} classes. "
          f"stage1: {len(stage1['new_classes'])} new classes, {len(stage1['unlabeled_paths'])} unlabeled\n")

    processor = VideoMAEImageProcessor.from_pretrained('MCG-NJU/videomae-base')

    print(f"=== training stage-0 ONCE at lr={STAGE0_LR:.0e} (shared starting point) ===")
    snapshot = train_stage0_once(stage0_paths, y0, n_old0, processor)

    class_to_stable_id = dict(stage0_labels)
    results = {}
    for lr in CONTINUAL_LR_CANDIDATES:
        print(f"\n=== continual lr={lr:.0e} ===")
        n_new = len(stage1['new_classes'])
        n_old = n_old0
        model = PartiallyFrozenVideoMAE().to(DEVICE)
        model.load_state_dict(snapshot['model_state'])
        clf = GrowingClassifier(768, n_old0).to(DEVICE)
        clf.load_state_dict(snapshot['clf_state'])
        mu, shared_var = snapshot['mu'].clone(), snapshot['shared_var']

        model.eval()
        with torch.no_grad():
            Zu_a, _ = embed_paths(model, processor, stage1['unlabeled_paths'], BATCH_SIZE, DEVICE, two_view=False)
        new_w, _ = cluster_guided_init(Zu_a, clf.weight.data.clone(), n_new, seed=SEED)
        clf.grow(new_w)
        new_ids = list(range(n_old, n_old + n_new))
        local_map = dict(class_to_stable_id)
        for i, c in enumerate(stage1['new_classes']):
            local_map[c] = new_ids[i]

        model.train()
        opt = torch.optim.SGD(list(clf.parameters()) + model.trainable_parameters(), lr=lr, momentum=0.9)
        dsU = TwoViewVideoDataset(stage1['unlabeled_paths'], processor)
        loaderU = DataLoader(dsU, batch_size=BATCH_SIZE, collate_fn=collate_two_view, num_workers=NUM_WORKERS,
                            shuffle=True, pin_memory=True, persistent_workers=True)
        diverged = False
        for ep in range(CONTINUAL_EPOCHS):
            for pa, pb, idxs in loaderU:
                if pa is None:
                    continue
                za = embed_batch(model, pa, DEVICE)
                logits = clf(za)
                p = F.softmax(logits / 0.1, dim=1)
                with torch.no_grad():
                    q = F.softmax(logits / 0.05, dim=1)
                loss = -(q * p.clamp_min(1e-12).log()).sum(1).mean()
                loss = loss + group_entropy_reg(p, n_old, clf.n_classes - n_old)
                zr, yr = sample_replay(mu, shared_var, n_old, n_samples=len(pa))
                loss = loss + F.cross_entropy(clf(zr) / 0.1, yr)
                zb = embed_batch(model, pb, DEVICE)
                loss = loss + unsupervised_contrastive_loss(za, zb)
                if torch.isnan(loss):
                    diverged = True
                    break
                opt.zero_grad(); loss.backward(); opt.step()
            if diverged:
                print(f"    epoch {ep}: DIVERGED (NaN)")
                break
            print(f"    epoch {ep}: loss={loss.item():.3f}")

        if diverged:
            results[lr] = {'old': float('nan'), 'new': float('nan'), 'all': float('nan')}
            continue

        model.eval()
        with torch.no_grad():
            Zt_a, _ = embed_paths(model, processor, stage1['test_paths'], BATCH_SIZE, DEVICE, two_view=False)
            pred = clf(Zt_a).argmax(1).cpu().numpy()
        t_true = stage1['test_true']
        old_mask = np.isin(t_true, known_classes)
        new_mask = np.isin(t_true, stage1['new_classes'])
        old_acc = np.mean([pred[i] == local_map[t_true[i]] for i in np.where(old_mask)[0]]) if old_mask.any() else float('nan')
        if new_mask.any():
            new_acc, hmap = hungarian_acc(t_true[new_mask], pred[new_mask], clf.n_classes)
        else:
            new_acc = float('nan')
        all_acc = np.mean([pred[i] == local_map.get(t_true[i], -999) for i in range(len(pred))])
        print(f"    RESULT lr={lr:.0e}: old={old_acc:.3f} new={new_acc:.3f} all={all_acc:.3f}")
        results[lr] = {'old': float(old_acc), 'new': float(new_acc), 'all': float(all_acc)}

    print("\n=== SUMMARY ===")
    for lr, r in results.items():
        print(f"lr={lr:.0e}  old={r['old']:.3f}  new={r['new']:.3f}  all={r['all']:.3f}")


if __name__ == '__main__':
    main()

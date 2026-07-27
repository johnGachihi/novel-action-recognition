#!/usr/bin/env python3
"""Stage-0-only learning-rate sweep for the live VideoMAE pipeline. Isolates just
the supervised stage-0 phase (skips the expensive continual stages entirely) so
each candidate LR can be tested in ~1/10th the time of a full run_happy_live.py
pass. Uses the SAME class subset (seed=0, n_known=15) as the medium-scale run
already completed, so the baseline lr=1e-5 result here is directly comparable to
that run's stage0 train_acc trajectory (0.595 -> 0.577 over 10 epochs).

Small increments as requested: 1e-5 was stable but underfit (plateaued ~0.58);
sweeping upward cautiously rather than jumping a full order of magnitude, given
how easily the Projector experiments destabilized under too-aggressive LR.
"""
import json

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import VideoMAEImageProcessor

from nac.happy import GrowingClassifier, supervised_contrastive_loss
from nac.live_videomae import (PartiallyFrozenVideoMAE, TwoViewVideoDataset, build_manifest,
                               collate_two_view)
from run_happy_live import build_stage_splits, embed_batch, make_class_subset

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
NUM_WORKERS = 12
BATCH_SIZE = 32
EPOCHS = 10
SEED = 0
LR_CANDIDATES = [1e-4, 1e-3, 1e-2]


def run_one_lr(lr, stage0_paths, y0, n_old0, processor):
    torch.manual_seed(SEED)
    model = PartiallyFrozenVideoMAE().to(DEVICE)
    clf = GrowingClassifier(768, n_old0).to(DEVICE)
    with torch.no_grad():
        model.eval()
        ds_init = TwoViewVideoDataset(stage0_paths, processor)
        loader_init = DataLoader(ds_init, batch_size=BATCH_SIZE, collate_fn=collate_two_view,
                                 num_workers=NUM_WORKERS, shuffle=False, pin_memory=True)
        all_z = []
        for pa, _, _ in loader_init:
            if pa is None:
                continue
            all_z.append(embed_batch(model, pa, DEVICE))
        Z0 = torch.cat(all_z, dim=0)
        for i in range(n_old0):
            m = y0 == i
            if m.any():
                clf.weight[i] = F.normalize(Z0[m].mean(0), dim=0)

    model.train()
    opt = torch.optim.SGD(list(clf.parameters()) + model.trainable_parameters(), lr=lr, momentum=0.9)
    loader = DataLoader(TwoViewVideoDataset(stage0_paths, processor), batch_size=BATCH_SIZE,
                       collate_fn=collate_two_view, num_workers=NUM_WORKERS, shuffle=True,
                       pin_memory=True, persistent_workers=True)
    trajectory = []
    for ep in range(EPOCHS):
        correct, total = 0, 0
        for pa, pb, idxs in loader:
            if pa is None:
                continue
            yb = y0[list(idxs)]
            za = embed_batch(model, pa, DEVICE)
            zb = embed_batch(model, pb, DEVICE)
            logits = clf(za)
            loss = F.cross_entropy(logits / 0.1, yb) + supervised_contrastive_loss(za, zb, yb)
            if torch.isnan(loss):
                trajectory.append(float('nan'))
                return trajectory  # diverged -- stop early, don't waste time
            opt.zero_grad(); loss.backward(); opt.step()
            correct += (logits.argmax(1) == yb).sum().item()
            total += len(yb)
        acc = correct / max(total, 1)
        trajectory.append(acc)
        print(f"    lr={lr:.0e} epoch {ep}: running_train_acc={acc:.3f}")
    del model, clf, opt, loader
    torch.cuda.empty_cache()
    return trajectory


def main():
    split = json.load(open('class_split_even_odd.json'))
    rng = np.random.default_rng(SEED)
    known_classes, seen_novel, unseen_novel = make_class_subset(split, 15, 8, 7, SEED)
    print(f"known={len(known_classes)} (matching prior medium-scale run's subset)")

    records = build_manifest()
    records = [(p, c) for p, c in records if c in set(known_classes) | set(seen_novel) | set(unseen_novel)]
    stage0_paths, y0_np, stage0_labels, _ = build_stage_splits(
        records, known_classes, seen_novel, unseen_novel, T=3, test_frac=0.15, old_per_class=5, rng=rng)
    n_old0 = len(known_classes)
    y0 = torch.tensor(y0_np, dtype=torch.long, device=DEVICE)
    print(f"stage0: {len(stage0_paths)} labeled videos across {n_old0} classes\n")

    processor = VideoMAEImageProcessor.from_pretrained('MCG-NJU/videomae-base')

    results = {}
    for lr in LR_CANDIDATES:
        print(f"=== lr={lr:.0e} ===")
        traj = run_one_lr(lr, stage0_paths, y0, n_old0, processor)
        results[lr] = traj

    print("\n=== SUMMARY (final epoch train_acc per LR) ===")
    for lr, traj in results.items():
        final = traj[-1] if traj else float('nan')
        peak = max(traj) if traj and not any(np.isnan(traj)) else float('nan')
        print(f"lr={lr:.0e}  final={final:.3f}  peak={peak:.3f}  trajectory={[round(x,3) for x in traj]}")


if __name__ == '__main__':
    main()

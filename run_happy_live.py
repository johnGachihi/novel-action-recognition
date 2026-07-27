#!/usr/bin/env python3
"""Happy-adapted continual GCD with LIVE VideoMAE fine-tuning: on-the-fly two-view
augmentation feeding gradients through the backbone's actual last transformer layer
(see nac/live_videomae.py), instead of nac/happy.py's frozen-feature + added-
Projector approximation. Reuses the same loss components (group-wise entropy reg,
hardness-aware replay, contrastive, KD) -- only the thing standing in for "the
trainable part of the model" changes: a real ViT block instead of a small MLP.

Much more expensive per step (full 12-layer ViT forward, live video decode, every
sample, both views) -- run on small subsets/short schedules first to validate
correctness before scaling up. Reports the same per-stage All/Old/New table as
run_happy.py.

Usage:
  python3 run_happy_live.py --n-known 10 --n-seen-novel 5 --n-unseen-novel 5 \\
      --stages 2 --epochs 2 --stage0-epochs 2 --batch-size 8   # quick smoke test
"""
import argparse
import copy
import json

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import VideoMAEImageProcessor

from nac.epic_data import build_epic_manifest
from nac.happy import (GrowingClassifier, cluster_guided_init, group_entropy_reg,
                       hardness_distribution, sample_replay, supervised_contrastive_loss,
                       unsupervised_contrastive_loss, update_prototypes)
from nac.live_videomae import (PartiallyFrozenVideoMAE, TwoViewVideoDataset, build_manifest,
                               collate_two_view)
from nac.metrics import hungarian_acc

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
NUM_WORKERS = 12


def embed_batch(model, pixel_values, device):
    return model(pixel_values.to(device))


def make_class_subset(split, n_known, n_seen_novel, n_unseen_novel, seed):
    rng = np.random.default_rng(seed)
    known = rng.choice(split['known'], size=min(n_known, len(split['known'])), replace=False).tolist()
    novel = rng.choice(split['novel'], size=min(n_seen_novel + n_unseen_novel, len(split['novel'])), replace=False).tolist()
    return known, novel[:n_seen_novel], novel[n_seen_novel:n_seen_novel + n_unseen_novel]


def build_stage_splits(records, known_classes, seen_novel, unseen_novel, T, test_frac, old_per_class, rng):
    """Manifest-based analogue of nac.happy.build_multistage_splits -- operates on
    (path, class) records directly instead of a pre-extracted feature array."""
    by_class = {}
    for p, c in records:
        by_class.setdefault(c, []).append(p)
    for c in by_class:
        rng.shuffle(by_class[c])

    test_pool, pool = {}, {}
    for c in list(known_classes) + seen_novel + unseen_novel:
        paths = by_class.get(c, [])
        n_test = max(1, int(test_frac * len(paths))) if paths else 0
        test_pool[c] = paths[:n_test]
        pool[c] = paths[n_test:]

    stage0_labels = {c: i for i, c in enumerate(known_classes)}
    stage0_paths, stage0_y = [], []
    for c in known_classes:
        n0 = max(1, int(0.4 * len(pool[c])))
        for p in pool[c][:n0]:
            stage0_paths.append(p); stage0_y.append(stage0_labels[c])
        pool[c] = pool[c][n0:]

    seen_chunks = np.array_split(rng.permutation(len(seen_novel)), T)
    unseen_chunks = np.array_split(rng.permutation(len(unseen_novel)), T)
    stages = []
    old_so_far = list(known_classes)
    cumulative_test_classes = list(known_classes)
    for t in range(T):
        new_classes = [seen_novel[i] for i in seen_chunks[t]] + [unseen_novel[i] for i in unseen_chunks[t]]
        unl_paths, unl_true = [], []
        for c in new_classes:
            for p in pool[c]:
                unl_paths.append(p); unl_true.append(c)
            pool[c] = []
        for c in old_so_far:
            take = pool[c][:old_per_class]
            pool[c] = pool[c][old_per_class:]
            for p in take:
                unl_paths.append(p); unl_true.append(c)
        cumulative_test_classes = cumulative_test_classes + new_classes
        test_paths = [p for c in cumulative_test_classes for p in test_pool[c]]
        test_true = [c for c in cumulative_test_classes for _ in test_pool[c]]
        stages.append({'new_classes': new_classes, 'unlabeled_paths': unl_paths,
                       'unlabeled_true': np.array(unl_true), 'test_paths': test_paths,
                       'test_true': np.array(test_true)})
        old_so_far = old_so_far + new_classes
    return stage0_paths, np.array(stage0_y), stage0_labels, stages


def embed_paths(model, processor, paths, batch_size, device, two_view=True, num_workers=NUM_WORKERS):
    """Runs the live model over a path list, batched, WITH grad if model.training."""
    ds = TwoViewVideoDataset(paths, processor)
    loader = DataLoader(ds, batch_size=batch_size, collate_fn=collate_two_view, num_workers=num_workers,
                        shuffle=False, pin_memory=True, persistent_workers=(num_workers > 0))
    all_a, all_b = [], []
    for pa, pb, _ in loader:
        if pa is None:
            continue
        za = embed_batch(model, pa, device)
        all_a.append(za)
        if two_view:
            zb = embed_batch(model, pb, device)
            all_b.append(zb)
    Za = torch.cat(all_a, dim=0)
    Zb = torch.cat(all_b, dim=0) if two_view else None
    return Za, Zb


def save_stage_checkpoint(prefix, stage_num, model, clf, mu, shared_var,
                          class_to_stable_id, all_classes_ordered, known_classes, n_old0):
    """Everything needed to reload the model + classifier and resume evaluation
    (or continue training) at this exact stage without retraining from scratch --
    same rationale as nac/stage2.py's checkpoint dict. stage_num=0 is the state
    right after stage-0 supervised training, before any continual stage runs."""
    if not prefix:
        return
    ckpt = {
        'stage': stage_num,
        'model_state': {k: v.cpu().clone() for k, v in model.state_dict().items()},
        'clf_state': {k: v.cpu().clone() for k, v in clf.state_dict().items()},
        'n_classes': clf.n_classes,
        'mu': mu.cpu().clone(),
        'shared_var': shared_var,
        'class_to_stable_id': dict(class_to_stable_id),
        'all_classes_ordered': list(all_classes_ordered),
        'known_classes': list(known_classes),
        'n_old0': n_old0,
    }
    path = f'{prefix}_stage{stage_num}.pt'
    torch.save(ckpt, path)
    print(f"  saved checkpoint: {path}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--n-known', type=int, default=10)
    ap.add_argument('--n-seen-novel', type=int, default=5)
    ap.add_argument('--n-unseen-novel', type=int, default=5)
    ap.add_argument('--stages', type=int, default=2)
    ap.add_argument('--epochs', type=int, default=2)
    ap.add_argument('--stage0-epochs', type=int, default=2)
    ap.add_argument('--old-per-class', type=int, default=5)
    ap.add_argument('--batch-size', type=int, default=32)
    ap.add_argument('--num-workers', type=int, default=12)
    ap.add_argument('--stage0-lr', type=float, default=1e-2,
                    help='validated via sweep_stage0_lr.py: supervised phase tolerates/needs a much higher LR than continual')
    ap.add_argument('--continual-lr', type=float, default=5e-5,
                    help='validated via sweep_continual_lr.py: self-training phase is far more LR-sensitive than stage0')
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--split', default='class_split_even_odd.json',
                    help='ignored when --dataset epic (uses class_split_epic.json[epic-label-space] instead)')
    ap.add_argument('--dataset', choices=['ucf_hmdb', 'epic'], default='ucf_hmdb')
    ap.add_argument('--epic-label-space', choices=['verb', 'noun'], default='verb',
                    help='which EPIC label space to use as the class set (only relevant with --dataset epic)')
    ap.add_argument('--checkpoint-prefix', default='happy_live_ckpt',
                    help='saves <prefix>_stage0.pt (post stage-0 training) and <prefix>_stage{1..T}.pt '
                         '(after each continual stage) -- everything needed to reload the model/classifier '
                         'and resume evaluation without retraining. Pass "" to disable.')
    args = ap.parse_args()

    if args.dataset == 'epic':
        split = json.load(open('class_split_epic.json'))[args.epic_label_space]
    else:
        split = json.load(open(args.split))
    rng = np.random.default_rng(args.seed)
    known_classes, seen_novel, unseen_novel = make_class_subset(
        split, args.n_known, args.n_seen_novel, args.n_unseen_novel, args.seed)
    print(f"known={len(known_classes)} seen_novel={len(seen_novel)} unseen_novel={len(unseen_novel)}")

    if args.dataset == 'epic':
        print(f"building EPIC manifest ({args.epic_label_space})...")
        records = build_epic_manifest(args.epic_label_space)
    else:
        print("building manifest from raw video files...")
        records = build_manifest()
    records = [(p, c) for p, c in records if c in set(known_classes) | set(seen_novel) | set(unseen_novel)]
    print(f"{len(records)} videos in the class subset")

    stage0_paths, y0_np, stage0_labels, stages = build_stage_splits(
        records, known_classes, seen_novel, unseen_novel, args.stages, 0.15, args.old_per_class, rng)
    n_old0 = len(known_classes)
    print(f"stage0: {len(stage0_paths)} labeled videos across {n_old0} classes")

    processor = VideoMAEImageProcessor.from_pretrained('MCG-NJU/videomae-base')
    model = PartiallyFrozenVideoMAE().to(DEVICE)
    clf = GrowingClassifier(768, n_old0).to(DEVICE)
    y0 = torch.tensor(y0_np, dtype=torch.long, device=DEVICE)

    print("initial frozen forward pass over stage-0 (for class-mean warm start)...")
    model.eval()
    with torch.no_grad():
        Z0_a_init, _ = embed_paths(model, processor, stage0_paths, args.batch_size, DEVICE, two_view=False)
    with torch.no_grad():
        for i in range(n_old0):
            m = y0 == i
            if m.any():
                clf.weight[i] = F.normalize(Z0_a_init[m].mean(0), dim=0)
    print(f"class-mean init done, embedding shape {Z0_a_init.shape}")

    # ---- stage 0: real gradient training through the last ViT block ----
    # Loader built ONCE outside the epoch loop (a fresh DataLoader every epoch would
    # respawn the whole worker pool each time, making persistent_workers a no-op).
    # Per-epoch accuracy is tracked from the training batches' OWN predictions --
    # no separate full-dataset eval pass, which previously doubled all of stage-0's
    # compute (one pass to train on, one full extra pass every epoch just to log
    # accuracy, computing the identical thing the training pass already computed).
    model.train()
    opt = torch.optim.SGD(list(clf.parameters()) + model.trainable_parameters(), lr=args.stage0_lr, momentum=0.9)
    ds0 = TwoViewVideoDataset(stage0_paths, processor)
    loader0 = DataLoader(ds0, batch_size=args.batch_size, collate_fn=collate_two_view,
                        num_workers=args.num_workers, shuffle=True, pin_memory=True,
                        persistent_workers=(args.num_workers > 0))
    for ep in range(args.stage0_epochs):
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
        print(f"  stage0 epoch {ep}: running_train_acc={correct / max(total, 1):.3f}")

    model.eval()
    with torch.no_grad():
        Z0_a, _ = embed_paths(model, processor, stage0_paths, args.batch_size, DEVICE, two_view=False)
    mu, shared_var, _ = update_prototypes(Z0_a, y0, n_old0)

    class_to_stable_id = dict(stage0_labels)
    all_classes_ordered = list(known_classes)
    per_stage_results = []
    prev_model_state = None

    save_stage_checkpoint(args.checkpoint_prefix, 0, model, clf, mu, shared_var,
                          class_to_stable_id, all_classes_ordered, known_classes, n_old0)

    for t, stage in enumerate(stages):
        n_new = len(stage['new_classes'])
        n_old = clf.n_classes
        print(f"\n=== stage {t+1}: {n_new} new classes, {len(stage['unlabeled_paths'])} unlabeled videos ===")

        model.eval()
        with torch.no_grad():
            Zu_a, _ = embed_paths(model, processor, stage['unlabeled_paths'], args.batch_size, DEVICE, two_view=False)
        new_w, is_novel = cluster_guided_init(Zu_a, clf.weight.data.clone(), n_new, seed=args.seed + t)
        clf.grow(new_w)
        true_is_new = np.isin(stage['unlabeled_true'], stage['new_classes'])
        contam = {
            'false_novel_rate': float(is_novel[~true_is_new].mean()) if (~true_is_new).any() else float('nan'),
            'missed_novel_rate': float((~is_novel[true_is_new]).mean()) if true_is_new.any() else float('nan'),
            'novel_precision': float(true_is_new[is_novel].mean()) if is_novel.any() else float('nan'),
        }
        print(f"  contamination: {contam}")

        prev_model_state = copy.deepcopy(model.state_dict())

        model.train()
        opt = torch.optim.SGD(list(clf.parameters()) + model.trainable_parameters(), lr=args.continual_lr, momentum=0.9)
        dsU = TwoViewVideoDataset(stage['unlabeled_paths'], processor)
        loaderU = DataLoader(dsU, batch_size=args.batch_size, collate_fn=collate_two_view,
                            num_workers=args.num_workers, shuffle=True, pin_memory=True,
                            persistent_workers=(args.num_workers > 0))
        for ep in range(args.epochs):
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
                opt.zero_grad(); loss.backward(); opt.step()
            print(f"  stage{t+1} epoch {ep}: loss={loss.item():.3f}")

        model.eval()
        with torch.no_grad():
            Zu_a_post, _ = embed_paths(model, processor, stage['unlabeled_paths'], args.batch_size, DEVICE, two_view=False)
            pseudo = clf(Zu_a_post).argmax(1)
        new_ids = list(range(n_old, n_old + n_new))
        for i, c in enumerate(stage['new_classes']):
            class_to_stable_id[c] = new_ids[i]
        all_classes_ordered += stage['new_classes']
        mu_all, shared_var, _ = update_prototypes(Zu_a_post, pseudo, clf.n_classes, shared_var=None)
        mu = torch.where((mu_all.abs().sum(1, keepdim=True) > 0), mu_all,
                        torch.cat([mu, torch.zeros(n_new, mu.shape[1], device=mu.device)], dim=0))

        with torch.no_grad():
            Zt_a, _ = embed_paths(model, processor, stage['test_paths'], args.batch_size, DEVICE, two_view=False)
            pred = clf(Zt_a).argmax(1).cpu().numpy()
        t_true = stage['test_true']
        old_mask = np.isin(t_true, all_classes_ordered[:n_old0] + [c for s in stages[:t] for c in s['new_classes']])
        new_mask = np.isin(t_true, stage['new_classes'])
        old_acc = np.mean([pred[i] == class_to_stable_id[t_true[i]] for i in np.where(old_mask)[0]]) if old_mask.any() else float('nan')
        if new_mask.any():
            new_acc, hmap = hungarian_acc(t_true[new_mask], pred[new_mask], clf.n_classes)
            for c in stage['new_classes']:
                if c in hmap:
                    class_to_stable_id[c] = hmap[c]
        else:
            new_acc = float('nan')
        all_acc = np.mean([pred[i] == class_to_stable_id.get(t_true[i], -999) for i in range(len(pred))])
        print(f"  RESULT stage {t+1}: all={all_acc:.3f} old={old_acc:.3f} new={new_acc:.3f}")
        per_stage_results.append({'stage': t + 1, 'all': float(all_acc), 'old': float(old_acc),
                                  'new': float(new_acc), 'contamination': contam})
        save_stage_checkpoint(args.checkpoint_prefix, t + 1, model, clf, mu, shared_var,
                              class_to_stable_id, all_classes_ordered, known_classes, n_old0)

    print("\n=== SUMMARY ===")
    for r in per_stage_results:
        print(r)


if __name__ == '__main__':
    main()

"""Feature loading and split construction for EPIC-KITCHENS-100 (verb/noun).

Mirrors nac/data.py's design for UCF101/HMDB51, generalized to two independent
label spaces (verb_class, noun_class) over the same clip pool, grouped by
video_id (EPIC's analogue of UCF101's source-video _gXX_cXX grouping -- many
narrations come from the same continuous kitchen recording, so a random
per-clip split would leak background/lighting/participant identity across
train/heldout).
"""
import glob
import os
import json

import numpy as np
import pandas as pd

CLIPS_ROOT = 'data/epic_kitchens_clips/clips'
ANNOT_TRAIN = 'epic-kitchens-100-annotations/EPIC_100_train.csv'


def load_epic_manifest(min_count=20, clips_root=CLIPS_ROOT, annot_path=ANNOT_TRAIN):
    """Join downloaded clips against EPIC-KITCHENS-100 annotations.
    If selected_participants.json exists, we load both train and validation annotation CSVs
    to support disjoint participant splits. Returns a filtered DataFrame."""
    if isinstance(clips_root, str):
        clips_roots = [r.strip() for r in clips_root.split(',')]
    else:
        clips_roots = clips_root
        
    files = []
    for root in clips_roots:
        files.extend(glob.glob(f'{root}/*/*.mp4'))
        
    ids_to_path = {os.path.basename(f)[:-4]: f for f in files}

    if os.path.exists('selected_participants.json') or annot_path == ANNOT_TRAIN:
        df_train = pd.read_csv('epic-kitchens-100-annotations/EPIC_100_train.csv')
        df_val = pd.read_csv('epic-kitchens-100-annotations/EPIC_100_validation.csv')
        df = pd.concat([df_train, df_val], ignore_index=True)
        df = df.drop_duplicates(subset=['narration_id'])
    else:
        df = pd.read_csv(annot_path)

    if os.path.exists('selected_participants.json'):
        with open('selected_participants.json') as f:
            sel = json.load(f)
        all_sel_pids = set(sel['train'] + sel['val'] + sel['test'])
        df = df[df.participant_id.isin(all_sel_pids)].copy()

    df = df[df.narration_id.isin(ids_to_path)].copy()
    df['path'] = df.narration_id.map(ids_to_path)

    vc = df.verb_class.value_counts()
    nc = df.noun_class.value_counts()
    df['verb_keep'] = df.verb_class.isin(vc[vc >= min_count].index)
    df['noun_keep'] = df.noun_class.isin(nc[nc >= min_count].index)
    return df


def build_epic_manifest(label_space, min_count=20, clips_root=CLIPS_ROOT, annot_path=ANNOT_TRAIN):
    """(path, class_id) pairs for the live fine-tuning pipeline (run_happy_live.py)
    -- EPIC analogue of nac.live_videomae.build_manifest. class_id is the raw
    verb_class/noun_class int (not a 'DATASET:name' string like UCF/HMDB use --
    downstream code here is already label-dtype-agnostic, see nac/epic_stage1.py)."""
    assert label_space in ('verb', 'noun')
    df = load_epic_manifest(min_count=min_count, clips_root=clips_root, annot_path=annot_path)
    keep = df[f'{label_space}_keep']
    labels = df[f'{label_space}_class']
    return list(zip(df.path[keep], labels[keep]))


def even_odd_split(class_ids):
    """Sorted numeric class ids, alternating even/odd index -> known/novel.
    Mirrors class_split_even_odd.json's alphabetical-name convention, adapted
    to integer verb/noun class ids (no natural name string to sort by here)."""
    ids = sorted(class_ids)
    known = [c for i, c in enumerate(ids) if i % 2 == 0]
    novel = [c for i, c in enumerate(ids) if i % 2 == 1]
    return known, novel


def _split_units(idx, video_ids, rng, cut_fn):
    """Split one class's samples at video-group level. Falls back to per-clip
    splitting if the class occurs in only a single video_id (can't group-split
    a single group) -- affects 3/163 kept noun classes, 0/69 kept verb classes."""
    groups = {}
    for i in idx:
        groups.setdefault(video_ids[i], []).append(i)
    if len(groups) > 1:
        units = list(groups.keys())
        rng.shuffle(units)
        expand = lambda us: [i for u in us for i in groups[u]]
    else:
        units = list(idx)
        rng.shuffle(units)
        expand = lambda us: list(us)
    cuts = cut_fn(len(units))
    parts, prev = [], 0
    for c in list(cuts) + [len(units)]:
        parts.append(expand(units[prev:c]))
        prev = c
    return parts


def known_train_heldout(classes, labels, video_ids, rng, frac=0.8):
    """Stage-1 split: per-class group-aware train/heldout (cut at int(frac*n))."""
    if os.path.exists('selected_participants.json'):
        with open('selected_participants.json') as f:
            sel = json.load(f)
        train_pids = set(sel['train'])
        val_pids = set(sel['val'])
        test_pids = set(sel['test'])
        
        train, heldout = [], []
        for c in classes:
            idx = np.where(labels == c)[0]
            for i in idx:
                pid = video_ids[i].split('_')[0]
                if pid in train_pids:
                    train.append(i)
                elif pid in val_pids or pid in test_pids:
                    heldout.append(i)
        return np.array(train, dtype=int), np.array(heldout, dtype=int)

    train, heldout = [], []
    for c in classes:
        idx = np.where(labels == c)[0]
        tr, ho = _split_units(idx, video_ids, rng, lambda n: [int(frac * n)])
        train += tr
        heldout += ho
    return np.array(train, dtype=int), np.array(heldout, dtype=int)


def known_three_way(classes, labels, video_ids, rng, frac=0.8):
    """Stage-2 split: train / phase-1 heldout / phase-2 heldout, group-aware."""
    if os.path.exists('selected_participants.json'):
        with open('selected_participants.json') as f:
            sel = json.load(f)
        train_pids = set(sel['train'])
        val_pids = set(sel['val'])
        test_pids = set(sel['test'])
        
        train, ho1, ho2 = [], [], []
        for c in classes:
            idx = np.where(labels == c)[0]
            for i in idx:
                pid = video_ids[i].split('_')[0]
                if pid in train_pids:
                    train.append(i)
                elif pid in val_pids:
                    ho1.append(i)
                elif pid in test_pids:
                    ho2.append(i)
        return np.array(train, dtype=int), np.array(ho1, dtype=int), np.array(ho2, dtype=int)

    train, ho1, ho2 = [], [], []

    def cuts(n):
        sp = int(frac * n)
        return [sp, sp + (n - sp) // 2]

    for c in classes:
        idx = np.where(labels == c)[0]
        tr, h1, h2 = _split_units(idx, video_ids, rng, cuts)
        train += tr
        ho1 += h1
        ho2 += h2
    return np.array(train, dtype=int), np.array(ho1, dtype=int), np.array(ho2, dtype=int)


def novel_phase_split(novel_classes, labels, rng, n_seen):
    """Split novel classes into seen (phase-1, samples halved across phases) and
    unseen (phase-2 only). Sample-level, not group-aware."""
    if os.path.exists('selected_participants.json'):
        with open('selected_participants.json') as f:
            sel = json.load(f)
        val_pids = set(sel['val'])
        test_pids = set(sel['test'])
        
        d = np.load('features_epic.npz', allow_pickle=True)
        video_ids = d['video_id']
        
        perm = rng.permutation(len(novel_classes))
        seen = set(novel_classes[perm[:n_seen]])
        sn1, sn2, un2 = [], [], []
        for c in novel_classes:
            idx = np.where(labels == c)[0]
            if c in seen:
                for i in idx:
                    pid = video_ids[i].split('_')[0]
                    if pid in val_pids:
                        sn1.append(i)
                    elif pid in test_pids:
                        sn2.append(i)
            else:
                for i in idx:
                    pid = video_ids[i].split('_')[0]
                    if pid in test_pids:
                        un2.append(i)
        return seen, np.array(sn1, dtype=int), np.array(sn2, dtype=int), np.array(un2, dtype=int)

    perm = rng.permutation(len(novel_classes))
    seen = set(novel_classes[perm[:n_seen]])
    sn1, sn2, un2 = [], [], []
    for c in novel_classes:
        idx = np.where(labels == c)[0]
        rng.shuffle(idx)
        if c in seen:
            half = len(idx) // 2
            sn1 += list(idx[:half])
            sn2 += list(idx[half:])
        else:
            un2 += list(idx)
    return seen, np.array(sn1, dtype=int), np.array(sn2, dtype=int), np.array(un2, dtype=int)

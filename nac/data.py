"""Feature loading and split construction.

All splits are deterministic given a seed. UCF101 splits are group-aware (clips from
the same source video — the _gXX_ id in the filename — never straddle a split
boundary); HMDB51 filenames carry no group structure, so those split per-clip.
"""
import re
import numpy as np


def load_features(path='features_videomae.npz'):
    d = np.load(path, allow_pickle=True)
    return {
        'features': d['features'],
        'status': d['labels'].astype(str),              # 'known' / 'novel'
        'class_labels': d['class_labels'].astype(str),  # e.g. 'UCF101:Archery'
        'paths': d['paths'],
    }


def get_group(path):
    m = re.search(r'_g(\d+)_c\d+', str(path))
    return m.group(1) if m else None


def subset_mask(class_labels, subset=None):
    """Boolean mask for a dataset subset ('UCF101', 'HMDB51') or everything (None)."""
    if subset is None:
        return np.ones(len(class_labels), bool)
    return np.char.startswith(class_labels, f'{subset}:')


def _split_units(idx, class_name, paths, rng, cut_fn):
    """Split one class's samples at unit level (source-video groups for UCF101,
    clips for HMDB51). cut_fn(n_units) -> list of cut positions. Returns the parts."""
    if class_name.startswith('UCF101:'):
        groups = {}
        for i in idx:
            groups.setdefault(get_group(paths[i]) or str(i), []).append(i)
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


def known_train_heldout(classes, class_labels, paths, rng, frac=0.8):
    """Stage-1 split: per-class group-aware train/heldout (cut at int(frac*n))."""
    train, heldout = [], []
    for c in classes:
        idx = np.where(class_labels == c)[0]
        tr, ho = _split_units(idx, c, paths, rng, lambda n: [int(frac * n)])
        train += tr
        heldout += ho
    return np.array(train), np.array(heldout)


def known_three_way(classes, class_labels, paths, rng, frac=0.8):
    """Stage-2 split: train / phase-1 heldout / phase-2 heldout.
    Cuts: sp = int(frac*n), sp2 = sp + (n - sp)//2 (heldout halved, integer math)."""
    train, ho1, ho2 = [], [], []

    def cuts(n):
        sp = int(frac * n)
        return [sp, sp + (n - sp) // 2]

    for c in classes:
        idx = np.where(class_labels == c)[0]
        tr, h1, h2 = _split_units(idx, c, paths, rng, cuts)
        train += tr
        ho1 += h1
        ho2 += h2
    return np.array(train), np.array(ho1), np.array(ho2)


def novel_phase_split(novel_classes, class_labels, rng, n_seen):
    """Split novel classes into seen (phase-1, samples halved across phases) and
    unseen (phase-2 only). Returns (seen_class_set, sn1, sn2, un2)."""
    perm = rng.permutation(len(novel_classes))
    seen = set(novel_classes[perm[:n_seen]])
    sn1, sn2, un2 = [], [], []
    for c in novel_classes:
        idx = np.where(class_labels == c)[0]
        rng.shuffle(idx)
        if c in seen:
            half = len(idx) // 2
            sn1 += list(idx[:half])
            sn2 += list(idx[half:])
        else:
            un2 += list(idx)
    return seen, np.array(sn1), np.array(sn2), np.array(un2)

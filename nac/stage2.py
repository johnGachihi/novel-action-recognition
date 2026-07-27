"""Stage 2: continual novel category discovery (notebook Section 10).

Phase 0: labeled knowns. Phase 1: GCD-style semi-supervised k-means over held-out
knowns + seen-novel classes. Phase 2: inference stream containing never-clustered
unseen-novel classes; methods must flag unknowns AND mint new categories.

Ablation switches (each defaults to the correct variant; the alternatives are the
failure modes documented in Section 10):
  whiten        True | False        (False: raw feature geometry)
  calib         'heldout'|'members' ('members': self-fulfilling radii)
  buffer_mode   'alone'|'anchored'  ('anchored': anchors re-absorb the buffer)
  merge         'spacing'|'sample-radius'|'none'
"""
import numpy as np

from .clustering import (discover_from_buffer, dpmeans_online, rejection_radii,
                         semisup_kmeans)
from .data import (get_group, known_three_way, load_features, novel_phase_split,
                   subset_mask)
from .evidential import make_edl_loss, predict_alpha, train_head, vacuity
from .geometry import dist2, fit_whitener
from .metrics import hungarian_acc, normalized_mutual_info_score

ALL_METHODS = ['closed_world', 'dist_reject', 'dpmeans', 'dear_reject']


def run_continual(subset=None, methods=ALL_METHODS, seed=0, seen_ratio=40 / 75,
                  whiten=True, calib='heldout', calib_q=90, buffer_mode='alone',
                  merge='spacing', gate_epochs=50, p1_assignment='greedy',
                  sinkhorn_eps=0.05, buffer_clusterer='kmeans', min_cluster_size=10,
                  features_path='features_videomae.npz', verbose=True):
    data = load_features(features_path)
    feats_raw, class_labels, paths = data['features'], data['class_labels'], data['paths']
    in_sub = subset_mask(class_labels, subset)
    known_classes = np.unique(class_labels[(data['status'] == 'known') & in_sub])
    novel_classes = np.unique(class_labels[(data['status'] == 'novel') & in_sub])
    NK = len(known_classes)
    known_to_idx = {c: i for i, c in enumerate(known_classes)}
    n_seen = round(len(novel_classes) * seen_ratio)
    n_unseen = len(novel_classes) - n_seen

    rng = np.random.default_rng(seed)
    train_idx, ho1, ho2 = known_three_way(known_classes, class_labels, paths, rng)
    _, sn1, sn2, un2 = novel_phase_split(novel_classes, class_labels, rng, n_seen)

    phase1_idx = np.concatenate([ho1, sn1])
    stream_idx = np.concatenate([ho2, sn2, un2])
    stream_type = np.array(['known'] * len(ho2) + ['seen_novel'] * len(sn2) + ['unseen_novel'] * len(un2))
    order = rng.permutation(len(stream_idx))
    stream_idx, stream_type = stream_idx[order], stream_type[order]
    if verbose:
        print(f"\n===== [{subset or 'combined'}] K={NK} seen-novel={n_seen} unseen-novel={n_unseen} | "
              f"train={len(train_idx)} phase1={len(phase1_idx)} stream={len(stream_idx)} =====")

    tr_labels = np.array([known_to_idx[c] for c in class_labels[train_idx]])
    if whiten:
        W, _ = fit_whitener(feats_raw[train_idx], tr_labels, NK)
        feats = feats_raw @ W
    else:
        feats = feats_raw

    # ---- phase 1: discovery ----
    X_lab, y_lab = feats[train_idx], tr_labels
    anchors0 = np.stack([X_lab[y_lab == k].mean(0) for k in range(NK)])
    C1, assign1 = semisup_kmeans(X_lab, y_lab, feats[phase1_idx], anchors0, n_free=n_seen,
                                 assignment=p1_assignment, sinkhorn_eps=sinkhorn_eps)
    N1 = len(C1)

    p1_true = class_labels[phase1_idx]
    p1_known = np.isin(p1_true, known_classes)
    p1_known_acc = float(np.mean(assign1[p1_known] == np.array([known_to_idx[c] for c in p1_true[p1_known]])))
    sn_acc, sn_map = hungarian_acc(p1_true[~p1_known], assign1[~p1_known], N1)
    frac_free = float(np.mean(assign1[~p1_known] >= NK))
    if verbose:
        print(f"[phase 1] known acc {p1_known_acc:.3f}  seen-novel Hungarian {sn_acc:.3f}  "
              f"novel->free rate {frac_free:.3f}")

    ho1_cls = np.array([known_to_idx[c] for c in class_labels[ho1]])
    radii = rejection_radii(C1, NK, feats, ho1, ho1_cls, feats[phase1_idx], assign1,
                            q=calib_q, calib=calib)

    # ---- phase 2: inference stream ----
    Xs = feats[stream_idx]
    s_true = class_labels[stream_idx]
    results = {}

    def evaluate(pred_cluster, is_new, new_ids, name):
        km = stream_type == 'known'
        sm = stream_type == 'seen_novel'
        um = stream_type == 'unseen_novel'
        r = {
            'known_acc': float(np.mean(pred_cluster[km] == np.array([known_to_idx[c] for c in s_true[km]]))),
            'seen_novel_acc': float(np.mean([sn_map.get(t, -1) == c
                                             for t, c in zip(s_true[sm], pred_cluster[sm])])),
            'unseen_detect_recall': float(is_new[um].mean()),
            'false_new_rate': float(is_new[km | sm].mean()),
            'n_new_categories': len(new_ids),
        }
        if is_new[um].sum() > 20 and len(new_ids) > 1:
            m = um & is_new
            acc, _ = hungarian_acc(s_true[m], pred_cluster[m] - N1, len(new_ids))
            r['new_cat_hungarian_acc'] = float(acc)
            r['new_cat_nmi'] = float(normalized_mutual_info_score(s_true[m], pred_cluster[m]))
        results[name] = r
        if verbose:
            print(f"  {name:18s} " + "  ".join(f"{k}={v:.3f}" if isinstance(v, float) else f"{k}={v}"
                                               for k, v in r.items()))

    D2s = dist2(Xs, C1)
    pred_nearest = D2s.argmin(1)

    def discover(buffer_idx, pred_base):
        """Route rejected samples to categories according to buffer_mode/merge."""
        out = pred_base.copy()
        new_centroids = None
        if buffer_mode == 'alone':
            out[buffer_idx], new_ids, new_centroids = discover_from_buffer(
                Xs[buffer_idx], C1, radii, n_unseen, merge=merge,
                clusterer=buffer_clusterer, min_cluster_size=min_cluster_size)
        elif buffer_mode == 'anchored':  # ablation: anchors re-absorb the buffer
            acc_m = np.ones(len(Xs), bool)
            acc_m[buffer_idx] = False
            C2, a2 = semisup_kmeans(np.concatenate([X_lab, Xs[acc_m]]),
                                    np.concatenate([y_lab, pred_base[acc_m]]),
                                    Xs[buffer_idx], C1, n_free=n_unseen, seed=1)
            out[buffer_idx] = a2
            new_ids = list(range(N1, len(C2)))
            new_centroids = C2[N1:]
        else:
            raise ValueError(buffer_mode)
        return out, new_ids, new_centroids

    checkpoint = None
    for name in methods:
        if name == 'closed_world':
            evaluate(pred_nearest, np.zeros(len(Xs), bool), [], name)
        elif name == 'dist_reject':
            rej = np.sqrt(D2s.min(1)) > radii[pred_nearest]
            pred, new_ids, _ = discover(np.where(rej)[0], pred_nearest)
            evaluate(pred, np.isin(pred, new_ids), new_ids, name)
        elif name == 'dpmeans':
            lam = np.percentile(radii, 90)
            pred, new_ids = dpmeans_online(Xs, C1, radii, lam)
            evaluate(pred, pred >= N1, new_ids, name)
        elif name == 'dear_reject':
            free_m = assign1 >= NK  # phase-1 anchored routings excluded: ho1 stays a clean calibration set
            Xd = np.concatenate([feats[train_idx], feats[phase1_idx][free_m]])
            yd = np.concatenate([y_lab, assign1[free_m]])
            mu, sig = Xd.mean(0), Xd.std(0) + 1e-6
            loss = make_edl_loss(N1, total_epoch=gate_epochs)  # DEAR best config
            head = train_head((Xd - mu) / sig, yd, N1, loss, epochs=gate_epochs, seed=0)
            u_ho = vacuity(predict_alpha(head, (feats[ho1] - mu) / sig), N1)
            tau = np.percentile(u_ho, calib_q)
            alpha_s = predict_alpha(head, (Xs - mu) / sig)
            u_s = vacuity(alpha_s, N1)
            pred_base = alpha_s.argmax(1).cpu().numpy()
            rej = u_s > tau
            pred, new_ids, new_centroids = discover(np.where(rej)[0], pred_base)
            evaluate(pred, np.isin(pred, new_ids), new_ids, name)
            # Full checkpoint for live inference on brand-new samples later: whitening
            # transform, all centroids (known + discovered), the DEAR head + its own
            # standardization stats, and the rejection threshold. Everything needed to
            # go from a raw feature vector to "known class X" or "novel -> cluster Y"
            # without re-running discovery.
            checkpoint = {
                'W': W if whiten else np.eye(feats_raw.shape[1]),
                'known_classes': list(known_classes),
                'centroids': np.concatenate([C1, new_centroids], axis=0) if len(new_ids) else C1,
                'new_cluster_ids': new_ids,
                'sn_map': sn_map,  # seen-novel true class name -> phase-1 free-centroid id
                'head_state': {k: v.cpu().clone() for k, v in head.state_dict().items()},
                'head_in_dim': feats.shape[1], 'head_n_classes': N1,
                'mu': mu, 'sig': sig, 'tau': float(tau),
                'radii': radii, 'NK': NK, 'N1': N1,
            }
        else:
            raise ValueError(f"unknown method {name}")

    return {'phase1': {'known_acc': p1_known_acc, 'seen_novel_hungarian_acc': float(sn_acc),
                       'novel_to_free_rate': frac_free},
            'phase2': results, 'checkpoint': checkpoint}

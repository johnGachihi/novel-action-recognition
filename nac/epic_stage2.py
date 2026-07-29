"""Stage 2 continual discovery, generalized to EPIC-KITCHENS-100's verb/noun
label spaces. Same protocol and same ablation switches as nac/stage2.py::
run_continual (see that module's docstring for what each switch guards
against); only the data loading/splitting is EPIC-specific (nac.epic_data,
grouped by video_id, instead of nac.data's UCF101/HMDB51 grouping).
"""
import json

import numpy as np
import torch
import torch.nn as nn

from .clustering import (discover_from_buffer, dpmeans_online, rejection_radii,
                         semisup_kmeans)
from .epic_data import known_three_way, novel_phase_split
from .evidential import LinearHead, make_edl_loss, predict_alpha, train_head, vacuity
from .geometry import dist2, fit_whitener
from .metrics import hungarian_acc, normalized_mutual_info_score

ALL_METHODS = ['closed_world', 'dist_reject', 'dpmeans', 'dear_reject', 'dear_weighted_reject', 'standard_classifier']


def compute_evidential_weights(alpha):
    """Computes uncertainty-guided weights from Dirichlet alpha parameters."""
    N, K = alpha.shape
    S = np.sum(alpha, axis=1, keepdims=True)  # (N, 1)
    
    # 1. Belief masses and Vacuity
    b = (alpha - 1) / S  # (N, K)
    u = K / S.squeeze(-1)  # (N,)
    
    # 2. Vectorized Dissonance Calculation
    sum_bi = np.sum(b, axis=1, keepdims=True)  # (N, 1)
    
    b_expanded_k = b[:, :, np.newaxis]  # (N, K, 1)
    b_expanded_j = b[:, np.newaxis, :]  # (N, 1, K)
    
    sum_pairs = b_expanded_k + b_expanded_j  # (N, K, K)
    diff_pairs = np.abs(b_expanded_k - b_expanded_j)  # (N, K, K)
    
    with np.errstate(divide='ignore', invalid='ignore'):
        bal = 1.0 - (diff_pairs / sum_pairs)
        bal[sum_pairs == 0] = 1.0
        
    weighted_bal = b_expanded_j * bal  # (N, K, K)
    for i in range(K):
        weighted_bal[:, i, i] = 0.0
        
    numerator_k = np.sum(weighted_bal, axis=2)  # (N, K)
    denominator_k = sum_bi - b  # (N, K)
    
    with np.errstate(divide='ignore', invalid='ignore'):
        diss_k = numerator_k / denominator_k
        diss_k[denominator_k == 0] = 0.0
        
    dissonance = np.sum(b * diss_k, axis=1)  # (N,)
    dissonance[sum_bi.squeeze(-1) == 0] = 0.0
    
    weights = u * (1.0 - dissonance)
    return weights


def run_epic_continual(label_space, methods=ALL_METHODS, seed=0, seen_ratio=40 / 75,
                       whiten=True, calib='heldout', calib_q=90, buffer_mode='alone',
                       merge='spacing', gate_epochs=50, p1_assignment='sinkhorn',
                       sinkhorn_eps=0.05, sinkhorn_col_weights='train_freq',
                       buffer_clusterer='kmeans', min_cluster_size=10,
                       features_path='features_epic.npz', split_path='class_split_epic.json',
                       verbose=True, feature_mode='multimodal'):
    """Defaults here diverge from nac/stage2.py's UCF101/HMDB51 defaults in one
    place: sinkhorn_eps=0.05 + sinkhorn_col_weights='train_freq' (vs eps=0.3,
    uniform there). UCF101/HMDB51 are curated to near-equal per-class sizes, so
    uniform Sinkhorn column marginals happen to be correct; EPIC verb/noun
    classes are severely long-tailed (observed counts 1 to ~9000), and uniform
    marginals at eps=0.3 were found to crater known-class phase-1 accuracy to
    ~chance (0.033 vs 0.029 chance on verb) while "fixing" anchored absorption
    by over-correcting. train_freq weighting (legitimate: derived from labeled
    training data, not from peeking at phase-1 pseudo-labels) plus a smaller,
    less-smoothed eps=0.05 recovers reasonable known accuracy (0.30 on verb)
    while still routing a meaningful fraction of novel samples to free
    centroids (0.20 on verb) -- eps=0.3 with this weighting still collapsed
    novel routing to ~0.01. Noun known-accuracy stays weak (~0.12-0.18) even
    after this fix, likely because frozen VideoMAE (pretrained for action/
    motion recognition) encodes verb-relevant dynamics far better than
    fine-grained object identity -- a real property of the features, not a
    bug in the assignment rule (verified: both greedy and reweighted-Sinkhorn
    give similarly weak noun known-accuracy)."""
    assert label_space in ('verb', 'noun')
    assert feature_mode in ('videomae', 'multimodal')
    d = np.load(features_path, allow_pickle=True)
    feats_raw = d['features']
    if feature_mode == 'videomae':
        feats_raw = feats_raw[:, :768]
    labels = d[f'{label_space}_class']
    keep = d[f'{label_space}_keep']
    video_ids = d['video_id']

    split = json.load(open(split_path))[label_space]
    known_classes = np.array(split['known'])
    novel_classes = np.array(split['novel'])
    NK = len(known_classes)
    known_to_idx = {c: i for i, c in enumerate(known_classes)}
    n_seen = round(len(novel_classes) * seen_ratio)
    n_unseen = len(novel_classes) - n_seen

    rng = np.random.default_rng(seed)
    train_idx, ho1, ho2 = known_three_way(known_classes, labels, video_ids, rng)
    _, sn1, sn2, un2 = novel_phase_split(novel_classes, labels, rng, n_seen)

    phase1_idx = np.concatenate([ho1, sn1])
    stream_idx = np.concatenate([ho2, sn2, un2])
    stream_type = np.array(['known'] * len(ho2) + ['seen_novel'] * len(sn2) + ['unseen_novel'] * len(un2))
    order = rng.permutation(len(stream_idx))
    stream_idx, stream_type = stream_idx[order], stream_type[order]
    if verbose:
        print(f"\n===== [EPIC:{label_space}] K={NK} seen-novel={n_seen} unseen-novel={n_unseen} | "
              f"train={len(train_idx)} phase1={len(phase1_idx)} stream={len(stream_idx)} =====")

    tr_labels = np.array([known_to_idx[c] for c in labels[train_idx]])
    if whiten:
        W, _ = fit_whitener(feats_raw[train_idx], tr_labels, NK)
        feats = feats_raw @ W
    else:
        feats = feats_raw

    # ---- phase 1: discovery ----
    X_lab, y_lab = feats[train_idx], tr_labels
    anchors0 = np.stack([X_lab[y_lab == k].mean(0) for k in range(NK)])
    C1, assign1 = semisup_kmeans(X_lab, y_lab, feats[phase1_idx], anchors0, n_free=n_seen,
                                 assignment=p1_assignment, sinkhorn_eps=sinkhorn_eps,
                                 sinkhorn_col_weights=sinkhorn_col_weights)
    N1 = len(C1)

    p1_true = labels[phase1_idx]
    p1_known = np.isin(p1_true, known_classes)
    p1_known_acc = float(np.mean(assign1[p1_known] == np.array([known_to_idx[c] for c in p1_true[p1_known]])))
    sn_acc, sn_map = hungarian_acc(p1_true[~p1_known], assign1[~p1_known], N1)
    frac_free = float(np.mean(assign1[~p1_known] >= NK))
    if verbose:
        print(f"[phase 1] known acc {p1_known_acc:.3f}  seen-novel Hungarian {sn_acc:.3f}  "
              f"novel->free rate {frac_free:.3f}")

    ho1_cls = np.array([known_to_idx[c] for c in labels[ho1]])
    radii = rejection_radii(C1, NK, feats, ho1, ho1_cls, feats[phase1_idx], assign1,
                            q=calib_q, calib=calib)

    # ---- phase 2: inference stream ----
    Xs = feats[stream_idx]
    s_true = labels[stream_idx]
    results = {}

    def evaluate(pred_cluster, is_new, new_ids, name):
        km = stream_type == 'known'
        sm = stream_type == 'seen_novel'
        um = stream_type == 'unseen_novel'
        r = {
            'known_acc': float(np.mean(pred_cluster[km] == np.array([known_to_idx[c] for c in s_true[km]]))),
            'seen_novel_acc': float(np.mean([sn_map.get(t, -1) == c
                                             for t, c in zip(s_true[sm], pred_cluster[sm])])),
            'unseen_detect_recall': float(is_new[um].mean()) if um.any() else float('nan'),
            'false_new_rate': float(is_new[km | sm].mean()),
            'n_new_categories': len(new_ids),
        }
        if is_new[um].sum() > 20 and len(new_ids) > 1:
            m = um & is_new
            acc, _ = hungarian_acc(s_true[m], pred_cluster[m] - N1, len(new_ids))
            r['new_cat_hungarian_acc'] = float(acc)
            r['new_cat_nmi'] = float(normalized_mutual_info_score(s_true[m], pred_cluster[m]))
            
        n_clusters_max = int(max(pred_cluster)) + 1 if len(pred_cluster) else 0
        overall_acc, _ = hungarian_acc(s_true, pred_cluster, max(n_clusters_max, 100))
        r['overall_hungarian_acc'] = float(overall_acc)
        results[name] = r
        if verbose:
            print(f"  {name:18s} " + "  ".join(f"{k}={v:.3f}" if isinstance(v, float) else f"{k}={v}"
                                               for k, v in r.items()))

    D2s = dist2(Xs, C1)
    pred_nearest = D2s.argmin(1)

    def discover(buffer_idx, pred_base, weights=None):
        out = pred_base.copy()
        new_centroids = None
        if buffer_mode == 'alone':
            out[buffer_idx], new_ids, new_centroids = discover_from_buffer(
                Xs[buffer_idx], C1, radii, n_unseen, merge=merge,
                clusterer=buffer_clusterer, min_cluster_size=min_cluster_size,
                weights=weights)
        elif buffer_mode == 'anchored':
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
    stream_predictions = {}
    history_stage2 = None
    for name in methods:
        if name == 'closed_world':
            pred = pred_nearest
            evaluate(pred, np.zeros(len(Xs), bool), [], name)
        elif name == 'dist_reject':
            rej = np.sqrt(D2s.min(1)) > radii[pred_nearest]
            pred, new_ids, _ = discover(np.where(rej)[0], pred_nearest)
            evaluate(pred, np.isin(pred, new_ids), new_ids, name)
        elif name == 'dpmeans':
            lam = np.percentile(radii, 90)
            pred, new_ids = dpmeans_online(Xs, C1, radii, lam)
            evaluate(pred, pred >= N1, new_ids, name)
        elif name == 'dear_reject':
            free_m = assign1 >= NK
            Xd = np.concatenate([feats[train_idx], feats[phase1_idx][free_m]])
            yd = np.concatenate([y_lab, assign1[free_m]])
            mu, sig = Xd.mean(0), Xd.std(0) + 1e-6
            loss = make_edl_loss(N1, total_epoch=gate_epochs)  # DEAR best config
            ckpt_path = "checkpoint.pt"
            
            # Prepare validation data (calibration set ho1)
            X_val_loss = (feats[ho1] - mu) / sig if len(ho1) else None
            y_val_loss = ho1_cls if len(ho1) else None
            
            head, history = train_head(
                (Xd - mu) / sig, yd, N1, loss, epochs=gate_epochs, seed=0, checkpoint_path=ckpt_path,
                X_val_loss=X_val_loss, y_val_loss=y_val_loss
            )
            history_stage2 = history

            u_ho = vacuity(predict_alpha(head, (feats[ho1] - mu) / sig), N1) if len(ho1) else np.array([], dtype=float)
            tau = float(np.percentile(u_ho, calib_q)) if u_ho.size else float('inf')
            alpha_s = predict_alpha(head, (Xs - mu) / sig)
            u_s = vacuity(alpha_s, N1)
            pred_base = alpha_s.argmax(1).cpu().numpy()
            rej = u_s > tau
            pred, new_ids, new_centroids = discover(np.where(rej)[0], pred_base)
            evaluate(pred, np.isin(pred, new_ids), new_ids, name)
            checkpoint = {
                'W': W if whiten else np.eye(feats_raw.shape[1]),
                'known_classes': list(known_classes),
                'centroids': np.concatenate([C1, new_centroids], axis=0) if len(new_ids) else C1,
                'new_cluster_ids': new_ids,
                'sn_map': sn_map,
                'head_state': {k: v.cpu().clone() for k, v in (head.module if isinstance(head, torch.nn.DataParallel) else head).state_dict().items()},
                'head_in_dim': feats.shape[1], 'head_n_classes': N1,
                'mu': mu, 'sig': sig, 'tau': float(tau),
                'radii': radii, 'NK': NK, 'N1': N1,
            }
        elif name == 'dear_weighted_reject':
            free_m = assign1 >= NK
            Xd = np.concatenate([feats[train_idx], feats[phase1_idx][free_m]])
            yd = np.concatenate([y_lab, assign1[free_m]])
            mu, sig = Xd.mean(0), Xd.std(0) + 1e-6
            loss = make_edl_loss(N1, total_epoch=gate_epochs)
            ckpt_path = "checkpoint.pt"
            
            X_val_loss = (feats[ho1] - mu) / sig if len(ho1) else None
            y_val_loss = ho1_cls if len(ho1) else None
            
            head, history = train_head(
                (Xd - mu) / sig, yd, N1, loss, epochs=gate_epochs, seed=0, checkpoint_path=ckpt_path,
                X_val_loss=X_val_loss, y_val_loss=y_val_loss
            )
            
            u_ho = vacuity(predict_alpha(head, (feats[ho1] - mu) / sig), N1) if len(ho1) else np.array([], dtype=float)
            tau = float(np.percentile(u_ho, calib_q)) if u_ho.size else float('inf')
            alpha_s = predict_alpha(head, (Xs - mu) / sig)
            u_s = vacuity(alpha_s, N1)
            pred_base = alpha_s.argmax(1).cpu().numpy()
            rej = u_s > tau
            
            alpha_np = alpha_s.cpu().numpy()
            weights = compute_evidential_weights(alpha_np)
            
            rejected_indices = np.where(rej)[0]
            pred, new_ids, new_centroids = discover(rejected_indices, pred_base, weights=weights[rejected_indices])
            evaluate(pred, np.isin(pred, new_ids), new_ids, name)
        elif name == 'standard_classifier':
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            loss_fn = lambda logits, target, epoch: nn.CrossEntropyLoss()(logits, target)
            ckpt_path = "checkpoint.pt"
            
            head = LinearHead(feats.shape[1], NK).to(device)
            head, history = train_head(
                feats[train_idx], y_lab, NK, loss_fn, epochs=gate_epochs, seed=0, checkpoint_path=ckpt_path,
                X_val_loss=feats[ho1] if len(ho1) else None, y_val_loss=ho1_cls if len(ho1) else None,
                device=device
            )
            history_stage2 = history
            
            head.eval()
            with torch.no_grad():
                logits = head(torch.tensor(Xs, dtype=torch.float32).to(device))
                pred = logits.argmax(1).cpu().numpy()
            
            evaluate(pred, np.zeros(len(Xs), bool), [], name)
        else:
            raise ValueError(f"unknown method {name}")
            
        stream_predictions[name] = pred.tolist()

    return {'phase1': {'known_acc': p1_known_acc, 'seen_novel_hungarian_acc': float(sn_acc),
                       'novel_to_free_rate': frac_free},
            'phase2': results, 'checkpoint': checkpoint,
            'stream_feats': Xs.tolist(),
            'stream_type': stream_type.tolist(),
            'stream_true_labels': s_true.tolist(),
            'stream_predictions': stream_predictions,
            'history': history_stage2}

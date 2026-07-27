"""Clustering machinery for Stage 2: k-means variants, rejection radii, and
inference-time category discovery from a rejected-sample buffer."""
import numpy as np

from .geometry import dist2


def kmeanspp_init(X, k, seed=0):
    r = np.random.default_rng(seed)
    cent = [X[r.integers(len(X))]]
    for _ in range(k - 1):
        d2 = np.min(dist2(X, np.stack(cent)), axis=1)
        p = d2 / d2.sum()
        cent.append(X[r.choice(len(X), p=p)])
    return np.stack(cent)


def plain_kmeans(X, k, iters=50, seed=1):
    C = kmeanspp_init(X, k, seed)
    for _ in range(iters):
        a = np.argmin(dist2(X, C), axis=1)
        for j in range(k):
            if (a == j).any():
                C[j] = X[a == j].mean(0)
            else:
                C[j] = X[np.argmax(np.min(dist2(X, C), axis=1))]
    return C, np.argmin(dist2(X, C), axis=1)


def sinkhorn_assign(D2, eps=0.05, iters=100, col_weights=None):
    """Balanced hard assignment via entropic optimal transport (Sinkhorn-Knopp,
    log-domain), as used for pseudo-labeling in UNO (Fini et al., ICCV 2021).
    Row marginals uniform over samples; column marginals uniform over centroids
    by default -- the equipartition constraint pushes mass toward otherwise-
    starved centroids. Cost is mean-normalized so eps is scale-free. Returns
    argmax assignments.

    col_weights (optional): relative column-mass targets instead of uniform.
    UCF101/HMDB51 are curated to near-equal per-class sizes, so uniform columns
    happen to be correct there; EPIC-KITCHENS verb/noun classes are severely
    long-tailed (observed counts from 1 to ~9000 in this project), and forcing
    equal mass onto every centroid regardless of true class size was found to
    crater known-class assignment accuracy (~random, 0.033 vs chance 0.029 on
    EPIC verb) -- col_weights lets the caller supply a legitimate non-uniform
    prior (e.g. labeled training-class frequency) instead."""
    C = D2 / (D2.mean() + 1e-12)
    n, k = C.shape
    logK = -C / eps
    f = np.zeros(n)
    g = np.zeros(k)
    logr = -np.log(n)
    logc = np.log(col_weights / col_weights.sum()) if col_weights is not None else np.full(k, -np.log(k))
    for _ in range(iters):
        M = logK + g[None, :]
        f = logr - np.log(np.exp(M - M.max(1, keepdims=True)).sum(1)) - M.max(1)
        M = logK + f[:, None]
        g = logc - np.log(np.exp(M - M.max(0, keepdims=True)).sum(0)) - M.max(0)
    return np.argmax(logK + g[None, :] + f[:, None], axis=1)


def semisup_kmeans(X_lab, y_lab, X_unl, anchored_centroids, n_free, iters=50, seed=0,
                   assignment='greedy', sinkhorn_eps=0.05, sinkhorn_col_weights='uniform'):
    """GCD-style: anchored centroids keep their labeled mass at every update; free
    centroids explain leftover unlabeled structure.

    assignment='greedy': nearest centroid (baseline). assignment='sinkhorn':
    balanced assignment via entropic OT — ablation targeting anchored absorption
    (greedy routes most novel samples into anchored centroids).

    sinkhorn_col_weights='uniform' (default, matches original UCF101/HMDB51
    behavior): equal column-mass target for every centroid. 'train_freq': anchored
    columns weighted by labeled training-class frequency (y_lab counts), free
    columns given the mean anchored weight -- needed for long-tailed label spaces
    (EPIC verb/noun) where uniform mass forces severe misassignment onto
    over-represented known classes (see sinkhorn_assign's docstring)."""
    n_anch = len(anchored_centroids)
    C = np.concatenate([anchored_centroids, kmeanspp_init(X_unl, n_free, seed)])
    lab_sums = np.zeros_like(anchored_centroids)
    lab_counts = np.zeros(n_anch)
    for k in range(n_anch):
        m = y_lab == k
        lab_sums[k] = X_lab[m].sum(0)
        lab_counts[k] = m.sum()

    col_weights = None
    if assignment == 'sinkhorn' and sinkhorn_col_weights == 'train_freq':
        col_weights = np.concatenate([lab_counts, np.full(n_free, lab_counts.mean())])

    def assign_fn(D2u):
        if assignment == 'greedy':
            return np.argmin(D2u, axis=1)
        if assignment == 'sinkhorn':
            return sinkhorn_assign(D2u, eps=sinkhorn_eps, col_weights=col_weights)
        raise ValueError(assignment)

    for _ in range(iters):
        assign = assign_fn(dist2(X_unl, C))
        newC = np.zeros_like(C)
        counts = np.zeros(len(C))
        np.add.at(newC, assign, X_unl)
        np.add.at(counts, assign, 1)
        newC[:n_anch] += lab_sums
        counts[:n_anch] += lab_counts
        empty = counts == 0
        if empty.any():  # re-seed dead free centroids at worst-explained points
            far = np.argsort(np.min(dist2(X_unl, C), axis=1))[::-1][:empty.sum()]
            newC[empty] = X_unl[far]
            counts[empty] = 1
        C = newC / counts[:, None]
    return C, assign_fn(dist2(X_unl, C))


def rejection_radii(C, n_known, feats, calib_known_idx, calib_known_cls,
                    phase1_feats, phase1_assign, q=90, calib='heldout'):
    """Per-centroid rejection radius at the q-th percentile.

    calib='heldout' (correct): anchored radii from held-out known samples of the
    centroid's class. calib='members' (ablation — the self-fulfilling variant):
    anchored radii from the samples assigned to the centroid, which by construction
    lie inside it and reject almost nothing.
    Free-centroid radii always come from their phase-1 members (best available).
    """
    radii = np.zeros(len(C))
    pool = []
    for k in range(n_known):
        if calib == 'heldout':
            m = calib_known_cls == k
            member = feats[calib_known_idx][m]
        elif calib == 'members':
            member = phase1_feats[phase1_assign == k]
        else:
            raise ValueError(calib)
        if len(member) >= 5:
            dd = np.linalg.norm(member - C[k], axis=1)
            radii[k] = np.percentile(dd, q)
            pool.append(dd)
    for k in range(n_known, len(C)):
        member = phase1_feats[phase1_assign == k]
        if len(member) >= 5:
            dd = np.linalg.norm(member - C[k], axis=1)
            radii[k] = np.percentile(dd, q)
            pool.append(dd)
    radii[radii == 0] = np.percentile(np.concatenate(pool), q)
    return radii


def discover_from_buffer(Xb, C_existing, radii, n_new, merge='spacing', seed=1,
                         clusterer='kmeans', min_cluster_size=10, hdbscan_pca=50):
    """Cluster a rejected buffer alone, then merge whole clusters back into existing
    centroids. merge='spacing' (correct): merge when the new centroid sits within
    half the nearest inter-centroid spacing — cluster MEANS are far closer to existing
    centroids than samples are (noise cancels), so sample radii are the wrong scale.
    merge='sample-radius' (ablation): the wrong-scale variant. merge='none': keep all.
    Samples of merged clusters are re-assigned individually by nearest centroid.

    clusterer='kmeans': k clusters with k = n_new (K oracle). clusterer='hdbscan':
    density-based — discovers the cluster count itself (n_new ignored) and marks
    low-density samples as noise; noise routes back to nearest existing centroid,
    absorbing buffer contamination instead of polluting minted categories.
    Returns (assignments with new ids starting at len(C_existing), new_id_list,
    new_centroids) -- new_centroids are the actual vectors for the surviving
    (non-merged) categories, needed to classify brand-new samples later without
    re-running discovery (e.g. for live inference against a saved checkpoint)."""
    N1 = len(C_existing)
    if clusterer == 'kmeans':
        Cn, asn = plain_kmeans(Xb, n_new, seed=seed)
    elif clusterer == 'hdbscan':
        from sklearn.cluster import HDBSCAN
        Xc = Xb
        if hdbscan_pca and Xb.shape[1] > hdbscan_pca and len(Xb) > hdbscan_pca:
            # density estimation degenerates in 768-d (distance concentration: the
            # whole buffer reads as noise); PCA-reduce for clustering only —
            # minted centroids are still computed in the full space below
            from sklearn.decomposition import PCA
            Xc = PCA(n_components=hdbscan_pca, random_state=0).fit_transform(Xb)
        asn = HDBSCAN(min_cluster_size=min_cluster_size).fit_predict(Xc)
        n_new = asn.max() + 1
        if n_new == 0:  # everything judged noise -> nothing minted
            return np.argmin(dist2(Xb, C_existing), axis=1), [], np.empty((0, Xb.shape[1]))
        Cn = np.stack([Xb[asn == j].mean(0) for j in range(n_new)])
    else:
        raise ValueError(clusterer)
    near = np.argmin(dist2(Cn, C_existing), axis=1)
    near_d = np.sqrt(np.min(dist2(Cn, C_existing), axis=1))
    if merge == 'spacing':
        cc = dist2(C_existing, C_existing)
        np.fill_diagonal(cc, np.inf)
        merged = near_d < 0.5 * np.sqrt(cc.min(1))[near]
    elif merge == 'sample-radius':
        merged = near_d < radii[near]
    elif merge == 'none':
        merged = np.zeros(n_new, bool)
    else:
        raise ValueError(merge)
    pred_out = np.empty(len(Xb), int)
    new_ids, next_id, remap = [], N1, {}
    for j in range(n_new):
        if merged[j]:
            remap[j] = -1
        else:
            remap[j] = next_id
            new_ids.append(next_id)
            next_id += 1
    nearest_existing = np.argmin(dist2(Xb, C_existing), axis=1)
    pred_out[asn == -1] = nearest_existing[asn == -1]  # hdbscan noise bucket
    for j in range(n_new):
        m = asn == j
        pred_out[m] = nearest_existing[m] if remap[j] == -1 else remap[j]
    new_centroids = np.stack([Cn[j] for j in range(n_new) if not merged[j]]) if new_ids else \
        np.empty((0, Xb.shape[1]))
    return pred_out, new_ids, new_centroids


def dpmeans_online(Xs, C_init, radii, lam, damp=10.0):
    """DP-means-style streaming: a sample farther than the radius of every existing
    centroid (lam for inference-minted ones) spawns a new centroid; minted centroids
    update as running means, pre-existing ones stay fixed (damped by their mass)."""
    N1 = len(C_init)
    C = list(C_init)
    counts = [damp] * N1
    pred = np.zeros(len(Xs), int)
    for t in range(len(Xs)):
        x = Xs[t]
        dd = np.linalg.norm(np.stack(C) - x, axis=1)
        j = dd.argmin()
        if dd[j] > (radii[j] if j < N1 else lam):
            C.append(x.copy())
            counts.append(1.0)
            pred[t] = len(C) - 1
        else:
            counts[j] += 1
            if j >= N1:
                C[j] = C[j] + (x - C[j]) / counts[j]
            pred[t] = j
    return pred, list(range(N1, len(C)))

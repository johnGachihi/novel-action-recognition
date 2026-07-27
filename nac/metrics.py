"""Evaluation metrics: Hungarian-matched clustering accuracy, NMI, AUROC wrappers."""
import numpy as np
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import normalized_mutual_info_score, roc_auc_score  # noqa: F401 (re-exported)


def hungarian_acc(true_labels, cluster_ids, n_clusters):
    """Optimal one-to-one class<->cluster matching (Kuhn-Munkres), then accuracy.
    Every sample in an unmatched cluster counts as wrong — this is what penalizes
    over-fragmentation (unlike purity/NMI). Returns (accuracy, class->cluster map)."""
    classes = np.unique(true_labels)
    cost = np.zeros((len(classes), n_clusters))
    for i, c in enumerate(classes):
        ids, cnt = np.unique(cluster_ids[true_labels == c], return_counts=True)
        cost[i, ids] = cnt
    ri, ci = linear_sum_assignment(-cost)
    mapping = dict(zip(classes[ri], ci))
    acc = np.mean([mapping.get(t, -1) == c for t, c in zip(true_labels, cluster_ids)])
    return acc, mapping

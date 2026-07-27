"""Whitening and distance helpers.

Whitening with the shared within-class covariance of the labeled train split turns
Euclidean distance into Mahalanobis distance. Do NOT L2-normalize afterwards: the
radial component in whitened space is exactly the novelty signal (Stage 1), and the
sphere projection erases it.
"""
import numpy as np


def fit_whitener(X_train, y_train, n_classes, ridge=1e-3):
    """Shared within-class covariance -> W = cov^(-1/2). Returns (W, class_means)."""
    means = np.stack([X_train[y_train == k].mean(0) for k in range(n_classes)])
    centered = X_train - means[y_train]
    cov = (centered.T @ centered) / len(centered)
    cov += ridge * np.trace(cov) / cov.shape[0] * np.eye(cov.shape[0])
    evals, evecs = np.linalg.eigh(cov)
    W = evecs @ np.diag(evals ** -0.5) @ evecs.T
    return W, means


def dist2(X, C):
    """Pairwise squared Euclidean distances, clipped at 0 against float roundoff."""
    return np.maximum((X ** 2).sum(1)[:, None] - 2 * X @ C.T + (C ** 2).sum(1)[None, :], 0)

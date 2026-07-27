"""Training-free Stage-1 detectors: cosine-to-prototype and Mahalanobis."""
import numpy as np


def cosine_prototype(features, train_idx, eval_idx, class_labels, class_names):
    fn = features / np.linalg.norm(features, axis=1, keepdims=True)
    protos = np.stack([fn[train_idx][class_labels[train_idx] == c].mean(0) for c in class_names])
    protos /= np.linalg.norm(protos, axis=1, keepdims=True)
    sims = fn[eval_idx] @ protos.T
    return 1 - sims.max(1), sims.argmax(1)          # (novelty score, class pred)


def mahalanobis(features, train_idx, eval_idx, class_labels, class_names, ridge=1e-3):
    means = np.stack([features[train_idx][class_labels[train_idx] == c].mean(0) for c in class_names])
    centered = np.concatenate([features[train_idx][class_labels[train_idx] == c] - means[i]
                               for i, c in enumerate(class_names)])
    cov = (centered.T @ centered) / len(centered)
    cov += ridge * np.trace(cov) / cov.shape[0] * np.eye(cov.shape[0])
    ci = np.linalg.inv(cov)
    X = features[eval_idx]
    d = (np.einsum('nd,de,ne->n', X, ci, X)[:, None]
         - 2 * (X @ ci @ means.T)
         + np.einsum('cd,de,ce->c', means, ci, means)[None, :])
    return d.min(1), d.argmin(1)

# Uncertainty-Weighted Clustering: Technical Design & Implementation Guide

This document outlines the mathematical framework and step-by-step implementation guide for **Uncertainty-Weighted Clustering** using Evidential Deep Learning (EDL) predictions.

---

## 1. Mathematical Framework

Using Subjective Logic and Evidential Deep Learning, the model outputs Dirichlet distribution parameters $\boldsymbol{\alpha}_i = [\alpha_{i1}, \dots, \alpha_{iK}]^T$ for each sample $i$. 

We define the Dirichlet strength as:
$$S_i = \sum_{k=1}^K \alpha_{ik}$$

From $S_i$ and $\boldsymbol{\alpha}_i$, we calculate:
1. **Belief Masses**: $b_{ik} = \frac{\alpha_{ik} - 1}{S_i}$
2. **Vacuity (Epistemic Uncertainty)**: $u_i = \frac{K}{S_i}$

### Evidential Dissonance (Aleatoric Uncertainty)
Dissonance measures the amount of conflicting evidence between classes. For a belief mass vector $\mathbf{b}_i = [b_{i1}, \dots, b_{iK}]^T$, the dissonance is:

$$d_i = \sum_{k=1}^K \frac{b_{ik} \sum_{j \neq k} b_{ij} \text{bal}(b_{ik}, b_{ij})}{\sum_{j \neq k} b_{ij}}$$

Where the balance function $\text{bal}(b_{ik}, b_{ij})$ measures the relative agreement between two beliefs:

$$\text{bal}(b_{ik}, b_{ij}) = \begin{cases} 
1 - \frac{|b_{ik} - b_{ij}|}{b_{ik} + b_{ij}} & \text{if } b_{ik} + b_{ij} > 0 \\ 
1 & \text{otherwise} 
\end{cases}$$

### Sample Weight Calculation
We calculate a reliability weight $w_i \in [0, 1]$ for each rejected sample $i$:
$$w_i = u_i \cdot (1 - d_i)$$

## Full $w_i$ expression: 

$$
w_i = \frac{K}{\sum_{k=1}^K \alpha_{ik}} \left( 1 - \sum_{k=1}^K \frac{b_{ik} \sum_{j \neq k} b_{ij} \text{bal}(b_{ik}, b_{ij})}{\sum_{j \neq k} b_{ij}} \right)
$$

---

## 2. Python Implementation: Dissonance & Weighting

Here is a clean, vectorized NumPy implementation to calculate vacuity, dissonance, and weights from the Dirichlet parameters $\boldsymbol{\alpha}$:

```python
import numpy as np

def compute_evidential_weights(alpha):
    """
    Computes uncertainty-guided weights from Dirichlet alpha parameters.
    
    Args:
        alpha: np.ndarray of shape (N, K) where K is the number of known classes.
        
    Returns:
        weights: np.ndarray of shape (N,) containing values in [0, 1].
    """
    N, K = alpha.shape
    S = np.sum(alpha, axis=1, keepdims=True)  # (N, 1)
    
    # 1. Belief masses and Vacuity
    b = (alpha - 1) / S  # (N, K)
    u = K / S.squeeze(-1)  # (N,)
    
    # 2. Vectorized Dissonance Calculation
    dissonance = np.zeros(N)
    for i in range(N):
        bi = b[i]
        sum_bi = np.sum(bi)
        if sum_bi == 0:
            dissonance[i] = 0
            continue
            
        # Outer sum and difference of belief masses
        bi_col = bi[:, np.newaxis]
        bi_row = bi[np.newaxis, :]
        sum_pairs = bi_col + bi_row
        diff_pairs = np.abs(bi_col - bi_row)
        
        # bal(b_k, b_j)
        with np.errstate(divide='ignore', invalid='ignore'):
            bal = 1.0 - (diff_pairs / sum_pairs)
            bal[sum_pairs == 0] = 1.0
            
        # Compute weights for the double sum: b_j * bal(b_k, b_j)
        weighted_bal = bi_row * bal  # (K, K)
        
        # Exclude self-dissonance (j != k)
        np.fill_diagonal(weighted_bal, 0.0)
        
        # Sum over j
        numerator_k = np.sum(weighted_bal, axis=1)  # (K,)
        denominator_k = sum_bi - bi  # (K,)
        
        # Calculate class-conditional dissonance
        with np.errstate(divide='ignore', invalid='ignore'):
            diss_k = numerator_k / denominator_k
            diss_k[denominator_k == 0] = 0.0
            
        dissonance[i] = np.sum(bi * diss_k)
        
    # 3. Compute final sample weights
    weights = u * (1.0 - dissonance)
    return weights
```

---

## 3. Weighted K-Means Implementation

Below is a custom NumPy implementation of **Weighted K-Means** that uses these weights:

```python
class WeightedKMeans:
    def __init__(self, n_clusters, max_iter=300, tol=1e-4, random_state=None):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = random_state
        self.centroids = None

    def fit(self, X, weights):
        """
        Fits Weighted K-Means.
        
        Args:
            X: np.ndarray of shape (N, D)
            weights: np.ndarray of shape (N,) containing sample weights
        """
        rng = np.random.default_rng(self.random_state)
        N, D = X.shape
        
        # Normalize weights to sum to N (or prevent scale issues)
        w = np.clip(weights, 1e-6, 1.0)
        
        # Initialize centroids randomly choosing from X weighted by w
        p = w / np.sum(w)
        idx = rng.choice(N, size=self.n_clusters, replace=False, p=p)
        self.centroids = X[idx].copy()
        
        for iteration in range(self.max_iter):
            # E-step: Assign each sample to the nearest centroid
            # Compute pairwise distances (N, n_clusters)
            dists = np.linalg.norm(X[:, np.newaxis, :] - self.centroids[np.newaxis, :, :], axis=2)
            labels = np.argmin(dists, axis=1)
            
            # M-step: Update centroids as weighted mean of assigned points
            new_centroids = np.zeros_like(self.centroids)
            for k in range(self.n_clusters):
                mask = (labels == k)
                if np.sum(mask) == 0:
                    # Handle empty cluster: reinitialize with a random point
                    new_centroids[k] = X[rng.choice(N)]
                else:
                    cluster_X = X[mask]
                    cluster_w = w[mask][:, np.newaxis]
                    new_centroids[k] = np.sum(cluster_X * cluster_w, axis=0) / np.sum(cluster_w)
            
            # Convergence check
            shift = np.sum(np.linalg.norm(self.centroids - new_centroids, axis=1))
            self.centroids = new_centroids.copy()
            if shift < self.tol:
                break
                
        return self

    def predict(self, X):
        dists = np.linalg.norm(X[:, np.newaxis, :] - self.centroids[np.newaxis, :, :], axis=2)
        return np.argmin(dists, axis=1)
```

---

## 4. Integration Blueprint

To integrate this into the current pipeline:

1. **Calculate Weights**:
   In [epic_stage2.py](file:///home/beria/Documents/novel-action-recognition/nac/epic_stage2.py), after collecting the Dirichlet parameters $\boldsymbol{\alpha}$ for the stream, pass them to `compute_evidential_weights(alpha)` to get the weight vector $w$.

2. **Cluster Selected Samples**:
   Filter both features and weights to include only rejected samples:
   ```python
   rejected_mask = (vacuity_scores > tau)
   rejected_feats = stream_feats[rejected_mask]
   rejected_weights = weights[rejected_mask]
   ```

3. **Run Weighted Clustering**:
   Instead of calling standard `KMeans` inside [clustering.py](file:///home/beria/Documents/novel-action-recognition/nac/clustering.py), invoke `WeightedKMeans` passing `rejected_feats` and `rejected_weights`.

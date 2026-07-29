# Experimental Setup and Hyperparameters

This document details the configuration, architectures, hyperparameter settings, and loss formulations used across the various stages of the Novel Action Recognition pipeline.

---

## 1. Feature Extraction (Stage 0)
The pipeline supports extracting features from either raw videos, pre-clipped narration clips, or raw frame directories.

| Component | Configuration / Backbone | Parameters & Settings |
|---|---|---|
| **Video Backbone** | `MCG-NJU/videomae-base` (Frozen) | Uniform 16-frame sampling, spatial size 224x224. |
| **Video Feature Dim** | 768 | Mean pooled across sequence dimension: `last_hidden_state.mean(dim=1)`. |
| **Audio Backbone** | `MIT/ast-finetuned-audioset-10-10-0.4593` (Frozen) | Input length: 10s. Target sampling rate: 16,000 Hz. |
| **Audio Feature Dim** | 768 | Base audio spectrogram transformer outputs. |
| **Fused Feature Dim** | 1536 | Concatenated VideoMAE (768) + AST (768) features (in multimodal mode). |
| **Batch Size** | 128 | Single-GPU execution (optimized to avoid OOM). |
| **Data Loader Workers** | Dynamically scaled | Set to `min(4, os.cpu_count() or 2)` to match CPU cores. |

---

## 2. Stage 1: Novelty Detection & Evidential Deep Learning (EDL)
In Stage 1, a linear classification head is trained on known classes, and evidential vacuity uncertainty is used as the anomaly score for novelty detection.

### Evidential Head Architecture
* **Classification Head**: Single linear layer mapping the input feature space (768 for VideoMAE-only, 1536 for Multimodal) to the number of known classes $K$.

### Optimization & Training Hyperparameters
* **Optimizer**: Adam
* **Learning Rate (LR)**: `1e-3`
* **Weight Decay**: `1e-4`
* **Batch Size**: 256
* **Training Epochs**: 75
* **Device**: CUDA (with auto-fallback to CPU, supports `DataParallel` for multiple GPUs)

### Loss Formulations & Ablation Setups
The pipeline compares multiple evidential configurations as defined in `nac/stage1.py`:

| Method Name | Loss Form | Evidence Function | KL Divergence Reg. | AvU / EUC Calibration | Annealing Setup |
|---|---|---|---|---|---|
| **`sensoy_tuned`** | MSE | Capped Softplus (`max=10.0`) | **Yes** | No | Linear ramp to `lambda_ceiling=0.02` (step 40) |
| **`dear`** | Log | Exponential (`exp`) | No | **Yes** | Exponential: `0.01 -> 1.0` over 75 epochs |
| **`dear_noreg`** | Log | Exponential (`exp`) | No | No | N/A |
| **`dear_kl`** | Log | Exponential (`exp`) | **Yes** | No | Exponential: `0.01 -> 1.0` |
| **`cosine`** | Baseline | N/A | N/A | N/A | Cosine similarity prototype matching |
| **`mahalanobis`** | Baseline | N/A | N/A | N/A | Mahalanobis distance to class distributions |

---

## 3. Stage 2: Continual Category Discovery
Stage 2 evaluates the model's ability to cluster unseen categories from a stream of video features while maintaining performance on known categories.

### Phase 1: Semi-Supervised Discovery & Centroid Initialization
* **Algorithm**: Semi-Supervised K-Means with Sinkhorn assignment.
* **Sinkhorn Regularization (`sinkhorn_eps`)**: `0.05`
* **Sinkhorn Column Weights (`sinkhorn_col_weights`)**: `'train_freq'` (weighted based on labeled training data frequency to handle long-tailed distributions like EPIC-KITCHENS).
* **Number of Free Clusters**: Set equal to the number of seen-novel categories ($N_{seen}$).

### Phase 2: Inference Stream & Online Clustering
* **Calibration Set**: Held-out known validation set (`ho1`).
* **Calibration Percentile (`calib_q`)**: 90th percentile of vacuity uncertainty.
* **Uncertainty Threshold ($\tau$)**: Set dynamically based on the 90th percentile value of the calibration set.
* **Rejection Radius Method**: `rejection_radii` based on class distances on the calibration set.
* **Buffer Clustering Mode**: `'alone'` (clusters rejected samples independently of anchors).
* **Min Cluster Size**: 10
* **DP-Means Online Lambda ($\lambda$)**: Set to the 90th percentile of the rejection radii.

### Evidential Weighting Metrics (DEAR Weighted Reject)
For the weighted reject formulation, cluster assignment weights are calculated dynamically using Dirichlet parameters $\alpha$:
* **Vacuity Uncertainty**: $u = \frac{K}{S}$ where $S = \sum_{k=1}^K \alpha_k$.
* **Dissonance Uncertainty**: Measures conflicting evidence across different classes:
  $$d(\mathbf{b}) = \sum_{i=1}^K \frac{b_i \sum_{j \neq i} b_j bal(b_i, b_j)}{\sum_{j \neq i} b_j}$$
  where $bal(b_i, b_j) = 1 - \frac{|b_i - b_j|}{b_i + b_j}$ represents the relative balance between belief masses.
* **Final Cluster Weights**: $W = u \times (1.0 - d)$.

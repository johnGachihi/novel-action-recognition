# Novel action recognition / continual category discovery

Frozen-VideoMAE pipeline on UCF101+HMDB51 with an even/odd known/novel class split.
Exploration and results live in `notebook_cleaned.ipynb`; the modular implementation
lives in `nac/` with two CLI entry points. Requires `features_videomae.npz`
(extracted by notebook Section 5).

## Layout

```
nac/data.py        feature loading, subset masks, group-aware + continual splits
nac/geometry.py    within-class covariance whitening, distances
nac/evidential.py  linear head, evidence fns, Sensoy/DEAR losses (all components switchable)
nac/baselines.py   cosine-prototype, Mahalanobis
nac/clustering.py  kmeans++/plain/semi-supervised k-means, DP-means, radii, buffer discovery
nac/metrics.py     Hungarian accuracy, NMI, AUROC
nac/stage1.py      novelty-detection benchmark (notebook Sections 6/8/9)
nac/stage2.py      continual discovery benchmark (notebook Section 10)
```

## Usage

```bash
# Stage 1: all methods, combined + UCF101-only + HMDB51-only
python3 run_stage1.py --out stage1_results.json

# Stage 1 ablation: DEAR loss components (any make_edl_loss kwarg)
python3 run_stage1.py --methods dear --override with_avuloss=False
python3 run_stage1.py --methods sensoy_tuned --override lambda_ceiling=0.05

# Stage 2: all methods/protocols
python3 run_stage2.py --out stage2_results.json

# Stage 2 component ablations (defaults are the correct variants;
# the alternatives reproduce the failure modes documented in Section 10)
python3 run_stage2.py --no-whiten            # raw feature geometry
python3 run_stage2.py --calib members        # self-fulfilling rejection radii
python3 run_stage2.py --buffer-mode anchored # anchors re-absorb the rejected buffer
python3 run_stage2.py --merge none           # keep every minted cluster
python3 run_stage2.py --calib-q 80           # stricter rejection operating point

# Phase-1 discovery ablation: UNO-style Sinkhorn-balanced assignment
# (fixes anchored absorption: novel->free 0.11 -> 0.64, DEAR-gated unseen recall
#  0.26 -> 0.43 on combined, at ~0.05 known-accuracy cost; eps=0.3 from sweep)
python3 run_stage2.py --p1-assign sinkhorn --sinkhorn-eps 0.3

# Buffer-clusterer ablation: HDBSCAN instead of k-means(K oracle).
# NEGATIVE RESULT in this feature space: small min_cluster_size mints near-duplicate
# micro-clusters (~100 cats for 35 classes, recall ~0.14); large calls the whole
# buffer noise (0 cats); PCA-50 + mid sizes yields 2-3 amorphous blobs (NMI ~0).
# Classes here are diffuse overlapping ellipsoids — centroid methods fit, density
# methods don't. The K oracle should be replaced via probe-class estimation instead.
python3 run_stage2.py --buffer-clusterer hdbscan --min-cluster-size 5

# Happy (Ma et al., NeurIPS 2024) ported onto frozen features: nac/happy.py + run_happy.py.
# Without a projector: classifier-side components (clustering-guided init, group-wise
# entropy reg, hardness-aware replay) verified individually correct, but combined
# they collapse (old acc -> 0.000) once cluster-init gives new heads real directions.
#
# Restoring representation-learning capacity (extract_view2.py -> a real second view
# per video via TSN-style temporal jitter + flip, plus a trainable Projector with
# two-view contrastive losses + feature-level KD) initially looked WORSE, not
# better -- but this turned out to be 3 real bugs, not a structural limitation
# (found after the user correctly pushed back on an "added capacity makes it worse"
# result): (1) train_stage0's CE loss was missing temperature scaling on the
# cosine-similarity classifier logits, silently crippling stage-0 training;
# (2) classifier heads were warm-started from RAW feature class-means but evaluated
# through the (random-init) projector from step 0, an inconsistent starting point;
# (3) the projector's residual wasn't an EXACT identity at init (default last-layer
# init gave cosine~0.97, not 1.0) -- fixed via zero-initializing the last layer,
# plus lowering LR to 0.001 for any projector-enabled run (0.01 diverges even from a
# perfect init). After all 3 fixes: full+projector substantially beats the
# no-projector version for stages 1-4 (old acc 0.73->0.36 vs a flat 0.000), i.e. the
# projector genuinely helps once correctly implemented. One open issue remains: a
# sharp, undiagnosed collapse specifically at stage 5 -- not yet root-caused.
# nac/stage2.py (centroid-based, DEAR-gated) is still the recommended production
# method (higher and more reliable numbers, far less complexity), but this Happy
# line is NOT a dead end the way it first looked -- see project memory for the full
# bug list before trusting any further results here at face value.
python3 run_happy.py --subset combined --stages 5
```

Reproducibility: Stage-2 numbers are exactly reproducible (numpy clustering,
fixed seeds). Stage-1 evidential heads vary by ~±0.004 AUROC across runs from GPU
matmul nondeterminism; training-free methods are exact.

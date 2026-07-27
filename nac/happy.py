"""Happy-adapted: continual generalized category discovery with a trainable projector
on top of frozen VideoMAE features.

Ported from Ma et al., "Happy: A Debiased Learning Framework for Continual
Generalized Category Discovery" (NeurIPS 2024). First version (see git history /
earlier notes) dropped every encoder-side component because the VideoMAE backbone
is fully frozen -- that version was diagnosed as BISTABLE (collapses to all-new or
all-old) once cluster-guided init gave new heads real directions, because nothing
was left to stabilize the classifier the way the paper's encoder-side losses do.

This version restores that stabilization cheaply: instead of fine-tuning the 86M-
parameter VideoMAE backbone (expensive -- would mean re-running the encoder every
step), a small trainable MLP Projector sits between the frozen features and the
classifier. Only the projector (+ classifier heads) trains; the backbone stays
frozen and is queried once (two cached views per video: features_videomae.npz and
features_videomae_view2.npz, the latter using TSN-style temporal jitter + random
flip -- see extract_view2.py). This mirrors the paper's own convention of only
fine-tuning the last transformer block, just implemented as an added module instead
of unfreezing part of the backbone.

  KEPT (now genuinely applicable)  L^l_con, L^u_con (Eq. 1): supervised contrastive
       on stage-0 labeled two-view projector outputs; unsupervised (SimCLR-style)
       contrastive on continual-stage two-view projector outputs. There are now two
       real, differently-augmented views per video, and a trainable module to learn
       from them.
  KEPT (now genuinely applicable)  L_kd (Eq. 11): the projector's output CAN drift
       between stages now (it's trained every stage), so distilling against a frozen
       snapshot of last stage's projector is real work, not a no-op.
  KEPT  Clustering-guided classifier initialization (Eq. 3), group-wise soft entropy
       regularization (Eq. 4-6), hardness-aware Gaussian prototype replay (Eq. 8-10)
       -- unchanged in mechanism, now operating on projector(feature) rather than
       raw whitened features directly.
  ADAPTED  Self-training / self-distillation (Eq. 7): still single-view pseudo-label
       sharpening for the classification loss itself (the paper's two-view version
       is for the classifier prediction, separate from the representation-learning
       contrastive losses above, which now do have two real views to use).
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from nac.clustering import dist2, kmeanspp_init


# ---------------------------------------------------------------------------
# Multi-stage benchmark construction
# ---------------------------------------------------------------------------

def build_multistage_splits(known_classes, novel_classes, class_labels, rng, T=5,
                            test_frac=0.15, old_per_class=30):
    """Stage-0 labeled known classes; novel classes split evenly across T stages.
    Each class gets one held-out test reserve (used at every stage from its
    introduction onward, matching Happy's protocol of testing on all classes seen
    so far); everything else is available to be consumed as train/discovery data.
    Returns: stage0_idx, stage0_labels(dict class->int), stages: list of dicts with
    'new_classes', 'unlabeled_idx' (new-class + old-background samples),
    'unlabeled_true' (class name per sample), 'test_idx', 'test_true'."""
    all_classes = list(known_classes) + list(novel_classes)
    test_pool, train_pool = {}, {}
    for c in all_classes:
        idx = np.where(class_labels == c)[0]
        rng.shuffle(idx)
        n_test = max(1, int(test_frac * len(idx)))
        test_pool[c] = idx[:n_test]
        train_pool[c] = list(idx[n_test:])

    # Known classes must keep a reserve so they can reappear as unlabeled "old"
    # background in later stages -- draining the whole pool into stage-0 would make
    # every continual stage's unlabeled batch 100% new-class, giving the classifier
    # no old-class signal at all to preserve (silently defeats the entire point of
    # C-GCD's "unlabeled data mixes old and new" setting). Use ~40% for stage-0
    # supervised init, reserve the rest for background draws across stages.
    stage0_labels = {c: i for i, c in enumerate(known_classes)}
    stage0_parts = []
    for c in known_classes:
        n_stage0 = max(1, int(0.4 * len(train_pool[c])))
        stage0_parts.append(train_pool[c][:n_stage0])
        train_pool[c] = train_pool[c][n_stage0:]  # remainder = background reserve
    stage0_idx = np.concatenate(stage0_parts)

    perm = rng.permutation(len(novel_classes))
    chunks = np.array_split(perm, T)
    stage_new_classes = [[novel_classes[i] for i in chunk] for chunk in chunks]

    stages = []
    old_so_far = list(known_classes)
    cumulative_test_classes = list(known_classes)
    for t in range(T):
        new_classes = stage_new_classes[t]
        unl_idx, unl_true = [], []
        for c in new_classes:  # all remaining samples of newly-introduced classes
            unl_idx += train_pool[c]
            unl_true += [c] * len(train_pool[c])
            train_pool[c] = []
        for c in old_so_far:  # a background slice of already-known classes
            take = train_pool[c][:old_per_class]
            train_pool[c] = train_pool[c][old_per_class:]
            unl_idx += take
            unl_true += [c] * len(take)
        cumulative_test_classes = cumulative_test_classes + new_classes
        test_idx = np.concatenate([test_pool[c] for c in cumulative_test_classes])
        test_true = np.concatenate([[c] * len(test_pool[c]) for c in cumulative_test_classes])
        stages.append({
            'new_classes': new_classes,
            'unlabeled_idx': np.array(unl_idx),
            'unlabeled_true': np.array(unl_true),
            'test_idx': test_idx,
            'test_true': test_true,
        })
        old_so_far = old_so_far + new_classes
    return stage0_idx, stage0_labels, stages


# ---------------------------------------------------------------------------
# Trainable projector (the piece that makes L^l_con, L^u_con, L_kd real again)
# ---------------------------------------------------------------------------

class Projector(nn.Module):
    """Small MLP standing in for "fine-tune the last transformer block": everything
    upstream (VideoMAE) is frozen and queried once; this is the only thing that
    trains from representation-learning signal.

    hidden=0 collapses to a single Linear(in_dim, in_dim) (no nonlinearity -- tests
    whether ANY hidden capacity is needed at all). hidden>0 with n_layers=1 is the
    original Linear-ReLU-Linear bottleneck/expansion MLP; n_layers>1 stacks
    additional ReLU-Linear hidden blocks for more depth.

    residual=True (default): forward is z + net(z), and the last layer is
    zero-initialized so this is an EXACT identity at t=0 (default init only gave
    cosine(z,proj(z))~0.97, confirmed sufficient on its own to destabilize joint
    training -- see git history). residual=False: forward is net(z) alone, plain
    feedforward, standard init -- zero-initting the last layer here would make
    EVERY input map to the same zero vector at t=0 (no residual to fall back on),
    a degenerate start with nothing to break the symmetry between classes, so the
    zero-init trick is deliberately skipped in this branch."""
    def __init__(self, in_dim, hidden=512, n_layers=1, residual=True):
        super().__init__()
        self.residual = residual
        if hidden == 0:
            layers = [nn.Linear(in_dim, in_dim)]
        else:
            layers = [nn.Linear(in_dim, hidden), nn.ReLU(inplace=True)]
            for _ in range(n_layers - 1):
                layers += [nn.Linear(hidden, hidden), nn.ReLU(inplace=True)]
            layers += [nn.Linear(hidden, in_dim)]
        self.net = nn.Sequential(*layers)
        if residual:
            nn.init.zeros_(self.net[-1].weight)
            nn.init.zeros_(self.net[-1].bias)

    def forward(self, z):
        return z + self.net(z) if self.residual else self.net(z)


def supervised_contrastive_loss(z1, z2, labels, tau=0.1):
    """Eq. 1's L^l_con: two-view SupCon on stage-0 labeled data. z1, z2: (B, d)
    projector outputs for view A / view B of the same batch; labels: (B,) int."""
    z = F.normalize(torch.cat([z1, z2], dim=0), dim=1)
    y = torch.cat([labels, labels], dim=0)
    n = z.shape[0]
    sim = (z @ z.T) / tau
    sim.fill_diagonal_(-1e9)
    same = (y.unsqueeze(0) == y.unsqueeze(1)) & ~torch.eye(n, dtype=torch.bool, device=z.device)
    log_prob = sim - torch.logsumexp(sim, dim=1, keepdim=True)
    pos_counts = same.sum(1).clamp_min(1)
    loss = -(log_prob * same).sum(1) / pos_counts
    return loss[pos_counts > 0].mean()


def unsupervised_contrastive_loss(z1, z2, tau=0.1):
    """Eq. 1's L^u_con: SimCLR-style two-view InfoNCE, no labels needed -- each
    instance's two views are the only positive pair."""
    n = z1.shape[0]
    z = F.normalize(torch.cat([z1, z2], dim=0), dim=1)
    sim = (z @ z.T) / tau
    sim.fill_diagonal_(-1e9)
    targets = torch.cat([torch.arange(n, 2 * n), torch.arange(0, n)]).to(z.device)
    return F.cross_entropy(sim, targets)


def projector_kd_loss(proj, proj_prev, z):
    """Eq. 11, now meaningful: penalize the current projector's output drifting from
    a frozen snapshot of last stage's projector, on the current stage's features."""
    with torch.no_grad():
        target = proj_prev(z)
    current = proj(z)
    return (1 - F.cosine_similarity(current, target, dim=1)).mean()


# ---------------------------------------------------------------------------
# Growing parametric classifier (Sec 3.1 notation: g_phi, L2-normalized, no bias)
# ---------------------------------------------------------------------------

class GrowingClassifier(nn.Module):
    def __init__(self, in_dim, n_init):
        super().__init__()
        self.in_dim = in_dim
        self.weight = nn.Parameter(torch.randn(n_init, in_dim) * 0.01)

    @property
    def n_classes(self):
        return self.weight.shape[0]

    def forward(self, z):
        z = F.normalize(z, dim=1)
        w = F.normalize(self.weight, dim=1)
        return z @ w.T  # cosine logits; paper applies softmax(.../tau)

    def grow(self, new_weight):
        """Append new class heads (Eq. 3 supplies new_weight via cluster centroids)."""
        with torch.no_grad():
            new_weight = new_weight.to(self.weight.device)
            self.weight = nn.Parameter(torch.cat([self.weight.data, new_weight], dim=0))


def cluster_guided_init(Z_unlabeled, old_heads, n_new, seed=0):
    """Eq. 3 exactly: k-means with K=n_old+n_new clusters on the unlabeled batch;
    for each candidate centroid compute max cosine similarity to the EXISTING OLD
    HEADS (not to other centroids); the n_new centroids least similar to any old
    head become the new heads. old_heads: (n_old, d) current classifier weights.

    Returns (new_head_weights, is_novel). is_novel is a boolean array, one entry
    per row of Z_unlabeled: True where that sample's k-means cluster was one of
    the n_new selected as novel, False where its cluster matched an old head
    instead (discarded, not given a new head). This is a per-CLUSTER decision
    applied to every member -- a cluster with mixed true-old/true-novel
    membership gets one answer for all of them, so is_novel can and does
    disagree with ground truth for individual samples; use it to measure that
    contamination directly (compare against the real old/new label), not to
    assume cluster purity."""
    # k-means itself is numpy/CPU (nac.clustering); GPU tensors round-trip through
    # .cpu() here and the result is moved back to old_heads' device by clf.grow().
    n_old = old_heads.shape[0]
    Zn = F.normalize(Z_unlabeled, dim=1).cpu().numpy()
    C = kmeanspp_init(Zn, n_old + n_new, seed)
    for _ in range(50):
        a = np.argmin(dist2(Zn, C), axis=1)
        for j in range(len(C)):
            if (a == j).any():
                C[j] = Zn[a == j].mean(0)
    Cn = C / (np.linalg.norm(C, axis=1, keepdims=True) + 1e-12)
    old_n = F.normalize(old_heads, dim=1).cpu().numpy()
    sim_to_old = Cn @ old_n.T          # (K, n_old)
    max_sim_old = sim_to_old.max(axis=1)
    new_ids = np.argsort(max_sim_old)[:n_new]  # least-similar-to-any-OLD-head centroids
    is_novel = np.isin(a, new_ids)     # a: final per-sample cluster assignment from the loop above
    return torch.tensor(Cn[new_ids], dtype=torch.float32), is_novel


# ---------------------------------------------------------------------------
# Group-wise soft entropy regularization (Eq. 4-6): alternative to Sinkhorn for the
# same old-class-absorption ("prediction bias") pathology.
# ---------------------------------------------------------------------------

def group_entropy_reg(probs, n_old, n_new):
    """probs: (B, K) softmax over ALL classes seen so far (old ∪ new-this-stage)."""
    pbar = probs.mean(0)
    p_old = pbar[:n_old].sum().clamp_min(1e-12)
    p_new = pbar[n_old:].sum().clamp_min(1e-12)
    l_old_new = p_old * p_old.log() + p_new * p_new.log()
    p_old_c = pbar[:n_old].clamp_min(1e-12)
    p_new_c = pbar[n_old:].clamp_min(1e-12)
    l_old_in = (p_old_c * p_old_c.log()).sum()
    l_new_in = (p_new_c * p_new_c.log()).sum()
    return l_old_new + l_old_in + l_new_in


# ---------------------------------------------------------------------------
# Hardness-aware Gaussian prototype replay (Eq. 8-10)
# ---------------------------------------------------------------------------

def update_prototypes(Z, pseudo_labels, n_classes, shared_var=None):
    """Eq. 8: class-wise mean/cov (here: shared isotropic variance r^2, the
    'empirically works fine' simplification the paper itself adopts)."""
    d = Z.shape[1]
    mu = torch.zeros(n_classes, d, device=Z.device)
    counts = torch.zeros(n_classes, device=Z.device)
    for c in range(n_classes):
        m = pseudo_labels == c
        if m.sum() > 0:
            mu[c] = Z[m].mean(0)
            counts[c] = m.sum()
    if shared_var is None:
        var_sum, n = 0.0, 0
        for c in range(n_classes):
            m = pseudo_labels == c
            if m.sum() > 1:
                var_sum += ((Z[m] - mu[c]) ** 2).sum().item()
                n += m.sum().item() * d
        shared_var = var_sum / max(n, 1)
    return mu, shared_var, counts


def hardness_distribution(mu, n_old, tau_h=0.1):
    """Eq. 9: harder (more confusable with other old classes) -> sampled more."""
    mun = F.normalize(mu[:n_old], dim=1)
    sim = mun @ mun.T
    sim.fill_diagonal_(0)
    h = sim.sum(1) / max(n_old - 1, 1)
    return F.softmax(h / tau_h, dim=0)


def sample_replay(mu, shared_var, n_old, n_samples, tau_h=0.1, seed=None):
    p_hard = hardness_distribution(mu, n_old, tau_h)
    g = torch.Generator(device=mu.device).manual_seed(seed) if seed is not None else None
    classes = torch.multinomial(p_hard, n_samples, replacement=True, generator=g)
    noise = torch.randn(n_samples, mu.shape[1], generator=g, device=mu.device) * (shared_var ** 0.5)
    return mu[classes] + noise, classes


# ---------------------------------------------------------------------------
# Training loops: stage-0 (labeled, two views, real gradient training) and each
# continual stage (unlabeled, self-training + optional projector/replay/KD).
# ---------------------------------------------------------------------------

def train_stage0(clf, projector, Z0_v1, Z0_v2, y0, epochs=30, lr=0.01,
                 lambda_con=1.0, tau_con=0.1, tau_p=0.1, seed=0):
    """Paper Sec 4.1: stage-0 trains on labeled data with CE + supervised
    contrastive (Eq. 1's L^l_con) using two augmented views. Classifier heads are
    still warm-started from class means (done by the caller) before this refines
    both classifier and projector jointly."""
    torch.manual_seed(seed)
    opt = torch.optim.SGD(list(clf.parameters()) + list(projector.parameters()), lr=lr, momentum=0.9)
    n = Z0_v1.shape[0]
    for ep in range(epochs):
        perm = torch.randperm(n)
        for i in range(0, n, 256):
            idx = perm[i:i + 256]
            zb1, zb2, yb = Z0_v1[idx], Z0_v2[idx], y0[idx]
            p1 = projector(zb1)
            # clf() returns raw cosine similarity in [-1,1] -- MUST divide by the
            # same temperature every other classifier-loss call in this file uses
            # (train_stage's lines use /tau_p, /tau_t throughout), else cross_entropy
            # sees a nearly-uniform distribution over n_old classes and trains almost
            # nothing regardless of epoch count. (Bug found 2026-07-26: this line was
            # the one call site missing it, silently crippling stage-0 training.)
            loss = F.cross_entropy(clf(p1) / tau_p, yb)
            loss = loss + lambda_con * supervised_contrastive_loss(p1, projector(zb2), yb, tau=tau_con)
            opt.zero_grad()
            loss.backward()
            opt.step()
    return clf, projector


def train_stage(clf, Z_unlabeled, n_old, n_new, use_entropy_reg, use_hardness_replay,
                mu=None, shared_var=None, epochs=30, tau_p=0.1, tau_t=0.05,
                lr=0.01, lambda_ent=1.0, lambda_hard=1.0, seed=0,
                projector=None, Z_view2=None, prev_projector=None,
                lambda_con=1.0, lambda_kd=1.0, tau_con=0.1):
    """Z_unlabeled is always view A, raw (pre-projector, pre-whitening already
    applied by the caller). If projector is given, classifier operates on
    projector(Z_unlabeled) and, if Z_view2 is given, an unsupervised contrastive
    loss (Eq. 1's L^u_con) trains the projector on the two views; if prev_projector
    is also given, a KD loss (Eq. 11) penalizes drift from last stage's projector.
    mu/shared_var (replay prototypes) always live in RAW pre-projector space -- they
    don't drift, so re-estimating them in a moving projected space isn't needed;
    replay samples are projected fresh at use time, same path as real samples."""
    torch.manual_seed(seed)
    params = list(clf.parameters())
    if projector is not None:
        params = params + list(projector.parameters())
    opt = torch.optim.SGD(params, lr=lr, momentum=0.9)
    n = Z_unlabeled.shape[0]
    K = clf.n_classes
    for ep in range(epochs):
        perm = torch.randperm(n)
        for i in range(0, n, 256):
            idx = perm[i:i + 256]
            zb_raw = Z_unlabeled[idx]
            zb = projector(zb_raw) if projector is not None else zb_raw
            logits = clf(zb)
            p = F.softmax(logits / tau_p, dim=1)
            with torch.no_grad():
                q = F.softmax(logits / tau_t, dim=1)  # sharpened self-target (Eq. 7, single-view)
            loss = -(q * p.clamp_min(1e-12).log()).sum(1).mean()
            if use_entropy_reg:
                loss = loss + lambda_ent * group_entropy_reg(p, n_old, K - n_old)
            if use_hardness_replay and mu is not None:
                zr, yr = sample_replay(mu, shared_var, n_old, n_samples=len(zb_raw))
                zr_proj = projector(zr) if projector is not None else zr
                loss = loss + lambda_hard * F.cross_entropy(clf(zr_proj) / tau_p, yr)
            if projector is not None and Z_view2 is not None:
                zb2 = projector(Z_view2[idx])
                loss = loss + lambda_con * unsupervised_contrastive_loss(zb, zb2, tau=tau_con)
            if projector is not None and prev_projector is not None:
                loss = loss + lambda_kd * projector_kd_loss(projector, prev_projector, zb_raw)
            opt.zero_grad()
            loss.backward()
            opt.step()
    return clf

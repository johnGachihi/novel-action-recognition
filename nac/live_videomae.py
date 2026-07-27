"""Live VideoMAE fine-tuning: on-the-fly two-view augmentation + an actually
UNFROZEN last transformer block, as opposed to nac/happy.py's frozen-feature +
small-added-Projector approximation.

Rationale: the Projector experiment approximates "fine-tune the last block" (Happy's
own convention) with a cheap added MLP on cached features, to avoid re-running
VideoMAE per step. This module does the real thing instead -- raw video frames are
decoded and augmented fresh every epoch (not cached), and gradients flow through
VideoMAE's actual last encoder layer + final layernorm. Much more expensive per
step (a full 12-layer ViT forward pass every batch, for every sample, both views),
so intended for a smaller-scale run than the frozen-feature experiments.

Two views per sample, each independently resampled every __getitem__ call (unlike
extract_view2.py's cache-once-to-disk approach):
  view A: TSN-style random-segment temporal sampling (a random frame within each of
          16 equal segments), no flip.
  view B: same temporal scheme, independent random draw, + random horizontal flip.
Both views differ from each other AND change every epoch (real stochastic
augmentation), unlike the frozen pipeline's two FIXED cached views.
"""
import glob
import os

import cv2
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset

NUM_FRAMES = 16


def decode_all_frames(path):
    """Single SEQUENTIAL decode pass, not per-frame cap.set(POS_FRAMES) seeks.
    OpenCV seeking on compressed video (H.264/AVI) typically has to decode forward
    from the last keyframe, so N random seeks costs far more than one sequential
    read -- confirmed as the dominant bottleneck here (GPU util 9-42%, CPU load
    3-4 of 24 cores: neither compute-bound, both starved waiting on seek latency)."""
    cap = cv2.VideoCapture(str(path))
    frames = []
    ok = True
    while ok:
        ok, frame = cap.read()
        if ok:
            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    cap.release()
    if not frames:
        frames = [np.zeros((224, 224, 3), np.uint8)]
    return frames


def pick_frames(all_frames, n=NUM_FRAMES, flip=False, rng=None):
    """TSN-style random-segment sampling from an already-decoded frame list."""
    rng = rng or np.random.default_rng()
    total = len(all_frames)
    bounds = np.linspace(0, total, n + 1)
    idxs = [rng.integers(int(bounds[i]), max(int(bounds[i + 1]), int(bounds[i]) + 1)) for i in range(n)]
    frames = [all_frames[min(i, total - 1)] for i in idxs]
    if flip:
        frames = [np.ascontiguousarray(f[:, ::-1, :]) for f in frames]
    return frames


def build_manifest(ucf_root='data/ucf101/videos/UCF-101', hmdb_root='data/hmdb51/extracted/hmdb51'):
    """(path, class_name) pairs, class_name e.g. 'UCF101:Archery' -- matches the
    naming used throughout the rest of this project (features_videomae.npz etc)."""
    records = []
    for c in sorted(os.listdir(ucf_root)):
        for v in glob.glob(f'{ucf_root}/{c}/*.avi'):
            records.append((v, f'UCF101:{c}'))
    for c in sorted(os.listdir(hmdb_root)):
        for v in glob.glob(f'{hmdb_root}/{c}/*.avi'):
            records.append((v, f'HMDB51:{c}'))
    return records


class TwoViewVideoDataset(Dataset):
    """paths: list of video file paths. Returns (pixel_values_a, pixel_values_b,
    index) -- ALREADY processed into model-ready tensors, not raw frame lists.

    This is the fix for the dominant bottleneck found by direct timing: for a
    batch of 32, VideoMAEImageProcessor alone took 1.31s (single-threaded, run
    once per batch in the main process) vs 0.90s for the model's own forward+
    backward -- the GPU was idle for the *larger* share of every step. Running
    the processor here, per-sample, inside __getitem__ means it executes in the
    parallel DataLoader worker pool and overlaps with GPU compute on the previous
    batch via normal prefetching, instead of blocking the main process serially."""
    def __init__(self, paths, processor, seed=0):
        self.paths = paths
        self.processor = processor
        self.seed = seed

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        rng = np.random.default_rng()  # fresh randomness every call -> real per-epoch augmentation
        path = self.paths[idx]
        try:
            all_frames = decode_all_frames(path)  # ONE sequential decode; both views drawn from it
            fa = pick_frames(all_frames, flip=False, rng=rng)
            fb = pick_frames(all_frames, flip=bool(rng.random() < 0.5), rng=rng)
            pa = self.processor(fa, return_tensors="pt")['pixel_values'][0]
            pb = self.processor(fb, return_tensors="pt")['pixel_values'][0]
            return pa, pb, idx
        except Exception:
            return None


def collate_two_view(batch):
    batch = [b for b in batch if b is not None]
    if not batch:
        return None, None, []
    pa, pb, idxs = zip(*batch)
    return torch.stack(pa), torch.stack(pb), list(idxs)


class PartiallyFrozenVideoMAE(nn.Module):
    """VideoMAE-base with everything frozen except the last encoder layer + final
    layernorm (Happy's own "fine-tune last block" convention). forward() returns
    the mean-pooled embedding, matching how features_videomae.npz was built
    (last_hidden_state.mean(dim=1)) -- so a class-mean warm-start computed from a
    quick frozen pass is directly comparable to what this model produces at t=0."""
    def __init__(self, model_name='MCG-NJU/videomae-base'):
        super().__init__()
        from transformers import VideoMAEModel
        self.backbone = VideoMAEModel.from_pretrained(model_name)
        for p in self.backbone.parameters():
            p.requires_grad_(False)
        for p in self.backbone.encoder.layer[-1].parameters():
            p.requires_grad_(True)
        for p in self.backbone.layernorm.parameters():
            p.requires_grad_(True)

    def trainable_parameters(self):
        return [p for p in self.parameters() if p.requires_grad]

    def forward(self, pixel_values):
        """Manually replicates VideoMAEModel's embeddings -> 12 encoder layers ->
        final layernorm, so the FROZEN prefix (embeddings + layers 0-10) can run
        under torch.no_grad() -- eval mode, no graph, no saved activations -- and
        only the trainable suffix (layer 11 + layernorm) builds an autograd graph.
        Without this split, autograd tracks and stores activations through all 12
        layers regardless of which parameters require grad, which is most of why
        GPU utilization and memory were so low relative to available headroom."""
        enc = self.backbone.encoder
        was_training = self.training
        with torch.no_grad():
            self.backbone.eval()  # frozen prefix always runs as eval (no dropout drift)
            hidden = self.backbone.embeddings(pixel_values, None)
            for layer in enc.layer[:-1]:
                hidden = layer(hidden, None, False)[0]
            hidden = hidden.detach()  # belt-and-suspenders: no graph leaks from the frozen prefix
        if was_training:
            enc.layer[-1].train()
            self.backbone.layernorm.train()
        hidden = enc.layer[-1](hidden, None, False)[0]
        hidden = self.backbone.layernorm(hidden)
        return hidden.mean(dim=1)


def frames_to_pixel_values(processor, frame_lists, device):
    inputs = processor(frame_lists, return_tensors="pt")
    return inputs['pixel_values'].to(device)

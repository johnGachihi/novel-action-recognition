#!/usr/bin/env python3
"""One-time frozen VideoMAE feature extraction for the EPIC-KITCHENS-100 clip
subset, producing features_epic.npz in the same spirit as features_videomae.npz
(UCF101/HMDB51): a single deterministic view per clip (uniform 16-frame
sampling, no augmentation), frozen VideoMAE-base, last_hidden_state.mean(dim=1).

Reuses nac.live_videomae.decode_all_frames (one sequential decode pass, not
per-frame seeks -- confirmed dominant-bottleneck fix from the earlier live
fine-tuning work) and moves VideoMAEImageProcessor into Dataset.__getitem__ so
it runs in parallel DataLoader workers rather than serially in the main process
(the other confirmed GPU-utilization fix from that work).

Checkpoints every CHECKPOINT_EVERY samples to features_epic.npz.partial so a
crash (this project has seen one unexplained background-job kill before) loses
at most one checkpoint interval, and resumes by skipping already-embedded
narration_ids on restart.
"""
import glob
import os
import time

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import VideoMAEImageProcessor, VideoMAEModel, ASTFeatureExtractor, ASTModel

from nac.epic_data import load_epic_manifest

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
NUM_FRAMES = 16
BATCH_SIZE = 32
NUM_WORKERS = 12
CHECKPOINT_EVERY = 5000
OUT_PATH = 'features_epic.npz'
PARTIAL_PATH = 'features_epic.npz.partial.npz'


def decode_frames_uniform(path, n=NUM_FRAMES):
    """Single sequential decode pass (fast on compressed video -- no per-frame
    seeks), but keeps only the n frames we actually need instead of materializing
    the whole clip. nac.live_videomae.decode_all_frames stores every frame and is
    fine for UCF101/HMDB51's uniformly short clips, but EPIC's per-narration clips
    can run minutes long (up to 297s / ~14,900 frames observed in this subset --
    one such clip fully decoded is ~18GB of raw RGB) and blowing that up across
    12 parallel workers is exactly what OOM-killed the first extraction run."""
    cap = cv2.VideoCapture(str(path))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total <= 0:  # unreliable metadata fallback: decode with a hard cap
        frames, cap_n = [], 4000
        ok = True
        while ok and len(frames) < cap_n:
            ok, frame = cap.read()
            if ok:
                frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        cap.release()
        if not frames:
            return [np.zeros((224, 224, 3), np.uint8)] * n
        idxs = np.linspace(0, len(frames) - 1, n).round().astype(int)
        return [frames[i] for i in idxs]

    target_idxs = np.linspace(0, total - 1, n).round().astype(int)
    target_set = set(target_idxs.tolist())
    max_idx = int(target_idxs.max())
    picked = {}
    i = 0
    while i <= max_idx:
        ok, frame = cap.read()
        if not ok:
            break
        if i in target_set:
            picked[i] = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        i += 1
    cap.release()
    if not picked:
        return [np.zeros((224, 224, 3), np.uint8)] * n
    last = picked[max(picked)]
    return [picked.get(idx, last) for idx in target_idxs]


def decode_audio_ffmpeg(path, target_sr=16000):
    target_length = target_sr * 10
    try:
        import tempfile
        import subprocess
        import scipy.io.wavfile as wavfile
        with tempfile.NamedTemporaryFile(suffix=".wav") as temp_wav:
            cmd = [
                'ffmpeg', '-y', '-i', str(path),
                '-vn', '-acodec', 'pcm_s16le', '-ar', str(target_sr),
                '-ac', '1', temp_wav.name
            ]
            subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
            sr, y = wavfile.read(temp_wav.name)
            y = y.astype(np.float32) / 32768.0
    except Exception:
        y = np.zeros(target_length, dtype=np.float32)
    
    if len(y) < target_length:
        y = np.pad(y, (0, target_length - len(y)))
    else:
        y = y[:target_length]
    return y


def decode_audio(path, target_sr=16000):
    target_length = target_sr * 10
    try:
        import torchaudio
        waveform, sr = torchaudio.load(path)
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)
        if sr != target_sr:
            waveform = torchaudio.functional.resample(waveform, sr, target_sr)
        y = waveform.squeeze().numpy()
    except Exception:
        return decode_audio_ffmpeg(path, target_sr)
    
    if len(y) < target_length:
        y = np.pad(y, (0, target_length - len(y)))
    else:
        y = y[:target_length]
    return y


class EpicFeatureDataset(Dataset):
    def __init__(self, paths, processor):
        self.paths = paths
        self.processor = processor

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        try:
            path = self.paths[idx]
            picked = decode_frames_uniform(path)
            pv = self.processor(picked, return_tensors="pt")['pixel_values'][0]
            aw = decode_audio(path)
            return pv, aw, idx
        except Exception:
            return None


def collate(batch):
    batch = [b for b in batch if b is not None]
    if not batch:
        return None, None, []
    pv, aw, idxs = zip(*batch)
    return torch.stack(pv), torch.tensor(np.array(aw), dtype=torch.float32), list(idxs)


def main():
    df = load_epic_manifest(min_count=20)
    print(f"manifest: {len(df)} clips ({df.verb_keep.sum()} verb-kept, {df.noun_keep.sum()} noun-kept)")

    # Smoke-test cap: honour EPIC_LIMIT env var set by run_end_to_end.sh --limit N
    limit = int(os.environ.get('EPIC_LIMIT', 0))
    if limit > 0:
        df = df.head(limit).copy()
        print(f"EPIC_LIMIT={limit}: capped manifest to {len(df)} narrations for smoke-test.")

    done_ids = set()
    feats_done, narr_done = [], []
    if os.path.exists(PARTIAL_PATH):
        d = np.load(PARTIAL_PATH, allow_pickle=True)
        done_ids = set(d['narration_id'].tolist())
        feats_done = [d['features']]
        narr_done = [d['narration_id']]
        print(f"resuming: {len(done_ids)} clips already embedded")

    remaining = df[~df.narration_id.isin(done_ids)].reset_index(drop=True)
    print(f"remaining to embed: {len(remaining)}")
    if len(remaining) == 0:
        print("nothing to do, already complete")
        merge_and_save(df, feats_done, narr_done)
        return

    processor = VideoMAEImageProcessor.from_pretrained('MCG-NJU/videomae-base')
    model = VideoMAEModel.from_pretrained('MCG-NJU/videomae-base').to(DEVICE).eval()
    for p in model.parameters():
        p.requires_grad_(False)

    ast_extractor = ASTFeatureExtractor.from_pretrained('MIT/ast-finetuned-audioset-10-10-0.45')
    ast_model = ASTModel.from_pretrained('MIT/ast-finetuned-audioset-10-10-0.45').to(DEVICE).eval()
    for p in ast_model.parameters():
        p.requires_grad_(False)

    ds = EpicFeatureDataset(remaining.path.tolist(), processor)
    loader = DataLoader(ds, batch_size=BATCH_SIZE, collate_fn=collate,
                        num_workers=NUM_WORKERS, shuffle=False, pin_memory=True)

    new_feats, new_narr = [], []
    t0 = time.time()
    n_done = 0
    with torch.no_grad():
        for pv, aw, idxs in loader:
            if pv is None or aw is None:
                continue
            pv = pv.to(DEVICE, non_blocking=True)
            video_out = model(pixel_values=pv).last_hidden_state.mean(dim=1)
            
            aw_list = [w.numpy() for w in aw]
            audio_inputs = ast_extractor(aw_list, sampling_rate=16000, return_tensors="pt")
            audio_in = audio_inputs['input_values'].to(DEVICE)
            audio_out = ast_model(audio_in).last_hidden_state.mean(dim=1)
            
            fused_out = torch.cat([video_out, audio_out], dim=-1)
            new_feats.append(fused_out.cpu().numpy())
            new_narr.append(remaining.narration_id.values[idxs])
            n_done += len(idxs)
            if n_done % (CHECKPOINT_EVERY // BATCH_SIZE * BATCH_SIZE) < BATCH_SIZE:
                rate = n_done / (time.time() - t0)
                print(f"  {n_done}/{len(remaining)} embedded ({rate:.1f} clips/sec)", flush=True)
                checkpoint(feats_done + new_feats, narr_done + new_narr)

    checkpoint(feats_done + new_feats, narr_done + new_narr)
    merge_and_save(df, feats_done + new_feats, narr_done + new_narr)


def checkpoint(feats_parts, narr_parts):
    np.savez(PARTIAL_PATH, features=np.concatenate(feats_parts),
             narration_id=np.concatenate(narr_parts))


def merge_and_save(df, feats_parts, narr_parts):
    features = np.concatenate(feats_parts)
    narration_id = np.concatenate(narr_parts)
    order = {n: i for i, n in enumerate(narration_id)}
    idx = [order[n] for n in df.narration_id]
    features = features[idx]
    np.savez(OUT_PATH,
            features=features,
            paths=df.path.values,
            narration_id=df.narration_id.values,
            video_id=df.video_id.values,
            participant_id=df.participant_id.values,
            verb_class=df.verb_class.values,
            noun_class=df.noun_class.values,
            verb_keep=df.verb_keep.values,
            noun_keep=df.noun_keep.values)
    print(f"saved {OUT_PATH}: {features.shape}")


if __name__ == '__main__':
    main()

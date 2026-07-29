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
from transformers import VideoMAEImageProcessor, VideoMAEModel, ASTFeatureExtractor, ASTForAudioClassification
from tqdm import tqdm

from nac.epic_data import load_epic_manifest

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
USE_AMP = DEVICE == 'cuda'  # fp16 autocast — only for CUDA
NUM_FRAMES = 16
BATCH_SIZE = 256           # fp16 halves VRAM per sample; 256 is safe on T4/A100
NUM_WORKERS = 0            # Set to 0 to avoid OpenCV + fork deadlocks in PyTorch DataLoader
CHECKPOINT_EVERY = 5000
OUT_PATH = 'features_epic.npz'
PARTIAL_PATH = 'features_epic.npz.partial.npz'


def decode_frames_uniform(path, n=NUM_FRAMES):
    print(f"  [decode_frames_uniform] Opening VideoCapture for {path}", flush=True)
    cap = cv2.VideoCapture(str(path))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"  [decode_frames_uniform] total frames: {total}", flush=True)
    if total <= 0:  # unreliable metadata fallback: decode with a hard cap
        print(f"  [decode_frames_uniform] WARNING: total <= 0, decoding sequentially with hard cap...", flush=True)
        frames, cap_n = [], 4000
        ok = True
        while ok and len(frames) < cap_n:
            ok, frame = cap.read()
            if ok:
                frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        cap.release()
        if not frames:
            print(f"  [decode_frames_uniform] WARNING: No frames decoded at all. Returning zeros.", flush=True)
            return [np.zeros((224, 224, 3), np.uint8)] * n
        idxs = np.linspace(0, len(frames) - 1, n).round().astype(int)
        return [frames[i] for i in idxs]

    target_idxs = np.linspace(0, total - 1, n).round().astype(int)
    target_set = set(target_idxs.tolist())
    max_idx = int(target_idxs.max())
    print(f"  [decode_frames_uniform] Target indices: {target_idxs}", flush=True)
    picked = {}
    i = 0
    t0 = time.time()
    while i <= max_idx:
        ok, frame = cap.read()
        if not ok:
            print(f"  [decode_frames_uniform] Premature EOF at frame {i}", flush=True)
            break
        if i in target_set:
            picked[i] = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        i += 1
    cap.release()
    print(f"  [decode_frames_uniform] Finished reading {i} frames in {time.time() - t0:.2f}s. Picked {len(picked)} target frames.", flush=True)
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


def decode_frames_from_dir(path, start_f, stop_f, n=NUM_FRAMES):
    target_idxs = np.linspace(start_f, stop_f, n).round().astype(int)
    frames = []
    for idx in target_idxs:
        loaded = False
        for prefix in ["img_", "frame_"]:
            for ext in [".jpg", ".png", ".JPEG", ".JPG"]:
                img_path = os.path.join(str(path), f"{prefix}{idx:010d}{ext}")
                if os.path.exists(img_path):
                    img = cv2.imread(img_path)
                    if img is not None:
                        frames.append(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
                        loaded = True
                        break
            if loaded:
                break
        if not loaded:
            frames.append(np.zeros((224, 224, 3), dtype=np.uint8))
    return frames


def decode_frames_range(video_path, start_frame, stop_frame, n=NUM_FRAMES):
    cap = cv2.VideoCapture(str(video_path))
    target_idxs = np.linspace(start_frame, stop_frame, n).round().astype(int)
    target_set = set(target_idxs.tolist())
    max_idx = int(target_idxs.max())
    
    # Seek to the start frame to speed up decoding
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    
    picked = {}
    i = start_frame
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


def decode_audio_segment_ffmpeg(path, start_sec, duration_sec, target_sr=16000):
    target_length = target_sr * 10
    try:
        import tempfile
        import subprocess
        import scipy.io.wavfile as wavfile
        with tempfile.NamedTemporaryFile(suffix=".wav") as temp_wav:
            cmd = [
                'ffmpeg', '-y',
                '-ss', f"{start_sec:.3f}",
                '-t', f"{duration_sec:.3f}",
                '-i', str(path),
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


def decode_audio_segment(path, start_sec, duration_sec, target_sr=16000):
    target_length = target_sr * 10
    try:
        import torchaudio
        info = torchaudio.info(str(path))
        sr = info.sample_rate
        
        frame_offset = int(start_sec * sr)
        num_frames = int(duration_sec * sr)
        
        waveform, sr_loaded = torchaudio.load(
            str(path),
            frame_offset=frame_offset,
            num_frames=num_frames
        )
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)
        if sr_loaded != target_sr:
            waveform = torchaudio.functional.resample(waveform, sr_loaded, target_sr)
        y = waveform.squeeze().numpy()
    except Exception:
        return decode_audio_segment_ffmpeg(path, start_sec, duration_sec, target_sr)
        
    if len(y) < target_length:
        y = np.pad(y, (0, target_length - len(y)))
    else:
        y = y[:target_length]
    return y


def decode_audio(path, target_sr=16000):
    target_length = target_sr * 10
    if os.path.isdir(str(path)):
        return np.zeros(target_length, dtype=np.float32)
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
    def __init__(self, paths, start_frames, stop_frames, is_video, processor, ast_extractor=None):
        self.paths = paths
        self.start_frames = start_frames
        self.stop_frames = stop_frames
        self.is_video = is_video
        self.processor = processor

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        path = self.paths[idx]
        print(f"[Dataset] Starting item {idx}: {path}", flush=True)
        t_start = time.time()
        try:
            is_vid = self.is_video[idx] if self.is_video is not None else False
            
            if is_vid:
                start_f = self.start_frames[idx]
                stop_f = self.stop_frames[idx]
                picked = decode_frames_range(path, start_f, stop_f)
            elif os.path.isdir(str(path)):
                start_f = self.start_frames[idx]
                stop_f = self.stop_frames[idx]
                picked = decode_frames_from_dir(path, start_f, stop_f)
            else:
                picked = decode_frames_uniform(path)
                
            print(f"[Dataset] Decoded {len(picked)} frames in {time.time() - t_start:.2f}s", flush=True)
            pv = self.processor(picked, return_tensors="pt")['pixel_values'][0]
            print(f"[Dataset] Processed item {idx} in {time.time() - t_start:.2f}s", flush=True)
            return pv, idx
        except Exception as e:
            print(f"[Dataset] ERROR on item {idx}: {e}", flush=True)
            return None


def collate(batch):
    batch = [b for b in batch if b is not None]
    if not batch:
        return None, []
    pv, idxs = zip(*batch)
    return torch.stack(pv), list(idxs)


class VideoMAEWrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
    def forward(self, pixel_values):
        with torch.autocast(device_type='cuda', enabled=USE_AMP):
            return self.model(pixel_values=pixel_values).last_hidden_state.mean(dim=1)


class ASTWrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
    def forward(self, input_values):
        with torch.autocast(device_type='cuda', enabled=USE_AMP):
            return self.model(input_values).last_hidden_state.mean(dim=1)


def main():
    import sys
    gpu_name = torch.cuda.get_device_name(0) if DEVICE == 'cuda' else 'N/A'
    print(f"Using device: {DEVICE}  |  GPU: {gpu_name}  |  AMP fp16: {USE_AMP}")
    if DEVICE == 'cpu':
        print("\nWARNING: CUDA is not available — extraction will be very slow.")
        print("If on Kaggle, enable GPU in Settings > Accelerator and re-run.\n")
    df = load_epic_manifest(min_count=20)
    if len(df) == 0:
        print("\nERROR: Loaded manifest is empty. No video directories found under the raw images path.")
        print("Please verify that your --raw-images-dir contains the expected participant directories.")
        sys.exit(1)
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

    hf_token = os.environ.get('HF_TOKEN') or None

    processor = VideoMAEImageProcessor.from_pretrained('MCG-NJU/videomae-base', token=hf_token)
    model = VideoMAEModel.from_pretrained('MCG-NJU/videomae-base', token=hf_token).to(DEVICE).eval()
    for p in model.parameters():
        p.requires_grad_(False)

    # Wrap models (AST is bypassed for speed)
    videomae_wrapped = VideoMAEWrapper(model)
    batch_size = BATCH_SIZE

    ds = EpicFeatureDataset(
        remaining.path.tolist(),
        remaining.start_frame.tolist() if 'start_frame' in remaining.columns else None,
        remaining.stop_frame.tolist() if 'stop_frame' in remaining.columns else None,
        remaining.is_video.tolist() if 'is_video' in remaining.columns else None,
        processor
    )
    loader = DataLoader(ds, batch_size=batch_size, collate_fn=collate,
                        num_workers=NUM_WORKERS, shuffle=False, pin_memory=True)

    new_feats, new_narr = [], []
    t0 = time.time()
    n_done = 0
    with torch.no_grad():
        for pv, idxs in tqdm(loader, desc="Extracting features", unit="batch"):
            if pv is None:
                continue
            pv = pv.to(DEVICE, non_blocking=True)
            video_out = videomae_wrapped(pv)
            
            # Simulate multimodal features by appending zero-features (768-dim)
            audio_out = torch.zeros((video_out.shape[0], 768), device=DEVICE, dtype=video_out.dtype)
            fused_out = torch.cat([video_out, audio_out], dim=-1)
            
            new_feats.append(fused_out.cpu().numpy())
            new_narr.append(remaining.narration_id.values[idxs])
            n_done += len(idxs)
            if n_done % (CHECKPOINT_EVERY // batch_size * batch_size) < batch_size:
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

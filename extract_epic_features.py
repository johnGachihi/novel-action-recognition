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
NUM_FRAMES = 16
BATCH_SIZE = 512
NUM_WORKERS = min(4, os.cpu_count() or 2)
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


def decode_audio_segment(path, start_sec, duration_sec, target_sr=16000):
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
    def __init__(self, paths, start_frames, stop_frames, is_video, processor, ast_extractor):
        self.paths = paths
        self.start_frames = start_frames
        self.stop_frames = stop_frames
        self.is_video = is_video
        self.processor = processor
        self.ast_extractor = ast_extractor

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        try:
            path = self.paths[idx]
            is_vid = self.is_video[idx] if self.is_video is not None else False
            
            if is_vid:
                start_f = self.start_frames[idx]
                stop_f = self.stop_frames[idx]
                picked = decode_frames_range(path, start_f, stop_f)
                
                # Decode segment audio to prevent OOM on full video files
                start_sec = start_f / 50.0
                duration_sec = (stop_f - start_f) / 50.0
                aw = decode_audio_segment(path, start_sec, duration_sec)
            elif os.path.isdir(str(path)):
                start_f = self.start_frames[idx]
                stop_f = self.stop_frames[idx]
                picked = decode_frames_from_dir(path, start_f, stop_f)
                aw = decode_audio(path)
            else:
                picked = decode_frames_uniform(path)
                aw = decode_audio(path)
                
            pv = self.processor(picked, return_tensors="pt")['pixel_values'][0]
            audio_feat = self.ast_extractor(aw, sampling_rate=16000, return_tensors="pt")['input_values'][0]
            return pv, audio_feat, idx
        except Exception:
            return None


def collate(batch):
    batch = [b for b in batch if b is not None]
    if not batch:
        return None, None, []
    pv, aw, idxs = zip(*batch)
    return torch.stack(pv), torch.stack(aw), list(idxs)


class VideoMAEWrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
    def forward(self, pixel_values):
        return self.model(pixel_values=pixel_values).last_hidden_state.mean(dim=1)


class ASTWrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
    def forward(self, input_values):
        return self.model(input_values).last_hidden_state.mean(dim=1)


def main():
    import sys
    print(f"Using device: {DEVICE}")
    if DEVICE == 'cpu':
        print("\nWARNING: CUDA is not available! Feature extraction will run on CPU and be extremely slow.")
        print("Please check that your Kaggle notebook has GPU acceleration enabled in settings.\n")
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

    AST_MODEL_ID = 'MIT/ast-finetuned-audioset-10-10-0.4593'
    ast_extractor = ASTFeatureExtractor.from_pretrained(AST_MODEL_ID, token=hf_token)
    _ast_full = ASTForAudioClassification.from_pretrained(AST_MODEL_ID, token=hf_token)
    ast_model = _ast_full.audio_spectrogram_transformer.to(DEVICE).eval()  # base model only
    for p in ast_model.parameters():
        p.requires_grad_(False)
    del _ast_full  # free classifier head weights

    # Wrap models
    videomae_wrapped = VideoMAEWrapper(model)
    ast_wrapped = ASTWrapper(ast_model)

    batch_size = BATCH_SIZE
    num_gpus = torch.cuda.device_count()
    if num_gpus > 1:
        device_ids = list(range(num_gpus))
        print(f"Using DataParallel on {num_gpus} GPUs: {device_ids}")
        videomae_wrapped = torch.nn.DataParallel(videomae_wrapped, device_ids=device_ids)
        ast_wrapped = torch.nn.DataParallel(ast_wrapped, device_ids=device_ids)
        batch_size = 32 * num_gpus
        print(f"Scaled batch size to {batch_size}")

    ds = EpicFeatureDataset(
        remaining.path.tolist(),
        remaining.start_frame.tolist() if 'start_frame' in remaining.columns else None,
        remaining.stop_frame.tolist() if 'stop_frame' in remaining.columns else None,
        remaining.is_video.tolist() if 'is_video' in remaining.columns else None,
        processor,
        ast_extractor
    )
    loader = DataLoader(ds, batch_size=batch_size, collate_fn=collate,
                        num_workers=NUM_WORKERS, shuffle=False, pin_memory=True)

    new_feats, new_narr = [], []
    t0 = time.time()
    n_done = 0
    with torch.no_grad():
        for pv, aw, idxs in tqdm(loader, desc="Extracting features", unit="batch"):
            if pv is None or aw is None:
                continue
            pv = pv.to(DEVICE, non_blocking=True)
            video_out = videomae_wrapped(pv)
            
            audio_in = aw.to(DEVICE, non_blocking=True)
            audio_out = ast_wrapped(audio_in)
            
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

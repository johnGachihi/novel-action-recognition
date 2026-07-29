#!/usr/bin/env bash
# =============================================================================
#  EPIC-KITCHENS-100 Multimodal Action Recognition — End-to-End Pipeline
#
#  Usage:
#    ./run_end_to_end.sh [OPTIONS]
#
#  Options:
#    --clips-dir PATH      Where to download/store pre-clipped narrations
#                          (default: ./data/epic_kitchens_clips/clips)
#    --participants LIST   Comma-separated participant IDs to download, e.g. P01,P02 (default: all)
#    --label-space SPACE   verb | noun | all (default: all)
#    --limit N             Smoke-test mode: cap at N narrations across all stages (default: no limit)
#    --hf-token TOKEN      HuggingFace token (or set HF_TOKEN env var). Optional for public repos.
#    --skip-download       Skip HF clip download (clips already present in --clips-dir)
#    --skip-extraction     Skip feature extraction (features_epic.npz already exists)
#    --skip-eval           Skip evaluation stage
#    --use-raw-download    Legacy: download full raw videos via Bristol server + ffmpeg clip instead of HF
#    -h | --help           Show this help
#
#  Requirements:
#    - python3 (>=3.9)
#    - uv or pip (for ML packages)
#    - ffmpeg (only when --use-raw-download is set)
# =============================================================================
set -euo pipefail

# ---------- colours ----------------------------------------------------------
RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'
BLUE='\033[0;34m'; BOLD='\033[1m'; NC='\033[0m'

# ---------- defaults ---------------------------------------------------------
RAW_DIR="data/epic_kitchens_raw"
CLIPS_DIR="data/epic_kitchens_clips/clips"
PARTICIPANTS="all"
LABEL_SPACE="all"
SKIP_DOWNLOAD=false
SKIP_CLIPPING=false
SKIP_EXTRACTION=false
SKIP_EVAL=false
USE_RAW_DOWNLOAD=false   # set to true to use the legacy Bristol raw-video path
LIMIT=0          # 0 means no limit
HF_TOKEN="${HF_TOKEN:-}"
ANNOT_DIR="epic-kitchens-100-annotations"
DL_SCRIPTS_DIR="epic-kitchens-download-scripts"
HF_REPO="lightly-ai/epic-kitchens-100-clips"
RUN="python3"   # overridden in step 1 (uv preferred); pre-init avoids set -u errors

# ---------- parse args -------------------------------------------------------
while [[ $# -gt 0 ]]; do
    case "$1" in
        --output-path)      RAW_DIR="$2";           shift 2 ;;
        --clips-dir)        CLIPS_DIR="$2";         shift 2 ;;
        --participants)     PARTICIPANTS="$2";      shift 2 ;;
        --label-space)      LABEL_SPACE="$2";       shift 2 ;;
        --skip-download)    SKIP_DOWNLOAD=true;     shift ;;
        --skip-clipping)    SKIP_CLIPPING=true;     shift ;;
        --skip-extraction)  SKIP_EXTRACTION=true;   shift ;;
        --skip-eval)        SKIP_EVAL=true;         shift ;;
        --use-raw-download) USE_RAW_DOWNLOAD=true;  shift ;;
        --limit)            LIMIT="$2";             shift 2 ;;
        --hf-token)         HF_TOKEN="$2";          shift 2 ;;
        -h|--help) sed -n '3,26p' "$0" | sed 's/^#  \?//'; exit 0 ;;
        *) echo -e "${RED}Unknown option: $1${NC}"; exit 1 ;;
    esac
done


step() { echo -e "\n${BOLD}${BLUE}==== $* ====${NC}"; }
ok()   { echo -e "${GREEN}✓ $*${NC}"; }
warn() { echo -e "${YELLOW}⚠  $*${NC}"; }
die()  { echo -e "${RED}✗ $*${NC}" >&2; exit 1; }

[[ "$LIMIT" -gt 0 ]] && warn "Smoke-test mode: pipeline limited to $LIMIT narrations."

# =============================================================================
step "[1/5] Environment setup"
# =============================================================================

if command -v uv &>/dev/null; then
    ok "Found uv in PATH."
    RUN="uv run python3"
    uv sync --quiet
elif [[ -f "$HOME/.local/bin/uv" ]]; then
    export PATH="$HOME/.local/bin:$PATH"
    ok "Found uv in ~/.local/bin."
    RUN="uv run python3"
    uv sync --quiet
else
    warn "uv not found — falling back to python3 venv + pip."
    python3 -m venv .venv
    source .venv/bin/activate
    pip install --upgrade pip --quiet
    pip install -e . --quiet
    RUN="python3"
fi
ok "Environment ready. Runner: $RUN"

# =============================================================================
step "[2/5] Downloading annotation CSVs and EPIC download scripts"
# =============================================================================

ANNOT_BASE="https://raw.githubusercontent.com/epic-kitchens/epic-kitchens-100-annotations/master"
DL_BASE="https://raw.githubusercontent.com/epic-kitchens/epic-kitchens-download-scripts/master"

mkdir -p "$ANNOT_DIR"
for f in EPIC_100_train.csv EPIC_100_validation.csv; do
    dest="$ANNOT_DIR/$f"
    [[ -f "$dest" ]] && ok "$f already present." && continue
    echo "Downloading $f..."
    curl -fsSL "$ANNOT_BASE/$f" -o "$dest" || die "Failed to download $f"
    ok "$f downloaded."
done

# Download the official downloader + its metadata CSVs (pure Python stdlib, no pip needed)
mkdir -p "$DL_SCRIPTS_DIR/data"
for f in epic_downloader.py data/epic_100_splits.csv data/epic_55_splits.csv data/md5.csv data/errata.csv; do
    dest="$DL_SCRIPTS_DIR/$f"
    [[ -f "$dest" ]] && continue
    echo "Downloading $f..."
    curl -fsSL "$DL_BASE/$f" -o "$dest" || die "Failed to download $f"
done
ok "EPIC download scripts ready."

# =============================================================================
step "[3/5] Downloading pre-clipped EPIC-KITCHENS-100 narrations (HuggingFace)"
# =============================================================================

if [[ "$USE_RAW_DOWNLOAD" == "true" ]]; then
    # -------------------------------------------------------------------------
    # LEGACY PATH: download full raw videos from Bristol server, then clip.
    # Only needed if you want uncompressed originals. Requires ffmpeg.
    # -------------------------------------------------------------------------
    warn "--use-raw-download: using legacy Bristol server + ffmpeg flow."

    if [[ "$SKIP_DOWNLOAD" == "true" ]]; then
        warn "--skip-download set: skipping raw video download."
    else
        echo "Downloading raw videos to: $RAW_DIR"
        echo "(This is ~700 GB for all participants — use --participants P01,P02 to limit scope)"

        PARTICIPANT_ARG=""
        [[ "$PARTICIPANTS" != "all" ]] && PARTICIPANT_ARG="--participants $PARTICIPANTS"

        SPECIFIC_VIDEOS_ARG=""
        if [[ "$LIMIT" -gt 0 ]]; then
            SPECIFIC_VIDEOS=$(python3 - <<PYEOF
import csv
rows = list(csv.DictReader(open("$ANNOT_DIR/EPIC_100_train.csv")))
seen_vids, seen_narr = [], set()
for r in rows:
    if r['video_id'] not in seen_narr:
        seen_narr.add(r['video_id'])
        seen_vids.append(r['video_id'])
    if len(seen_vids) >= $LIMIT:
        break
print(','.join(seen_vids))
PYEOF
)
            SPECIFIC_VIDEOS_ARG="--specific-videos $SPECIFIC_VIDEOS"
            warn "--limit $LIMIT: downloading only videos: $SPECIFIC_VIDEOS"
        fi

        ABS_RAW_DIR="$(cd "$(dirname "$RAW_DIR")" 2>/dev/null && pwd)/$(basename "$RAW_DIR")" || ABS_RAW_DIR="$PWD/$RAW_DIR"
        mkdir -p "$ABS_RAW_DIR"
        (
            cd "$DL_SCRIPTS_DIR"
            python3 epic_downloader.py \
                --videos \
                --output-path "$ABS_RAW_DIR" \
                --train \
                $PARTICIPANT_ARG \
                $SPECIFIC_VIDEOS_ARG \
                || exit 1
        ) || die "Video download failed."
        ok "Raw video download complete."
    fi

else
    # -------------------------------------------------------------------------
    # DEFAULT PATH: download pre-clipped narrations directly from HuggingFace.
    # Much faster — no ffmpeg needed, no 700 GB raw videos.
    # -------------------------------------------------------------------------
    SKIP_CLIPPING=true   # no clipping needed — HF already has per-narration clips

    if [[ "$SKIP_DOWNLOAD" == "true" ]]; then
        warn "--skip-download set: skipping HF clip download."
        IFS=',' read -r -a clip_dirs_arr <<< "$CLIPS_DIR"
        clip_count=$(find "${clip_dirs_arr[@]}" -name '*.mp4' 2>/dev/null | wc -l)
        [[ "$clip_count" -eq 0 ]] && die "No clips found in $CLIPS_DIR. Remove --skip-download or point --clips-dir at existing clips."
        ok "Using $clip_count existing clips in $CLIPS_DIR."
    else
        mkdir -p "$CLIPS_DIR"
        echo "Downloading pre-clipped narrations from HuggingFace: $HF_REPO"
        echo "Target: $CLIPS_DIR"

        HF_TOKEN_ARG=""
        [[ -n "$HF_TOKEN" ]] && HF_TOKEN_ARG="--token $HF_TOKEN"

        # Build allow-patterns: restrict to specific participants if requested
        ALLOW_PATTERNS="clips/P*/*.mp4"
        PARTICIPANT_FILTER=""
        if [[ "$PARTICIPANTS" != "all" ]]; then
            IFS=',' read -r -a parts_arr <<< "$PARTICIPANTS"
            patterns=$(printf "clips/%s/*.mp4 " "${parts_arr[@]}")
            ALLOW_PATTERNS="$patterns"
            warn "Downloading only participants: $PARTICIPANTS"
        fi

        # Use huggingface_hub to download — install it if missing
        LIMIT="$LIMIT" HF_REPO="$HF_REPO" CLIPS_DIR="$CLIPS_DIR" HF_TOKEN="$HF_TOKEN" PARTICIPANTS="$PARTICIPANTS" ANNOT_DIR="$ANNOT_DIR" $RUN - <<"PYEOF"
import sys, subprocess
try:
    import huggingface_hub
except ImportError:
    subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'huggingface_hub', '-q'])
    import huggingface_hub

from huggingface_hub import snapshot_download
import os, csv

repo_id   = os.environ.get("HF_REPO", "")
clips_dir = os.environ.get("CLIPS_DIR", "")
token     = os.environ.get("HF_TOKEN", "") or None
limit     = int(os.environ.get("LIMIT", "0"))
participants = os.environ.get("PARTICIPANTS", "all")
annot_dir = os.environ.get("ANNOT_DIR", "")

# Build allow_patterns
if participants == 'all':
    if limit > 0:
        # Only download participants needed for the first limit narrations
        rows = list(csv.DictReader(open(f"{annot_dir}/EPIC_100_train.csv")))[:limit]
        needed_pids = sorted({r['participant_id'] for r in rows})
        print(f"Smoke-test: downloading participants {needed_pids} (covers first {limit} narrations)")
        patterns = [f"clips/{p}/*.mp4" for p in needed_pids]
    else:
        patterns = ["clips/P*/*.mp4"]
else:
    patterns = [f"clips/{p.strip()}/*.mp4" for p in participants.split(',')]

print(f"Downloading with patterns: {patterns}")

local_dir = os.path.dirname(clips_dir)  # parent of 'clips/'
os.makedirs(local_dir, exist_ok=True)

snapshot_download(
    repo_id=repo_id,
    repo_type="dataset",
    local_dir=local_dir,
    allow_patterns=patterns,
    token=token,
    ignore_patterns=["*.json", "*.csv", "README*"],
    disable_tqdm=True,
)

# Count what we got
import glob
mp4s = glob.glob(f"{clips_dir}/*/*.mp4")
print(f"\nDownload complete: {len(mp4s)} clips in {clips_dir}")
if limit > 0 and len(mp4s) == 0:
    print("WARNING: 0 clips downloaded. Check your HF token or network.")
    sys.exit(1)
PYEOF
        ok "HuggingFace clip download complete."
    fi
fi

# =============================================================================
step "[4/5] Per-narration clipping (legacy --use-raw-download only)"
# =============================================================================

if [[ "$SKIP_CLIPPING" == "true" ]]; then
    warn "Clipping step skipped (HF pre-clipped mode or --skip-clipping set)."
else
    # This path is only reached when --use-raw-download is set and --skip-clipping is not.
    command -v ffmpeg &>/dev/null || die "ffmpeg is required for clipping but was not found. Install it with: sudo apt install ffmpeg"

    mkdir -p "$CLIPS_DIR"
    echo "Clipping narrations from $ANNOT_DIR/EPIC_100_train.csv..."

    # Python one-liner to read the CSV and produce ffmpeg cut commands
    python3 - <<PYEOF
import csv, os, subprocess, sys

annot = "$ANNOT_DIR/EPIC_100_train.csv"
raw_root = os.path.join("$RAW_DIR", "EPIC-KITCHENS")
clips_root = "$CLIPS_DIR"

with open(annot) as f:
    rows = list(csv.DictReader(f))

limit = $LIMIT
if limit > 0:
    rows = rows[:limit]

total = len(rows)
done = skipped = failed = 0

for i, row in enumerate(rows):
    pid   = row['participant_id']           # e.g. P01
    vid   = row['video_id']                 # e.g. P01_01
    nid   = row['narration_id']             # e.g. P01_01_0
    start = row['start_timestamp']          # HH:MM:SS.sss
    stop  = row['stop_timestamp']

    out_dir  = os.path.join(clips_root, pid)
    out_path = os.path.join(out_dir, f"{nid}.mp4")

    if os.path.exists(out_path):
        skipped += 1
        continue

    # Try both .MP4 (Bristol) and .mp4 layouts
    raw_candidates = [
        os.path.join(raw_root, pid, "videos", f"{vid}.MP4"),
        os.path.join(raw_root, pid, "videos", f"{vid}.mp4"),
    ]
    raw_path = next((p for p in raw_candidates if os.path.exists(p)), None)
    if raw_path is None:
        failed += 1
        continue

    os.makedirs(out_dir, exist_ok=True)
    cmd = [
        "ffmpeg", "-y", "-loglevel", "error",
        "-i", raw_path,
        "-ss", start, "-to", stop,
        "-c:v", "libx264", "-crf", "18", "-preset", "fast",
        "-c:a", "aac",
        out_path
    ]
    result = subprocess.run(cmd)
    if result.returncode == 0:
        done += 1
    else:
        failed += 1

    if (i + 1) % 500 == 0:
        print(f"  Progress: {i+1}/{total} — clipped={done} skipped={skipped} failed={failed}", flush=True)

print(f"\nClipping complete: clipped={done}  skipped={skipped}  failed={failed}")
if done == 0 and skipped == 0:
    print(f"ERROR: 0 clips produced — {failed} source videos not found.")
    print(f"  Searched under: {raw_root}")
    print(f"  Expected layout: {raw_root}/<participant>/videos/<video_id>.MP4")
    import sys; sys.exit(1)
elif failed > 0:
    print(f"WARNING: {failed} narrations could not be clipped (source video not found).")
PYEOF
    rc=$?
    [[ $rc -ne 0 ]] && die "Clipping produced 0 clips. Check that the download completed and the path layout is correct."

    ok "Clipping complete."
fi

# =============================================================================
step "[4b/5] Pre-flight check"
# =============================================================================

IFS=',' read -r -a clip_dirs_arr <<< "$CLIPS_DIR"
clip_count=$(find "${clip_dirs_arr[@]}" -name '*.mp4' 2>/dev/null | wc -l)
[[ "$clip_count" -eq 0 ]] && die "No .mp4 clips found in $CLIPS_DIR."
ok "Found $clip_count clips."

# Patch clips root into nac/epic_data.py if it differs from the default
$RUN - <<PYEOF
import re, pathlib
src = pathlib.Path("nac/epic_data.py").read_text()
current = re.search(r"CLIPS_ROOT = '(.+?)'", src)
if current and current.group(1) != "$CLIPS_DIR":
    src = src.replace(f"CLIPS_ROOT = '{current.group(1)}'", f"CLIPS_ROOT = '$CLIPS_DIR'")
    pathlib.Path("nac/epic_data.py").write_text(src)
    print(f"Updated CLIPS_ROOT to $CLIPS_DIR")
PYEOF

# =============================================================================
step "[5a/5] Multimodal feature extraction (VideoMAE + AST)"
# =============================================================================

if [[ "$SKIP_EXTRACTION" == "true" ]]; then
    warn "--skip-extraction set: skipping feature extraction."
    [[ -f "features_epic.npz" ]] || die "features_epic.npz not found."
else
    echo "Extracting features... (checkpointed every 5000 clips — safe to interrupt & resume)"
    EPIC_LIMIT="$LIMIT" $RUN extract_epic_features.py
    ok "Feature extraction complete → features_epic.npz"
fi

# =============================================================================
step "[5b/5] Evaluation — Stage 1 (Novelty Detection) + Stage 2 (Discovery)"
# =============================================================================

if [[ "$SKIP_EVAL" == "true" ]]; then
    warn "--skip-eval set: skipping evaluation."
else
    $RUN evaluate_pipeline.py --label-space "$LABEL_SPACE" --out evaluation_results.json
    rc=$?
    if [[ $rc -ne 0 ]]; then
      if [[ "$LABEL_SPACE" == "all" ]]; then
        warn "Evaluation failed for label-space 'all'; retrying with '--label-space verb'..."
        $RUN evaluate_pipeline.py --label-space verb --out evaluation_results.json || die "Evaluation failed again."
      else
        die "Evaluation failed."
      fi
    fi
    ok "Evaluation complete → evaluation_results.json"
fi

echo -e "\n${BOLD}${GREEN}Pipeline completed successfully!${NC}"
echo -e "  Results : ${BOLD}evaluation_results.json${NC}"
echo -e "  Features: ${BOLD}features_epic.npz${NC}"

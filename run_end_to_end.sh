#!/usr/bin/env bash
# =============================================================================
#  EPIC-KITCHENS-100 Multimodal Action Recognition — End-to-End Pipeline
#
#  Usage:
#    ./run_end_to_end.sh [OPTIONS]
#
#  Options:
#    --output-path PATH    Where to download raw videos (default: ./data/epic_kitchens_raw)
#    --clips-dir PATH      Where clipped narrations will be stored
#                          (default: ./data/epic_kitchens_clips/clips)
#    --participants LIST   Comma-separated participant IDs, e.g. P01,P02 (default: all)
#    --label-space SPACE   verb | noun | all (default: all)
#    --limit N             Smoke-test mode: cap at N narrations across all stages (default: no limit)
#    --skip-download       Skip video download (raw videos already present)
#    --skip-clipping       Skip per-narration clipping (clips already present)
#    --skip-extraction     Skip feature extraction (features_epic.npz already exists)
#    --skip-eval           Skip evaluation stage
#    -h | --help           Show this help
#
#  Requirements:
#    - python3 (>=3.9, stdlib only for download stage)
#    - ffmpeg (for narration clipping)
#    - uv or pip (for ML packages in extraction/eval stages)
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
LIMIT=0          # 0 means no limit
ANNOT_DIR="epic-kitchens-100-annotations"
DL_SCRIPTS_DIR="epic-kitchens-download-scripts"

# ---------- parse args -------------------------------------------------------
while [[ $# -gt 0 ]]; do
    case "$1" in
        --output-path)   RAW_DIR="$2";        shift 2 ;;
        --clips-dir)     CLIPS_DIR="$2";      shift 2 ;;
        --participants)  PARTICIPANTS="$2";   shift 2 ;;
        --label-space)   LABEL_SPACE="$2";    shift 2 ;;
        --skip-download)   SKIP_DOWNLOAD=true;   shift ;;
        --skip-clipping)   SKIP_CLIPPING=true;   shift ;;
        --skip-extraction) SKIP_EXTRACTION=true; shift ;;
        --skip-eval)       SKIP_EVAL=true;        shift ;;
        --limit)           LIMIT="$2";           shift 2 ;;
        -h|--help) sed -n '3,15p' "$0" | sed 's/^#  \?//'; exit 0 ;;
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
step "[3/5] Downloading EPIC-KITCHENS-100 videos"
# =============================================================================

if [[ "$SKIP_DOWNLOAD" == "true" ]]; then
    warn "--skip-download set: skipping video download."
else
    echo "Downloading videos to: $RAW_DIR"
    echo "(This is ~700 GB for all participants — use --participants P01,P02 to limit scope)"

    PARTICIPANT_ARG=""
    [[ "$PARTICIPANTS" != "all" ]] && PARTICIPANT_ARG="--participants $PARTICIPANTS"

    # In --limit mode, derive the minimal set of unique video_ids needed from the CSV
    # and pass them as --specific-videos so only those files are fetched.
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

    # The downloader resolves its metadata CSVs relative to its own cwd,
    # so we must cd into the scripts dir. We pass an absolute output path so
    # files land in the right place regardless of cwd.
    ABS_RAW_DIR="$(cd "$(dirname "$RAW_DIR")" 2>/dev/null && pwd)/$(basename "$RAW_DIR")" || ABS_RAW_DIR="$PWD/$RAW_DIR"
    mkdir -p "$ABS_RAW_DIR"

    # --extension-only: download only the EPIC-100 new videos, not EPIC-55 re-runs
    (
        cd "$DL_SCRIPTS_DIR"
        python3 epic_downloader.py \
            --videos \
            --extension-only \
            --output-path "$ABS_RAW_DIR" \
            --train \
            $PARTICIPANT_ARG \
            $SPECIFIC_VIDEOS_ARG \
            || exit 1
    ) || die "Video download failed."
    ok "Video download complete."
fi

# =============================================================================
step "[4/5] Clipping per-narration segments from full videos"
# =============================================================================

if [[ "$SKIP_CLIPPING" == "true" ]]; then
    warn "--skip-clipping set: skipping narration clipping."
    clip_count=$(find "$CLIPS_DIR" -name '*.mp4' 2>/dev/null | wc -l)
    [[ "$clip_count" -eq 0 ]] && die "No clips found in $CLIPS_DIR and --skip-clipping is set."
    ok "Using $clip_count existing clips."
else
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
if failed > 0:
    print(f"WARNING: {failed} narrations could not be clipped (source video not found).")
PYEOF

    ok "Clipping complete."
fi

# =============================================================================
step "[4b/5] Pre-flight check"
# =============================================================================

clip_count=$(find "$CLIPS_DIR" -name '*.mp4' 2>/dev/null | wc -l)
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
    ok "Evaluation complete → evaluation_results.json"
fi

echo -e "\n${BOLD}${GREEN}Pipeline completed successfully!${NC}"
echo -e "  Results : ${BOLD}evaluation_results.json${NC}"
echo -e "  Features: ${BOLD}features_epic.npz${NC}"

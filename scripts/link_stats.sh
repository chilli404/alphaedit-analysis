#!/usr/bin/env bash
set -euo pipefail

# Links precomputed covariance statistics into the AlphaEdit data directory.
# Uses S3 stats if available; otherwise falls back to project-local stats.
# Supports multiple models via MODEL_NAME env var.
# Usage: bash scripts/link_stats.sh
#        MODEL_NAME="Qwen/Qwen2.5-7B-Instruct" bash scripts/link_stats.sh

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
S3_DIR="/s3-data"

# Preserve caller's MODEL_NAME before sourcing .env (which may override it)
_CALLER_MODEL_NAME="${MODEL_NAME:-}"

# Load environment config
if [[ -f "$PROJECT_DIR/.env" ]]; then
    set -a; source "$PROJECT_DIR/.env"; set +a
fi

# Caller's explicit MODEL_NAME takes priority over .env
MODEL_NAME="${_CALLER_MODEL_NAME:-${MODEL_NAME:-meta-llama/Meta-Llama-3-8B-Instruct}}"
_MODEL_SHORT="${MODEL_NAME##*/}"

# Map model name to stats subdirectory
stats_subdir_for_model() {
    local model="$1"
    case "$model" in
        *Meta-Llama-3-8B*)          echo "llama3-8b-instruct" ;;
        *gpt-j-6[bB]*)             echo "gpt-j-6b" ;;
        *Qwen2.5-7B*)              echo "qwen2.5-7b-instruct" ;;
        *Mistral-7B*|*mistral-7b*) echo "mistral-7b" ;;
        *)                         echo "$(echo "${model##*/}" | tr '[:upper:]' '[:lower:]')" ;;
    esac
}

STATS_SUBDIR="$(stats_subdir_for_model "$MODEL_NAME")"

S3_STATS_SRC="$S3_DIR/continual-learning/alphaedit/stats/$STATS_SUBDIR"
PROJECT_STATS_SRC="$PROJECT_DIR/data/stats/$STATS_SUBDIR/wikipedia_stats"

STATS_SRC="$PROJECT_STATS_SRC"
[[ -d "$S3_STATS_SRC" ]] && STATS_SRC="$S3_STATS_SRC"

STATS_DST="$PROJECT_DIR/vendor/AlphaEdit/data/stats/${_MODEL_SHORT}/wikipedia_stats"

if [[ ! -d "$STATS_SRC" ]]; then
    echo "ERROR: Stats source directory not found."
    echo "Checked:"
    echo "  S3:      $S3_STATS_SRC"
    echo "  Project: $PROJECT_STATS_SRC"
    echo ""
    echo "To generate stats for this model, run:"
    echo "  uv run python scripts/build_stats.py --model $MODEL_NAME"
    exit 1
fi

echo "=== Linking Covariance Statistics ==="
echo "  Model:  $MODEL_NAME"
echo "  Stats:  $STATS_SUBDIR"
echo "  Source: $STATS_SRC"
echo "  Target: $STATS_DST"

mkdir -p "$STATS_DST"

COUNT=0
for f in "$STATS_SRC"/*.npz; do
    if [[ -f "$f" ]]; then
        ln -sf "$f" "$STATS_DST/$(basename "$f")"
        COUNT=$((COUNT + 1))
    fi
done

if [[ $COUNT -eq 0 ]]; then
    echo "ERROR: No .npz files found in $STATS_SRC"
    exit 1
fi

echo "  Linked $COUNT stats files."
echo ""

echo "SHA256 checksums:"
for f in "$STATS_DST"/*.npz; do
    if [[ -f "$f" ]]; then
        shasum -a 256 "$f"
    fi
done

echo ""

# Link cached null-space projection (P) if available (avoids 45-min SVD recomputation)
P_CACHE_SRC="$STATS_SRC/null_space_project.pt"
P_CACHE_DST="$PROJECT_DIR/vendor/AlphaEdit/null_space_project.pt"
if [[ -f "$P_CACHE_SRC" ]]; then
    ln -sf "$P_CACHE_SRC" "$P_CACHE_DST"
    echo "  Linked cached null-space projection: $P_CACHE_SRC"
else
    echo "  No cached null-space projection found (will compute on first run)"
fi

# Link per-threshold P caches (projection capacity sweep)
P_THRESHOLD_COUNT=0
for p_file in "$STATS_SRC"/null_space_project_t*.pt; do
    if [[ -f "$p_file" ]]; then
        ln -sf "$p_file" "$PROJECT_DIR/vendor/AlphaEdit/$(basename "$p_file")"
        P_THRESHOLD_COUNT=$((P_THRESHOLD_COUNT + 1))
    fi
done
if [[ $P_THRESHOLD_COUNT -gt 0 ]]; then
    echo "  Linked $P_THRESHOLD_COUNT per-threshold P cache(s)"
fi

echo "=== Done ==="

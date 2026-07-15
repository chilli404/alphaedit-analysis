#!/usr/bin/env bash
set -euo pipefail

# Links precomputed covariance statistics into the AlphaEdit data directory.
# Uses S3 stats if available; otherwise falls back to project-local stats.
# Usage: bash scripts/link_stats.sh

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
S3_DIR="/s3-data"

# Load environment config
if [[ -f "$PROJECT_DIR/.env" ]]; then
    set -a; source "$PROJECT_DIR/.env"; set +a
fi

MODEL_NAME="${MODEL_NAME:-meta-llama/Meta-Llama-3-8B-Instruct}"
_MODEL_SHORT="${MODEL_NAME##*/}"

S3_STATS_SRC="$S3_DIR/continual-learning/alphaedit/stats/llama3-8b-instruct"
PROJECT_STATS_SRC="$PROJECT_DIR/data/stats/llama3-8b-instruct/wikipedia_stats"

STATS_SRC="$PROJECT_STATS_SRC"
[[ -d "$S3_STATS_SRC" ]] && STATS_SRC="$S3_STATS_SRC"

STATS_DST="$PROJECT_DIR/vendor/AlphaEdit/data/stats/${_MODEL_SHORT}/wikipedia_stats"

if [[ ! -d "$STATS_SRC" ]]; then
    echo "ERROR: Stats source directory not found."
    echo "Checked:"
    echo "  S3:      $S3_STATS_SRC"
    echo "  Project: $PROJECT_STATS_SRC"
    exit 1
fi

echo "=== Linking Covariance Statistics ==="
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
echo "Expected files: model.layers.{4,5,6,7,8}.mlp.down_proj_float32_mom2_100000.npz"

# Link cached null-space projection (P) if available (avoids 45-min SVD recomputation)
P_CACHE_SRC="$STATS_SRC/null_space_project.pt"
P_CACHE_DST="$PROJECT_DIR/vendor/AlphaEdit/null_space_project.pt"
if [[ -f "$P_CACHE_SRC" ]]; then
    ln -sf "$P_CACHE_SRC" "$P_CACHE_DST"
    echo "  Linked cached null-space projection: $P_CACHE_SRC"
else
    echo "  No cached null-space projection found (will compute on first run)"
fi

echo "=== Done ==="
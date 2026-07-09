#!/usr/bin/env bash
set -euo pipefail

# Links precomputed covariance statistics into the AlphaEdit data directory.
# Usage: bash scripts/link_stats.sh [/path/to/stats/dir]

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

# Default source: user's Downloads folder
STATS_SRC="${1:-/Users/xksc003/Downloads/llama3-8b-instruct/wikipedia_stats}"
STATS_DST="$PROJECT_DIR/vendor/AlphaEdit/data/stats/Meta-Llama-3-8B-Instruct/wikipedia_stats"

if [[ ! -d "$STATS_SRC" ]]; then
    echo "ERROR: Stats source directory not found: $STATS_SRC"
    echo "Usage: bash scripts/link_stats.sh /path/to/wikipedia_stats/"
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

# Print checksums for reproducibility record
echo "SHA256 checksums:"
for f in "$STATS_DST"/*.npz; do
    shasum -a 256 "$f"
done

echo ""
echo "Expected files: model.layers.{4,5,6,7,8}.mlp.down_proj_float32_mom2_100000.npz"
echo "=== Done ==="

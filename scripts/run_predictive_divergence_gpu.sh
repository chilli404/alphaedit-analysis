#!/usr/bin/env bash
set -euo pipefail

# Extended Cache Metrics (GPU) for Predictive Divergence Analysis
#
# Computes rayleigh_quotient, key_crowding_proxy, linear_effective_rank,
# and top_eigenvalue_concentration from raw cache_c.pt checkpoints.
#
# Usage:
#   bash scripts/run_predictive_divergence_gpu.sh [SEED]
#   bash scripts/run_predictive_divergence_gpu.sh 42
#   bash scripts/run_predictive_divergence_gpu.sh 2024

SEED="${1:-42}"
CHECKPOINT_BASE="${CHECKPOINT_BASE:-/s3-data/continual-learning/alphaedit/checkpoints/AlphaEdit/seed${SEED}}"

# Fallback to local cache if S3 not mounted
if [[ ! -d "$CHECKPOINT_BASE" ]]; then
    CHECKPOINT_BASE="$HOME/.cache/alphaedit_checkpoints/AlphaEdit/seed${SEED}"
fi

echo "=== Extended Cache Metrics (GPU) ==="
echo "  Seed: $SEED"
echo "  Checkpoint base: $CHECKPOINT_BASE"
echo ""

if [[ ! -d "$CHECKPOINT_BASE" ]]; then
    echo "ERROR: Checkpoint directory not found: $CHECKPOINT_BASE"
    echo "Pull checkpoints from S3 first or set CHECKPOINT_BASE."
    exit 1
fi

uv run python -m analysis.predictive_divergence_gpu \
    --seed "$SEED" \
    --checkpoint_base "$CHECKPOINT_BASE"

echo ""
echo "=== Done (seed=$SEED) ==="

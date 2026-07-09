#!/usr/bin/env bash
set -euo pipefail

# Semantic Coupling Stress Test
# Tests whether AlphaEdit's null-space projection removes more of the
# edit direction when edits are semantically related to preserved knowledge.
#
# Usage:
#   bash scripts/run_coupling_stress.sh [SEED]
#   CUDA_DEVICE=1 bash scripts/run_coupling_stress.sh 42

SEED="${1:-42}"
CUDA_DEVICE="${CUDA_DEVICE:-0}"
MAX_PAIRS="${MAX_PAIRS:-60}"
WARMUP="${WARMUP:-20}"

echo "=== Semantic Coupling Stress Test ==="
echo "  Seed: $SEED"
echo "  CUDA device: $CUDA_DEVICE"
echo "  Pairs/type: $MAX_PAIRS"
echo "  Warmup edits: $WARMUP"
echo ""

# Run the coupling stress runner (generates dataset internally)
uv run python src/coupling_stress_runner.py \
    --seed "$SEED" \
    --cuda_device "$CUDA_DEVICE" \
    --max_pairs_per_type "$MAX_PAIRS" \
    --warmup_count "$WARMUP" \
    --conserve_memory

echo ""
echo "=== Done (seed=$SEED) ==="

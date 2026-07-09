#!/usr/bin/env bash
set -euo pipefail

# Null-Space Rank Consumption Analysis
#
# Runs the instrumented AlphaEdit experiment that tracks how the null-space
# is consumed as sequential edits accumulate. This produces the mechanistic
# analysis figure showing WHY AlphaEdit eventually degrades.
#
# The tracker records per-edit-batch:
#   - Initial null-space rank per layer (from SVD threshold)
#   - Accumulated covariance cache_c rank (grows with edits)
#   - Consumption ratio (cache rank / null-space rank)
#   - Spectral properties (top singular values)
#
# Usage:
#   bash scripts/run_nullspace_analysis.sh [SEED] [DATASET_SIZE_LIMIT]
#   bash scripts/run_nullspace_analysis.sh 42 2000   # Full run
#   bash scripts/run_nullspace_analysis.sh 42 500    # Quick run

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

SEED="${1:-42}"
DATASET_SIZE_LIMIT="${2:-2000}"
CUDA_DEVICE="${CUDA_DEVICE:-0}"
NUM_EDITS="${NUM_EDITS:-100}"
MODEL_NAME="${MODEL_NAME:-meta-llama/Meta-Llama-3-8B-Instruct}"
HPARAMS_FNAME="${HPARAMS_FNAME:-Llama3-8B.json}"

echo "=== Null-Space Rank Consumption Analysis ==="
echo "  Seed: $SEED"
echo "  Dataset size: $DATASET_SIZE_LIMIT"
echo "  Num edits per batch: $NUM_EDITS"
echo "  Model: $MODEL_NAME"
echo "  Started: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo ""

cd "$PROJECT_DIR"

uv run python src/nullspace_tracker.py \
    --seed "$SEED" \
    --cuda_device "$CUDA_DEVICE" \
    --model_name "$MODEL_NAME" \
    --hparams_fname "$HPARAMS_FNAME" \
    --ds_name mcf \
    --dataset_size_limit "$DATASET_SIZE_LIMIT" \
    --num_edits "$NUM_EDITS" \
    --downstream_eval_steps 5 \
    --conserve_memory

echo ""
echo "=== Null-space analysis complete ==="
echo "  Finished: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "  Results: results/nullspace_tracking/"
echo ""
echo "Next: python analysis/plots.py  (generates mechanistic figure)"

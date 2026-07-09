#!/usr/bin/env bash
set -euo pipefail

# ROME Baseline: Calibration baseline for evaluation harness validation.
#
# ROME (Rank-One Model Editing) is the predecessor to MEMIT, editing one
# fact at a time. It should perform worse than both MEMIT and AlphaEdit
# on sequential editing tasks. If it doesn't, the evaluation harness has
# a bug.
#
# ROME only edits one layer at a time (single fact per call), so for
# sequential editing we apply it iteratively. The hparams already exist
# in the vendored AlphaEdit repo.
#
# Usage:
#   bash scripts/run_rome_baseline.sh [SEED] [DATASET_SIZE_LIMIT]
#   bash scripts/run_rome_baseline.sh 42       # Default: 2000 edits
#   bash scripts/run_rome_baseline.sh 42 500   # Quick: 500 edits

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

SEED="${1:-42}"
DATASET_SIZE_LIMIT="${2:-2000}"
CUDA_DEVICE="${CUDA_DEVICE:-0}"

echo "=== ROME Baseline (seed=$SEED, limit=$DATASET_SIZE_LIMIT) ==="
echo "  Purpose: Calibration baseline (expected: worse than MEMIT and AlphaEdit)"
echo "  Started: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo ""

cd "$PROJECT_DIR"

uv run python src/seeded_runner.py \
    --seed "$SEED" \
    --cuda_device "$CUDA_DEVICE" \
    --alg_name ROME \
    --model_name meta-llama/Meta-Llama-3-8B-Instruct \
    --hparams_fname Llama3-8B.json \
    --ds_name mcf \
    --dataset_size_limit "$DATASET_SIZE_LIMIT" \
    --num_edits 1 \
    --downstream_eval_steps 5 \
    --conserve_memory

echo ""
echo "=== ROME baseline complete ==="
echo "  Finished: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "  Results: vendor/AlphaEdit/results/"

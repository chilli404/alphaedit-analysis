#!/usr/bin/env bash
set -euo pipefail

# MEMIT+SeqReg: Non-projected analogue of AlphaEdit's sequential regularization.
#
# Scientific Question:
#   Does MEMIT with AlphaEdit-like regularization (Eq. 12) close the gap to
#   AlphaEdit, or is the null-space projection P necessary?
#
# Calibration settings:
#   A: λ_prev=1, λ_delta=1        # Direct Eq. 12 coefficient analogue
#   B: λ_prev=1, λ_delta=1e-4     # Weak ridge
#   C: λ_prev=10, λ_delta=1       # Strong prev-key protection
#   D: λ_prev=100, λ_delta=1      # Very strong prev-key protection
#
# Usage:
#   bash scripts/run_memit_sequential.sh [SEED] [LAMBDA_PREV] [LAMBDA_DELTA]
#   bash scripts/run_memit_sequential.sh 42 1 1          # Direct Eq. 12 analogue
#   bash scripts/run_memit_sequential.sh 42 0 0          # Original MEMIT
#   bash scripts/run_memit_sequential.sh 42 10 1         # Strong prev-key reg
#
# Environment variables:
#   CUDA_DEVICE      - GPU device index (default: 0)
#   MODEL_NAME       - Model to use (default: meta-llama/Meta-Llama-3-8B-Instruct)
#   CACHE_STRATEGY   - recent|all (default: recent)
#   CACHE_MAX        - Max batches in cache (default: 20, use "none" for unlimited)
#   DEBUG_BATCH      - If set, run same-state diagnostic at this batch
#   FAST_CHECKPOINT  - If "true", only evaluate edited batch (much faster)

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

# Load environment config
if [[ -f "$PROJECT_DIR/.env" ]]; then
    set -a; source "$PROJECT_DIR/.env"; set +a
fi

MODEL_NAME="${MODEL_NAME:-meta-llama/Meta-Llama-3-8B-Instruct}"

SEED="${1:-42}"
LAMBDA_PREV="${2:-${LAMBDA_PREV:-1.0}}"
LAMBDA_DELTA="${3:-${LAMBDA_DELTA:-1.0}}"
CUDA_DEVICE="${CUDA_DEVICE:-0}"
CACHE_STRATEGY="${CACHE_STRATEGY:-recent}"
CACHE_MAX="${CACHE_MAX:-20}"

echo "=== MEMIT+SeqReg ==="
echo "  Seed: $SEED"
echo "  λ_prev: $LAMBDA_PREV"
echo "  λ_delta: $LAMBDA_DELTA"
echo "  Cache: strategy=$CACHE_STRATEGY, max=$CACHE_MAX"
echo "  Started: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo ""

cd "$PROJECT_DIR"

# Build debug arg if set
DEBUG_ARG=""
if [[ -n "${DEBUG_BATCH:-}" ]]; then
    DEBUG_ARG="--debug_freeze_batch $DEBUG_BATCH"
    echo "  DEBUG MODE: freeze at batch $DEBUG_BATCH"
fi

# Build fast checkpoint flag if set
FAST_FLAG=""
if [[ "${FAST_CHECKPOINT:-false}" == "true" ]]; then
    FAST_FLAG="--fast_checkpoint"
    echo "  FAST MODE: only evaluate edited batch"
fi

uv run python src/memit_sequential_runner.py \
    --seed "$SEED" \
    --cuda_device "$CUDA_DEVICE" \
    --model_name "$MODEL_NAME" \
    --hparams_fname Llama3-8B.json \
    --ds_name mcf \
    --dataset_size_limit 2000 \
    --num_edits 100 \
    --downstream_eval_steps 10 \
    --conserve_memory \
    --lambda_prev "$LAMBDA_PREV" \
    --lambda_delta "$LAMBDA_DELTA" \
    --cache_strategy "$CACHE_STRATEGY" \
    --cache_max "$CACHE_MAX" \
    $DEBUG_ARG \
    $FAST_FLAG

echo ""
echo "=== MEMIT+SeqReg complete ==="
echo "  Finished: $(date -u +%Y-%m-%dT%H:%M:%SZ)"

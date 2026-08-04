#!/usr/bin/env bash
set -euo pipefail

# MEMIT-Seq: Qwen2.5-7B-Instruct
# Non-projected analogue of AlphaEdit Eq. 12 on Qwen architecture.
#
# Usage:
#   bash scripts/run_memit_sequential_qwen.sh [SEED] [LAMBDA_PREV] [LAMBDA_DELTA]
#   bash scripts/run_memit_sequential_qwen.sh 42 1 1        # Direct Eq. 12 analogue
#   bash scripts/run_memit_sequential_qwen.sh 42 10 1       # Strong prev-key protection
#
# Environment variables:
#   TARGET_EDITS    - Total edits (default: 5000)
#   FAST_CHECKPOINT - If "true", evaluate only edited batch
#   SAVE_INTERVAL   - Checkpoint every N batches (default: 10)

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

# Preserve caller's MODEL_NAME before sourcing .env (which may override it)
_CALLER_MODEL_NAME="${MODEL_NAME:-}"

# Load environment config
if [[ -f "$PROJECT_DIR/.env" ]]; then
    set -a; source "$PROJECT_DIR/.env"; set +a
fi

# Force Qwen for this script (caller override > hardcoded > .env)
MODEL_NAME="${_CALLER_MODEL_NAME:-Qwen/Qwen2.5-7B-Instruct}"

SEED="${1:-42}"
LAMBDA_PREV="${2:-${LAMBDA_PREV:-1}}"
LAMBDA_DELTA="${3:-${LAMBDA_DELTA:-1}}"
TARGET_EDITS="${TARGET_EDITS:-5000}"
CUDA_DEVICE="${CUDA_DEVICE:-0}"
SAVE_INTERVAL="${SAVE_INTERVAL:-10}"
NUM_EDITS=100

echo "=== MEMIT-Seq: Qwen2.5-7B-Instruct ==="
echo "  Model: $MODEL_NAME"
echo "  Seed: $SEED"
echo "  lambda_prev: $LAMBDA_PREV"
echo "  lambda_delta: $LAMBDA_DELTA"
echo "  Target edits: $TARGET_EDITS"
echo "  Started: $(date -u +%Y-%m-%dT%H:%M:%SZ)"

cd "$PROJECT_DIR"

# Build evaluation mode flag
EVAL_FLAG=""
if [[ "${EVAL_AT_CHECKPOINTS_ONLY:-false}" == "true" ]]; then
    EVAL_FLAG="--eval_at_checkpoints_only"
elif [[ "${FAST_CHECKPOINT:-false}" == "true" ]]; then
    EVAL_FLAG="--fast_checkpoint"
fi

uv run python src/runners/memit_sequential_runner.py \
    --seed "$SEED" \
    --cuda_device "$CUDA_DEVICE" \
    --model_name "$MODEL_NAME" \
    --hparams_fname Qwen2.5-7B.json \
    --ds_name mcf \
    --dataset_size_limit "$TARGET_EDITS" \
    --num_edits "$NUM_EDITS" \
    --save_interval "$SAVE_INTERVAL" \
    --lambda_prev "$LAMBDA_PREV" \
    --lambda_delta "$LAMBDA_DELTA" \
    --downstream_eval_steps 10 \
    --conserve_memory \
    $EVAL_FLAG

echo "Completed: $(date -u +%Y-%m-%dT%H:%M:%SZ)"

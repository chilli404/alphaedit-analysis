#!/usr/bin/env bash
set -euo pipefail

# MEMIT-Seq: GPT-J-6B
# Non-projected analogue of AlphaEdit Eq. 12 on GPT-J architecture.
#
# Usage:
#   bash scripts/run_memit_sequential_gptj.sh [SEED] [LAMBDA_PREV] [LAMBDA_DELTA]
#   bash scripts/run_memit_sequential_gptj.sh 42 1 1        # Direct Eq. 12 analogue
#   bash scripts/run_memit_sequential_gptj.sh 42 10 1       # Strong prev-key protection
#
# Environment variables:
#   TARGET_EDITS    - Total edits (default: 5000)
#   FAST_CHECKPOINT - If "true", evaluate only edited batch
#   SAVE_INTERVAL   - Checkpoint every N batches (default: 10)

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

# Load environment config (for HF_TOKEN etc, but NOT MODEL_NAME)
if [[ -f "$PROJECT_DIR/.env" ]]; then
    set -a; source "$PROJECT_DIR/.env"; set +a
fi

# Always GPT-J for this script — ignore .env MODEL_NAME
MODEL_NAME="EleutherAI/gpt-j-6b"

SEED="${1:-42}"
LAMBDA_PREV="${2:-${LAMBDA_PREV:-1}}"
LAMBDA_DELTA="${3:-${LAMBDA_DELTA:-1}}"
TARGET_EDITS="${TARGET_EDITS:-5000}"
CUDA_DEVICE="${CUDA_DEVICE:-0}"
SAVE_INTERVAL="${SAVE_INTERVAL:-10}"
NUM_EDITS=100

echo "=== MEMIT-Seq: GPT-J-6B ==="
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
    --hparams_fname EleutherAI_gpt-j-6B.json \
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

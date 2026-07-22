#!/usr/bin/env bash
set -euo pipefail

# Checkpoint-Based Failure Curve Runner
#
# Runs AlphaEdit or MEMIT up to a target edit count, automatically resuming
# from the latest checkpoint. Designed to fit within 8-hour SkyPilot cluster
# limits by saving model state every save_interval batches.
#
# Usage:
#   bash scripts/run_failure_curve_checkpointed.sh [SEED] [ALG] [TARGET_EDITS]
#   bash scripts/run_failure_curve_checkpointed.sh 42 AlphaEdit 5000
#   bash scripts/run_failure_curve_checkpointed.sh 42 MEMIT 10000
#   bash scripts/run_failure_curve_checkpointed.sh 42 both 5000
#
# Environment variables:
#   TARGET_EDITS    - Override target edit count (default: 5000)
#   SAVE_INTERVAL   - Checkpoint every N batches (default: 10 = every 1000 edits)
#   CHECKPOINT_DIR  - Override checkpoint directory
#   CUDA_DEVICE     - GPU device index (default: 0)
#   MODEL_NAME      - Model to use (default: meta-llama/Meta-Llama-3-8B-Instruct)

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

# Load environment config
if [[ -f "$PROJECT_DIR/.env" ]]; then
    set -a; source "$PROJECT_DIR/.env"; set +a
fi

MODEL_NAME="${MODEL_NAME:-meta-llama/Meta-Llama-3-8B-Instruct}"

SEED="${1:-42}"
ALG="${2:-${ALG_NAME:-both}}"
TARGET_EDITS="${3:-${TARGET_EDITS:-5000}}"
CUDA_DEVICE="${CUDA_DEVICE:-0}"
SAVE_INTERVAL="${SAVE_INTERVAL:-10}"
NUM_EDITS=100  # Edits per batch (matches all other experiments)

echo "=== Checkpoint-Based Failure Curve ==="
echo "  Seed: $SEED"
echo "  Algorithm: $ALG"
echo "  Target edits: $TARGET_EDITS"
echo "  Save interval: every $SAVE_INTERVAL batches ($(( SAVE_INTERVAL * NUM_EDITS )) edits)"
echo "  Started: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo ""

cd "$PROJECT_DIR"

# Build checkpoint_dir arg if CHECKPOINT_DIR is explicitly set (overrides CHECKPOINT_ROOT)
CKPT_ARGS=""
if [[ -n "${CHECKPOINT_DIR:-}" ]]; then
    CKPT_ARGS="--checkpoint_dir $CHECKPOINT_DIR"
fi

run_checkpointed() {
    local alg_name="$1"
    local target="$2"
    local seed="$3"

    echo "--- $alg_name: target $target edits (seed=$seed) ---"

    # Build evaluation mode flag (mutually exclusive)
    EVAL_FLAG=""
    if [[ "${EVAL_AT_CHECKPOINTS_ONLY:-false}" == "true" ]]; then
        EVAL_FLAG="--eval_at_checkpoints_only"
    elif [[ "${FAST_CHECKPOINT:-false}" == "true" ]]; then
        EVAL_FLAG="--fast_checkpoint"
    fi

    uv run python src/runners/checkpoint_runner.py \
        --seed "$seed" \
        --cuda_device "$CUDA_DEVICE" \
        --alg_name "$alg_name" \
        --model_name "$MODEL_NAME" \
        --hparams_fname Llama3-8B.json \
        --ds_name mcf \
        --dataset_size_limit "$target" \
        --num_edits "$NUM_EDITS" \
        --save_interval "$SAVE_INTERVAL" \
        --downstream_eval_steps 10 \
        --conserve_memory \
        $EVAL_FLAG \
        $CKPT_ARGS

    echo "--- $alg_name at $target edits: DONE ---"
    echo ""
}

FAILED=0

case "$ALG" in
    AlphaEdit)
        run_checkpointed "AlphaEdit" "$TARGET_EDITS" "$SEED" || FAILED=$((FAILED + 1))
        ;;
    MEMIT)
        run_checkpointed "MEMIT" "$TARGET_EDITS" "$SEED" || FAILED=$((FAILED + 1))
        ;;
    both)
        run_checkpointed "AlphaEdit" "$TARGET_EDITS" "$SEED" || FAILED=$((FAILED + 1))
        run_checkpointed "MEMIT" "$TARGET_EDITS" "$SEED" || FAILED=$((FAILED + 1))
        ;;
    *)
        echo "ERROR: Unknown algorithm '$ALG'. Use: AlphaEdit, MEMIT, or both"
        exit 1
        ;;
esac

echo "=== Checkpoint failure curve complete ==="
echo "  Finished: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
if [[ $FAILED -gt 0 ]]; then
    echo "  WARNING: $FAILED runs failed (checkpoints preserved for resume)"
fi
echo ""
echo "Re-run the same command to resume from last checkpoint."
echo "Results: vendor/AlphaEdit/results/"

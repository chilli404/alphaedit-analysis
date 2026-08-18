#!/usr/bin/env bash
set -euo pipefail

# Projection Capacity Sweep: GPT-J-6B
#
# Varies nullspace_threshold to control the effective rank of P, then runs
# AlphaEdit (P only) and AlphaEdit-C0 (P + C₀) at each threshold level.
# The MEMIT-Seq baseline (no P, C₀) is threshold-independent and already complete.
#
# Usage:
#   bash scripts/run_projection_sweep_gptj.sh [SEED] [THRESHOLD] [CELL]
#   bash scripts/run_projection_sweep_gptj.sh 42 0.05 AlphaEdit
#   bash scripts/run_projection_sweep_gptj.sh 42 0.05 AlphaEdit-C0
#   bash scripts/run_projection_sweep_gptj.sh 42 0.05 both
#
# Environment variables:
#   NULLSPACE_THRESHOLD  - Override threshold (alternative to positional arg)
#   TARGET_EDITS         - Total edits (default: 10000)
#   INJECT_C0            - If "true", run AlphaEdit-C0 cell
#   EVAL_AT_CHECKPOINTS_ONLY - Recommended for 10K runs

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

export MODEL_NAME="EleutherAI/gpt-j-6b"
export HPARAMS_FNAME="EleutherAI_gpt-j-6B.json"

SEED="${1:-42}"
THRESHOLD="${2:-${NULLSPACE_THRESHOLD:-0.02}}"
CELL="${3:-${SWEEP_CELL:-both}}"
TARGET_EDITS="${TARGET_EDITS:-10000}"
CUDA_DEVICE="${CUDA_DEVICE:-0}"
SAVE_INTERVAL="${SAVE_INTERVAL:-10}"
NUM_EDITS=100

echo "=== Projection Capacity Sweep (GPT-J-6B) ==="
echo "  Seed: $SEED"
echo "  Threshold: $THRESHOLD"
echo "  Cell: $CELL"
echo "  Target edits: $TARGET_EDITS"
echo "  Started: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo ""

cd "$PROJECT_DIR"

# Load environment
if [[ -f "$PROJECT_DIR/.env" ]]; then
    set -a; source "$PROJECT_DIR/.env"; set +a
fi

EVAL_FLAG=""
if [[ "${EVAL_AT_CHECKPOINTS_ONLY:-true}" == "true" ]]; then
    EVAL_FLAG="--eval_at_checkpoints_only"
elif [[ "${FAST_CHECKPOINT:-false}" == "true" ]]; then
    EVAL_FLAG="--fast_checkpoint"
fi

run_cell() {
    local cell_name="$1"
    local c0_args=""

    if [[ "$cell_name" == "AlphaEdit-C0" ]]; then
        c0_args="--inject_c0"
        if [[ -n "${C0_WEIGHT:-}" ]]; then
            c0_args="$c0_args --c0_weight $C0_WEIGHT"
        fi
    fi

    echo "--- $cell_name at threshold=$THRESHOLD (seed=$SEED) ---"

    uv run python src/runners/checkpoint_runner.py \
        --seed "$SEED" \
        --cuda_device "$CUDA_DEVICE" \
        --alg_name AlphaEdit \
        --model_name "$MODEL_NAME" \
        --hparams_fname "$HPARAMS_FNAME" \
        --ds_name mcf \
        --dataset_size_limit "$TARGET_EDITS" \
        --num_edits "$NUM_EDITS" \
        --save_interval "$SAVE_INTERVAL" \
        --downstream_eval_steps 0 \
        --conserve_memory \
        --nullspace_threshold "$THRESHOLD" \
        $EVAL_FLAG \
        $c0_args

    echo "--- $cell_name at threshold=$THRESHOLD: DONE ---"
    echo ""
}

FAILED=0

case "$CELL" in
    AlphaEdit)
        run_cell "AlphaEdit" || FAILED=$((FAILED + 1))
        ;;
    AlphaEdit-C0)
        run_cell "AlphaEdit-C0" || FAILED=$((FAILED + 1))
        ;;
    both)
        run_cell "AlphaEdit" || FAILED=$((FAILED + 1))
        run_cell "AlphaEdit-C0" || FAILED=$((FAILED + 1))
        ;;
    *)
        echo "ERROR: Unknown cell '$CELL'. Use: AlphaEdit, AlphaEdit-C0, or both"
        exit 1
        ;;
esac

echo "=== Projection sweep complete ==="
echo "  Finished: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
if [[ $FAILED -gt 0 ]]; then
    echo "  WARNING: $FAILED cells failed"
fi

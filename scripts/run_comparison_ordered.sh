#!/usr/bin/env bash
# run_comparison_ordered.sh — P2: Single (algorithm, order) comparison run
#
# Runs ONE algorithm at ONE ordering with checkpointing.
# Designed to be launched as separate clusters via sky_launch.sh.
#
# Usage:
#   bash scripts/run_comparison_ordered.sh SEED
#
# Required environment variables:
#   ALG_NAME=AlphaEdit|MEMIT|MEMIT_seq   (which algorithm)
#   ORDER_ID=0|1|2|...                    (which ordering)
#
# Optional environment variables:
#   EVAL_AT_CHECKPOINTS_ONLY=true  (recommended)
#   FAST_CHECKPOINT=true           (for rapid iteration)
#   TARGET_EDITS=3000              (default)
#   LAMBDA_PREV=1                  (for MEMIT_seq)
#   LAMBDA_DELTA=1                 (for MEMIT_seq)
#   SAVE_INTERVAL=10               (default)

set -euo pipefail

SEED="${1:?Usage: $0 SEED}"
TARGET_EDITS="${TARGET_EDITS:-3000}"
ORDER_ID="${ORDER_ID:?ORDER_ID must be set (0, 1, 2, ...)}"
ALG_NAME="${ALG_NAME:?ALG_NAME must be set (AlphaEdit, MEMIT, or MEMIT_seq)}"
LAMBDA_PREV="${LAMBDA_PREV:-1}"
LAMBDA_DELTA="${LAMBDA_DELTA:-1}"
SAVE_INTERVAL="${SAVE_INTERVAL:-10}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# Build eval mode flags
EVAL_FLAGS=""
if [ "${EVAL_AT_CHECKPOINTS_ONLY:-}" = "true" ]; then
    EVAL_FLAGS="--eval_at_checkpoints_only"
elif [ "${FAST_CHECKPOINT:-}" = "true" ]; then
    EVAL_FLAGS="--fast_checkpoint"
fi

echo "========================================================================"
echo "P2: Ordered Comparison Run (single pair)"
echo "  Algorithm:     $ALG_NAME"
echo "  Order ID:      $ORDER_ID"
echo "  Seed:          $SEED"
echo "  Target edits:  $TARGET_EDITS"
echo "  Eval flags:    ${EVAL_FLAGS:-normal}"
echo "  Save interval: $SAVE_INTERVAL"
if [ "$ALG_NAME" = "MEMIT_seq" ]; then
    echo "  Lambda prev:   $LAMBDA_PREV"
    echo "  Lambda delta:  $LAMBDA_DELTA"
fi
echo "========================================================================"

if [ "$ALG_NAME" = "MEMIT_seq" ]; then
    # Use memit_sequential_runner
    uv run python "$PROJECT_ROOT/src/runners/memit_sequential_runner.py" \
        --seed "$SEED" \
        --ds_name mcf \
        --dataset_size_limit "$TARGET_EDITS" \
        --num_edits 100 \
        --lambda_prev "$LAMBDA_PREV" \
        --lambda_delta "$LAMBDA_DELTA" \
        --downstream_eval_steps 10 \
        --order_id "$ORDER_ID" \
        ${EVAL_FLAGS:+--fast_checkpoint} \
        --conserve_memory
else
    # Use checkpoint_runner for AlphaEdit and MEMIT
    # Checkpoints go to comparison_ordered/ namespace to avoid colliding with failure curve
    CKPT_DIR="${CHECKPOINT_DIR:-/s3-data/continual-learning/alphaedit/checkpoints}/comparison_ordered/${ALG_NAME}/seed${SEED}/order${ORDER_ID}"
    uv run python "$PROJECT_ROOT/src/runners/checkpoint_runner.py" \
        --seed "$SEED" \
        --alg_name "$ALG_NAME" \
        --ds_name mcf \
        --dataset_size_limit "$TARGET_EDITS" \
        --num_edits 100 \
        --save_interval "$SAVE_INTERVAL" \
        --downstream_eval_steps 10 \
        --order_id "$ORDER_ID" \
        --checkpoint_dir "$CKPT_DIR" \
        $EVAL_FLAGS \
        --conserve_memory
fi

echo ""
echo "========================================================================"
echo "Completed: $ALG_NAME, order_id=$ORDER_ID, seed=$SEED"
echo "========================================================================"

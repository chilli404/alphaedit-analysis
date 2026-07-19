#!/usr/bin/env bash
# run_factorial_ablation.sh — P5: 4-cell factorial ablation
#
# Runs all 4 cells (MEMIT, MEMIT-seq, MEMIT-seq+ridge, AlphaEdit) on
# the same dataset with the same ordering for direct comparison.
#
# Usage:
#   bash scripts/run_factorial_ablation.sh SEED [CELLS]
#
# Examples:
#   bash scripts/run_factorial_ablation.sh 42           # All 4 cells, 3K edits
#   bash scripts/run_factorial_ablation.sh 42 A,D       # Only vanilla MEMIT + AlphaEdit
#   bash scripts/run_factorial_ablation.sh 42 B,C       # Only MEMIT-seq variants
#
# Environment variables:
#   FAST_CHECKPOINT=true           (recommended for first pass)
#   EVAL_AT_CHECKPOINTS_ONLY=true  (for paper-quality runs)
#   TARGET_EDITS=3000              (default)
#   ORDER_ID=0                     (default)

set -euo pipefail

SEED="${1:?Usage: $0 SEED [CELLS]}"
CELLS="${2:-A,B,C,D}"
TARGET_EDITS="${TARGET_EDITS:-3000}"
ORDER_ID="${ORDER_ID:-0}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# Build eval flags
EVAL_FLAGS=""
if [ "${EVAL_AT_CHECKPOINTS_ONLY:-}" = "true" ]; then
    EVAL_FLAGS="--eval_at_checkpoints_only"
elif [ "${FAST_CHECKPOINT:-}" = "true" ]; then
    EVAL_FLAGS="--fast_checkpoint"
fi

echo "========================================================================"
echo "P5: Factorial Ablation"
echo "  Seed:         $SEED"
echo "  Cells:        $CELLS"
echo "  Target edits: $TARGET_EDITS"
echo "  Order ID:     $ORDER_ID"
echo "  Eval flags:   ${EVAL_FLAGS:-normal}"
echo "========================================================================"

uv run python "$PROJECT_ROOT/src/runners/factorial_ablation_runner.py" \
    --seed "$SEED" \
    --dataset_size_limit "$TARGET_EDITS" \
    --num_edits 100 \
    --save_interval 10 \
    --order_id "$ORDER_ID" \
    --cells "$CELLS" \
    $EVAL_FLAGS \
    --conserve_memory

echo "========================================================================"
echo "Factorial ablation complete."
echo "========================================================================"

#!/usr/bin/env bash
set -euo pipefail

# Edit Order Sensitivity Experiment
#
# Runs the same 2000 MCF edits in 10 different random orderings for both
# AlphaEdit and MEMIT. Measures whether edit ordering affects final performance.
#
# Total: 10 orderings × 2 algorithms = 20 runs (same model seed throughout).
#
# Usage:
#   bash scripts/run_order_sensitivity.sh [SEED] [DATASET_SIZE_LIMIT]
#   bash scripts/run_order_sensitivity.sh 42 2000   # Full run
#   bash scripts/run_order_sensitivity.sh 42 500    # Quick run

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

# Load environment config
if [[ -f "$PROJECT_DIR/.env" ]]; then
    set -a; source "$PROJECT_DIR/.env"; set +a
fi

SEED="${1:-42}"
DATASET_SIZE_LIMIT="${2:-2000}"
CUDA_DEVICE="${CUDA_DEVICE:-0}"
NUM_EDITS="${NUM_EDITS:-100}"
MODEL_NAME="${MODEL_NAME:-meta-llama/Meta-Llama-3-8B-Instruct}"
HPARAMS_FNAME="${HPARAMS_FNAME:-Llama3-8B.json}"
NUM_ORDERINGS="${NUM_ORDERINGS:-10}"

echo "=== Edit Order Sensitivity Experiment ==="
echo "  Model seed: $SEED"
echo "  Orderings: $NUM_ORDERINGS"
echo "  Dataset size: $DATASET_SIZE_LIMIT"
echo "  Num edits per batch: $NUM_EDITS"
echo "  Model: $MODEL_NAME"
echo "  CUDA device: $CUDA_DEVICE"
echo "  Started: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo ""

cd "$PROJECT_DIR"

COMPLETED=0
FAILED=0

for order_seed in $(seq 0 $((NUM_ORDERINGS - 1))); do
    for alg in AlphaEdit MEMIT; do
        echo ""
        echo "--- Running: $alg order_seed=$order_seed ---"
        if uv run python src/order_sensitivity_runner.py \
            --seed "$SEED" \
            --order_seed "$order_seed" \
            --cuda_device "$CUDA_DEVICE" \
            --alg_name "$alg" \
            --model_name "$MODEL_NAME" \
            --hparams_fname "$HPARAMS_FNAME" \
            --ds_name mcf \
            --dataset_size_limit "$DATASET_SIZE_LIMIT" \
            --num_edits "$NUM_EDITS" \
            --downstream_eval_steps 5 \
            --conserve_memory; then
            COMPLETED=$((COMPLETED + 1))
            echo "--- Completed: $alg order_seed=$order_seed ---"
        else
            FAILED=$((FAILED + 1))
            echo "--- FAILED: $alg order_seed=$order_seed ---"
        fi
    done
done

TOTAL=$((NUM_ORDERINGS * 2))
echo ""
echo "=== Order sensitivity experiment complete ==="
echo "  Completed: $COMPLETED / $TOTAL"
echo "  Failed: $FAILED / $TOTAL"
echo "  Results: results/order_sensitivity/"
echo "  Finished: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo ""
echo "Next: python analysis/plots.py  (generates order sensitivity figure)"

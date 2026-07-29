#!/usr/bin/env bash
set -euo pipefail

# Kernel Editor (Polynomial or RBF)
#
# Runs MEMIT or AlphaEdit with kernel-weighted key regularization.
# Replaces K@K^T with K @ (G_kernel * scale) @ K^T in the solve.
#
# ALG_NAME can be "AlphaEdit", "MEMIT", or "both" (runs both sequentially).
# KERNEL_TYPE can be "poly" or "rbf".
#
# Usage:
#   bash scripts/run_polykernel_editor.sh [SEED] [ALG_NAME] [KERNEL_DEGREE] [DATASET_SIZE_LIMIT] [NUM_EDITS]
#   bash scripts/run_polykernel_editor.sh 42 AlphaEdit 2
#   KERNEL_TYPE=rbf KERNEL_SIGMA=median bash scripts/run_polykernel_editor.sh 42 AlphaEdit
#   KERNEL_TYPE=rbf KERNEL_SIGMA=0.5 bash scripts/run_polykernel_editor.sh 42 MEMIT

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

# Load environment config
if [[ -f "$PROJECT_DIR/.env" ]]; then
    set -a; source "$PROJECT_DIR/.env"; set +a
fi

SEED="${1:-42}"
ALG_NAME="${2:-${ALG_NAME:-AlphaEdit}}"
KERNEL_DEGREE="${3:-${KERNEL_DEGREE:-2}}"
DATASET_SIZE_LIMIT="${4:-${DATASET_SIZE_LIMIT:-2000}}"
NUM_EDITS="${5:-${NUM_EDITS:-100}}"
CUDA_DEVICE="${CUDA_DEVICE:-0}"
DOWNSTREAM_EVAL_STEPS="${DOWNSTREAM_EVAL_STEPS:-0}"
KERNEL_TYPE="${KERNEL_TYPE:-poly}"
KERNEL_SIGMA="${KERNEL_SIGMA:-median}"

# Handle "both" by running each algorithm sequentially
if [[ "$ALG_NAME" == "both" ]]; then
    echo "=== ALG_NAME=both: running AlphaEdit then MEMIT ==="
    echo ""
    ALG_NAME=AlphaEdit bash "$0" "$SEED" AlphaEdit "$KERNEL_DEGREE" "$DATASET_SIZE_LIMIT" "$NUM_EDITS"
    ALG_NAME=MEMIT bash "$0" "$SEED" MEMIT "$KERNEL_DEGREE" "$DATASET_SIZE_LIMIT" "$NUM_EDITS"
    echo "=== Both algorithms complete ==="
    exit 0
fi

echo "=== Kernel Editor ==="
echo "  Seed: $SEED"
echo "  Algorithm: $ALG_NAME"
echo "  Kernel type: $KERNEL_TYPE"
if [[ "$KERNEL_TYPE" == "poly" ]]; then
    echo "  Kernel degree: $KERNEL_DEGREE"
else
    echo "  Kernel sigma: $KERNEL_SIGMA"
fi
echo "  Dataset: mcf (limit=$DATASET_SIZE_LIMIT, batch=$NUM_EDITS)"
echo "  Eval steps: $DOWNSTREAM_EVAL_STEPS"
echo "  CUDA device: $CUDA_DEVICE"
echo "  Started: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo ""

cd "$PROJECT_DIR"

# Build extra args for edit_only / eval_only modes
EXTRA_ARGS=""
if [[ "${EDIT_ONLY:-false}" == "true" ]]; then
    EXTRA_ARGS="$EXTRA_ARGS --edit_only --save_interval ${SAVE_INTERVAL:-10}"
    if [[ -n "${CHECKPOINT_DIR:-}" ]]; then
        EXTRA_ARGS="$EXTRA_ARGS --checkpoint_dir $CHECKPOINT_DIR"
    fi
fi
if [[ "${EVAL_ONLY:-false}" == "true" ]]; then
    EXTRA_ARGS="$EXTRA_ARGS --eval_only --load_checkpoint ${LOAD_CHECKPOINT:?LOAD_CHECKPOINT required for eval_only mode}"
fi

uv run python src/polykernel/polykernel_editor_runner.py \
    --seed "$SEED" \
    --alg_name "$ALG_NAME" \
    --kernel_type "$KERNEL_TYPE" \
    --kernel_degree "$KERNEL_DEGREE" \
    --kernel_sigma "$KERNEL_SIGMA" \
    --cuda_device "$CUDA_DEVICE" \
    --ds_name mcf \
    --dataset_size_limit "$DATASET_SIZE_LIMIT" \
    --num_edits "$NUM_EDITS" \
    --downstream_eval_steps "$DOWNSTREAM_EVAL_STEPS" \
    --conserve_memory \
    $EXTRA_ARGS

echo ""
echo "=== Kernel editor complete ($ALG_NAME, type=$KERNEL_TYPE) ==="
echo "  Finished: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo ""

#!/usr/bin/env bash
set -euo pipefail

# Polykernel-augmented MEMIT+SeqReg: kernel-weighted regularization.
#
# Combines sequential regularization (lambda_prev, lambda_delta) with
# kernel-weighted K@K^T terms (polynomial or RBF) to amplify protection
# in crowded key directions.
#
# Usage:
#   bash scripts/run_polykernel_seqreg.sh [SEED] [LAMBDA_PREV] [LAMBDA_DELTA]
#   ORDERING=key_clustered bash scripts/run_polykernel_seqreg.sh 42 1.0 1.0
#   KERNEL_TYPE=rbf KERNEL_SIGMA=median bash scripts/run_polykernel_seqreg.sh 42 1.0 1.0
#   KERNEL_TYPE=poly KERNEL_DEGREE=3 bash scripts/run_polykernel_seqreg.sh 42 10.0 1.0
#   FAST_CHECKPOINT=true TARGET_EDITS=500 bash scripts/run_polykernel_seqreg.sh 42 1.0 1.0

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

# Load environment config
if [[ -f "$PROJECT_DIR/.env" ]]; then
    set -a; source "$PROJECT_DIR/.env"; set +a
fi

SEED="${1:-42}"
LAMBDA_PREV="${2:-${LAMBDA_PREV:-1.0}}"
LAMBDA_DELTA="${3:-${LAMBDA_DELTA:-1.0}}"
KERNEL_TYPE="${KERNEL_TYPE:-poly}"
KERNEL_DEGREE="${KERNEL_DEGREE:-2}"
KERNEL_SIGMA="${KERNEL_SIGMA:-median}"
DATASET_SIZE_LIMIT="${TARGET_EDITS:-2000}"
NUM_EDITS="${NUM_EDITS:-100}"
CUDA_DEVICE="${CUDA_DEVICE:-0}"
DOWNSTREAM_EVAL_STEPS="${DOWNSTREAM_EVAL_STEPS:-0}"
SAVE_INTERVAL="${SAVE_INTERVAL:-10}"
CACHE_STRATEGY="${CACHE_STRATEGY:-all}"
CACHE_MAX="${CACHE_MAX:-none}"
ORDERING="${ORDERING:-}"

# Resolve project-level path env vars
export RESULT_ROOT="${RESULT_ROOT:-$PROJECT_DIR/results}"
export CHECKPOINT_ROOT="${CHECKPOINT_ROOT:-${HOME}/.cache/alphaedit_checkpoints}"

# Build evaluation mode flags
FAST_FLAG=""
if [[ "${FAST_CHECKPOINT:-}" == "true" ]]; then
    FAST_FLAG="--fast_checkpoint"
fi
EVAL_FLAG=""
if [[ "${EVAL_AT_CHECKPOINTS_ONLY:-}" == "true" ]]; then
    EVAL_FLAG="--eval_at_checkpoints_only"
fi

# Resolve stream path if ordering is set
DATASET_OVERRIDE_FLAG=""
if [[ -n "$ORDERING" ]]; then
    STREAM_FILE="${ORDERING}_seed${SEED}.json"
    STREAM_PATH="$RESULT_ROOT/matched_ordering/orderings/$STREAM_FILE"
    if [[ -f "$STREAM_PATH" ]]; then
        DATASET_OVERRIDE_FLAG="--dataset_override $STREAM_PATH --ordering $ORDERING"
    else
        echo "ERROR: Stream not found: $STREAM_PATH"
        echo "Generate with: uv run python src/datasets/generate_orderings.py --seed $SEED"
        exit 1
    fi
fi

echo "=== Polykernel+SeqReg ==="
echo "  Seed: $SEED"
echo "  Kernel: $KERNEL_TYPE (degree=$KERNEL_DEGREE, sigma=$KERNEL_SIGMA)"
echo "  lambda_prev: $LAMBDA_PREV"
echo "  lambda_delta: $LAMBDA_DELTA"
echo "  Cache: strategy=$CACHE_STRATEGY, max=$CACHE_MAX"
echo "  Dataset: mcf (limit=$DATASET_SIZE_LIMIT, batch=$NUM_EDITS)"
echo "  CUDA device: $CUDA_DEVICE"
if [[ -n "$ORDERING" ]]; then
    echo "  Ordering: $ORDERING"
fi
if [[ -n "$FAST_FLAG" ]]; then
    echo "  Mode: fast checkpoint"
elif [[ -n "$EVAL_FLAG" ]]; then
    echo "  Mode: milestone eval (every $SAVE_INTERVAL batches)"
else
    echo "  Mode: full evaluation"
fi
echo "  Started: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo ""

cd "$PROJECT_DIR"

uv run python src/polykernel/polykernel_seqreg_runner.py \
    --seed "$SEED" \
    --cuda_device "$CUDA_DEVICE" \
    --dataset_size_limit "$DATASET_SIZE_LIMIT" \
    --num_edits "$NUM_EDITS" \
    --lambda_prev "$LAMBDA_PREV" \
    --lambda_delta "$LAMBDA_DELTA" \
    --cache_strategy "$CACHE_STRATEGY" \
    --cache_max "$CACHE_MAX" \
    --kernel_type "$KERNEL_TYPE" \
    --kernel_degree "$KERNEL_DEGREE" \
    --kernel_sigma "$KERNEL_SIGMA" \
    --save_interval "$SAVE_INTERVAL" \
    --downstream_eval_steps "$DOWNSTREAM_EVAL_STEPS" \
    --conserve_memory \
    $FAST_FLAG $EVAL_FLAG $DATASET_OVERRIDE_FLAG

echo ""
echo "=== Polykernel+SeqReg complete ==="
echo "  Finished: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo ""

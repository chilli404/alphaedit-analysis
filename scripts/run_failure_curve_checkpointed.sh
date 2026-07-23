#!/usr/bin/env bash
set -euo pipefail

# Checkpoint-Based Failure Curve Runner
#
# Runs AlphaEdit, MEMIT, or MEMIT-Seq up to a target edit count, automatically
# resuming from the latest checkpoint. Designed to fit within 8-hour SkyPilot
# cluster limits by saving model state every save_interval batches.
#
# Usage:
#   bash scripts/run_failure_curve_checkpointed.sh [SEED] [ALG] [TARGET_EDITS]
#   bash scripts/run_failure_curve_checkpointed.sh 42 AlphaEdit 5000
#   bash scripts/run_failure_curve_checkpointed.sh 42 MEMIT 10000
#   bash scripts/run_failure_curve_checkpointed.sh 42 both 5000
#   bash scripts/run_failure_curve_checkpointed.sh 42 MEMIT-Seq-lp1.0-ld0.0-cache0 5000
#
# Environment variables:
#   TARGET_EDITS              - Override target edit count (default: 5000)
#   SAVE_INTERVAL             - Checkpoint every N batches (default: 10 = every 1000 edits)
#   CHECKPOINT_DIR            - Override checkpoint directory
#   CUDA_DEVICE               - GPU device index (default: 0)
#   MODEL_NAME                - Model to use (default: meta-llama/Meta-Llama-3-8B-Instruct)
#   EVAL_AT_CHECKPOINTS_ONLY  - If "true", evaluate only at checkpoint boundaries (RECOMMENDED)
#   FAST_CHECKPOINT           - If "true", evaluate only the edited batch
#   CACHE_STRATEGY            - For MEMIT-Seq: recent|all (default: from ALG name)
#   CACHE_MAX                 - For MEMIT-Seq: max batches in cache (default: from ALG name)
#   DEBUG_BATCH               - For MEMIT-Seq: run same-state diagnostic at this batch

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

run_memit_seq() {
    local alg_name="$1"
    local target="$2"
    local seed="$3"

    echo "--- $alg_name: target $target edits (seed=$seed) ---"

    # Parse λ_prev, λ_delta, cache from ALG name
    # Format: MEMIT-Seq-lp{LP}-ld{LD}-cache{CM}
    local LP LD CM CACHE_MAX_ARG CACHE_STRAT_ARG
    LP=$(echo "$alg_name" | sed -n 's/.*lp\([^-]*\).*/\1/p')
    LD=$(echo "$alg_name" | sed -n 's/.*ld\([^-]*\).*/\1/p')
    CM=$(echo "$alg_name" | sed -n 's/.*cache\(.*\)/\1/p')
    LP="${LP:-1.0}"
    LD="${LD:-0.0}"

    # Allow env var overrides
    LP="${LAMBDA_PREV:-$LP}"
    LD="${LAMBDA_DELTA:-$LD}"

    # cache0 means unlimited (none)
    if [[ "$CM" == "0" ]]; then
        CACHE_MAX_ARG="${CACHE_MAX:-none}"
        CACHE_STRAT_ARG="${CACHE_STRATEGY:-all}"
    else
        CACHE_MAX_ARG="${CACHE_MAX:-$CM}"
        CACHE_STRAT_ARG="${CACHE_STRATEGY:-recent}"
    fi

    # Build evaluation mode flag
    local EVAL_FLAG=""
    if [[ "${EVAL_AT_CHECKPOINTS_ONLY:-false}" == "true" ]]; then
        EVAL_FLAG="--eval_at_checkpoints_only"
    elif [[ "${FAST_CHECKPOINT:-false}" == "true" ]]; then
        EVAL_FLAG="--fast_checkpoint"
    fi

    # Build debug arg if set
    local DEBUG_ARG=""
    if [[ -n "${DEBUG_BATCH:-}" ]]; then
        DEBUG_ARG="--debug_freeze_batch $DEBUG_BATCH"
    fi

    uv run python src/runners/memit_sequential_runner.py \
        --seed "$seed" \
        --cuda_device "$CUDA_DEVICE" \
        --model_name "$MODEL_NAME" \
        --hparams_fname Llama3-8B.json \
        --ds_name mcf \
        --dataset_size_limit "$target" \
        --num_edits "$NUM_EDITS" \
        --downstream_eval_steps 0 \
        --conserve_memory \
        --lambda_prev "$LP" \
        --lambda_delta "$LD" \
        --cache_strategy "$CACHE_STRAT_ARG" \
        --cache_max "$CACHE_MAX_ARG" \
        --save_interval "$SAVE_INTERVAL" \
        $EVAL_FLAG \
        $DEBUG_ARG \
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
    MEMIT-Seq-*)
        run_memit_seq "$ALG" "$TARGET_EDITS" "$SEED" || FAILED=$((FAILED + 1))
        ;;
    *)
        echo "ERROR: Unknown algorithm '$ALG'. Use: AlphaEdit, MEMIT, both, or MEMIT-Seq-lp{X}-ld{Y}-cache{Z}"
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

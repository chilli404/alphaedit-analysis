#!/usr/bin/env bash
set -euo pipefail

# Matched ordering experiment: same 5K facts, clustered vs dispersed.
#
# Runs either AlphaEdit or full-history MEMIT-Seq on the matched ordering
# streams. Designed for SkyPilot parallel execution.
#
# If all checkpoints already exist in the checkpoint directory, automatically
# runs post-hoc evaluation via eval_matched_ordering.py instead of editing.
#
# Usage:
#   bash scripts/run_matched_ordering.sh [SEED] [ALG] [ORDERING]
#   bash scripts/run_matched_ordering.sh 42 MEMIT-Seq-lp1.0-ld0.0-cache0 key_clustered
#   bash scripts/run_matched_ordering.sh 42 MEMIT-Seq-lp1.0-ld0.0-cache0 key_dispersed
#   bash scripts/run_matched_ordering.sh 42 AlphaEdit key_clustered
#   bash scripts/run_matched_ordering.sh 42 AlphaEdit key_dispersed
#
# Environment variables:
#   CUDA_DEVICE      - GPU device index (default: 0)
#   TARGET_EDITS     - Stream length (default: 5000)
#   SAVE_INTERVAL    - Checkpoint save interval (default: 10)
#   FAST_CHECKPOINT  - "true" for fast mode (default: true)
#   EVAL_CHECKPOINTS - Space-separated batch indices to eval (default: auto-detect all)

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

# Load environment config
if [[ -f "$PROJECT_DIR/.env" ]]; then
    set -a; source "$PROJECT_DIR/.env"; set +a
fi

MODEL_NAME="${MODEL_NAME:-meta-llama/Meta-Llama-3-8B-Instruct}"
SEED="${1:-42}"
ALG="${2:-${ALG_NAME:-MEMIT-Seq-lp1.0-ld0.0-cache0}}"
ORDERING="${3:-${ORDERING:-clustered}}"
CUDA_DEVICE="${CUDA_DEVICE:-0}"
DATASET_SIZE_LIMIT="${TARGET_EDITS:-5000}"
SAVE_INTERVAL="${SAVE_INTERVAL:-10}"

# Resolve stream path: use RESULT_ROOT or project results dir
RESULT_ROOT="${RESULT_ROOT:-$PROJECT_DIR/results}"
STREAM_DIR="$RESULT_ROOT/matched_ordering/orderings"
STREAM_FILE="${ORDERING}_seed${SEED}.json"

if [[ -f "$STREAM_DIR/$STREAM_FILE" ]]; then
    STREAM_PATH="$STREAM_DIR/$STREAM_FILE"
else
    echo "ERROR: Stream file not found for ordering=$ORDERING seed=$SEED"
    echo "  Tried: $STREAM_DIR/$STREAM_FILE"
    echo ""
    echo "  Generate with: uv run python src/datasets/generate_orderings.py --seed $SEED"
    exit 1
fi

# Resolve checkpoint and results dirs from CHECKPOINT_ROOT / RESULT_ROOT
CHECKPOINT_ROOT="${CHECKPOINT_ROOT:-${HOME}/.cache/alphaedit_checkpoints}"
CKPT_DIR="$CHECKPOINT_ROOT/matched_ordering/${ALG}/${ORDERING}/seed${SEED}"
mkdir -p "$CKPT_DIR"

RESULTS_DIR="$RESULT_ROOT/matched_ordering/${ALG}/${ORDERING}/seed${SEED}"
mkdir -p "$RESULTS_DIR"

echo "=== Matched Ordering Experiment ==="
echo "  Seed:       $SEED"
echo "  Algorithm:  $ALG"
echo "  Ordering:   $ORDERING"
echo "  Stream:     $STREAM_PATH"
echo "  Checkpoint: $CKPT_DIR"
echo "  Results:    $RESULTS_DIR"
echo "  Edits:      $DATASET_SIZE_LIMIT"
echo "  Started:    $(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo ""

cd "$PROJECT_DIR"

NUM_EDITS=100
TOTAL_BATCHES=$((DATASET_SIZE_LIMIT / NUM_EDITS))
FINAL_BATCH_IDX=$((TOTAL_BATCHES - 1))

# Check if all checkpoints exist — if so, run eval instead of editing
all_checkpoints_present=true
AVAILABLE_BATCHES=()
for ((b=SAVE_INTERVAL-1; b<TOTAL_BATCHES; b+=SAVE_INTERVAL)); do
    if [[ -d "$CKPT_DIR/batch_${b}" ]] && [[ -f "$CKPT_DIR/batch_${b}/model_weights.pt" ]]; then
        AVAILABLE_BATCHES+=("$b")
    else
        all_checkpoints_present=false
    fi
done

if [[ "$all_checkpoints_present" == "true" ]] && [[ ${#AVAILABLE_BATCHES[@]} -gt 0 ]]; then
    echo "  All ${#AVAILABLE_BATCHES[@]} checkpoints found — running post-hoc evaluation"
    echo "  Checkpoints: ${AVAILABLE_BATCHES[*]}"

    # Use explicit list or default to all available
    EVAL_BATCHES="${EVAL_CHECKPOINTS:-${AVAILABLE_BATCHES[*]}}"

    # Use the STREAM file as dataset — it contains the actual edited records in order.
    # Using raw multi_counterfact.json would evaluate the wrong facts since the stream
    # selects and reorders records from across the full 20K+ MCF dataset.
    uv run python scripts/eval_matched_ordering.py \
        --seed "$SEED" \
        --alg_name "$ALG" \
        --ordering "$ORDERING" \
        --model_name "$MODEL_NAME" \
        --checkpoint_dir "$CKPT_DIR" \
        --checkpoints $EVAL_BATCHES \
        --num_edits "$NUM_EDITS" \
        --dataset_path "$STREAM_PATH"

    echo ""
    echo "=== Matched Ordering eval complete ==="
    echo "  Algorithm: $ALG"
    echo "  Ordering:  $ORDERING"
    echo "  Results:   $PROJECT_DIR/results/matched_ordering/$ALG/$ORDERING/seed$SEED/"
    echo "  Finished:  $(date -u +%Y-%m-%dT%H:%M:%SZ)"
    exit 0
fi

# --- Normal edit mode (checkpoints incomplete or missing) ---

FAST_FLAG=""
if [[ "${FAST_CHECKPOINT:-true}" == "true" ]]; then
    FAST_FLAG="--fast_checkpoint"
    echo "  FAST MODE: only evaluate edited batch"
fi

if [[ "$ALG" == MEMIT-Seq-* ]]; then
    # MEMIT-Seq variant: parse λ_prev, λ_delta, cache from ALG name
    # Format: MEMIT-Seq-lp{LP}-ld{LD}-cache{CM}
    LP=$(echo "$ALG" | sed -n 's/.*lp\([^-]*\).*/\1/p')
    LD=$(echo "$ALG" | sed -n 's/.*ld\([^-]*\).*/\1/p')
    CM=$(echo "$ALG" | sed -n 's/.*cache\(.*\)/\1/p')
    LP="${LP:-1.0}"
    LD="${LD:-0.0}"
    # cache0 means unlimited (none)
    if [[ "$CM" == "0" ]]; then
        CACHE_MAX="none"
        CACHE_STRATEGY="all"
    else
        CACHE_MAX="$CM"
        CACHE_STRATEGY="recent"
    fi

    uv run python src/runners/memit_sequential_runner.py \
        --seed "$SEED" \
        --cuda_device "$CUDA_DEVICE" \
        --model_name "$MODEL_NAME" \
        --hparams_fname Llama3-8B.json \
        --ds_name mcf \
        --dataset_size_limit "$DATASET_SIZE_LIMIT" \
        --num_edits "$NUM_EDITS" \
        --downstream_eval_steps 0 \
        --conserve_memory \
        --lambda_prev "$LP" \
        --lambda_delta "$LD" \
        --cache_strategy "$CACHE_STRATEGY" \
        --cache_max "$CACHE_MAX" \
        --save_interval "$SAVE_INTERVAL" \
        --ordering "$ORDERING" \
        --dataset_override "$STREAM_PATH" \
        $FAST_FLAG

elif [[ "$ALG" == "AlphaEdit" ]]; then
    # AlphaEdit via alphaedit_stream_runner (checkpointed stream editing with mechanism measurement)
    uv run python src/runners/alphaedit_stream_runner.py \
        --seed "$SEED" \
        --cuda_device "$CUDA_DEVICE" \
        --model_name "$MODEL_NAME" \
        --stream_length "$DATASET_SIZE_LIMIT" \
        --num_edits "$NUM_EDITS" \
        --save_interval "$SAVE_INTERVAL" \
        --stream "$ORDERING" \
        --stream_path "$STREAM_PATH" \
        --checkpoint_base "$CKPT_DIR" \
        ${FAST_CHECKPOINT:+--eval_at_checkpoints_only}

else
    echo "ERROR: Unknown algorithm '$ALG'. Use 'AlphaEdit' or 'MEMIT-Seq-lp{LP}-ld{LD}-cache{CM}'."
    exit 1
fi

# Copy results to S3 output dir if on cluster
# memit_sequential_runner now writes directly to matched_ordering/ structure;
# just sync to S3 results dir if available.
_LOCAL_RESULTS="$PROJECT_DIR/results/matched_ordering/${ORDERING}/seed${SEED}/${DATASET_SIZE_LIMIT}edits"
if [[ -d "/s3-data/continual-learning/alphaedit" ]] && [[ -d "$_LOCAL_RESULTS" ]]; then
    mkdir -p "$RESULTS_DIR"
    cp -r "$_LOCAL_RESULTS"/* "$RESULTS_DIR/" 2>/dev/null || true
fi

echo ""
echo "=== Matched Ordering complete ==="
echo "  Algorithm: $ALG"
echo "  Ordering:  $ORDERING"
echo "  Results:   $RESULTS_DIR"
echo "  Finished:  $(date -u +%Y-%m-%dT%H:%M:%SZ)"

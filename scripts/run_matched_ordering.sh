#!/usr/bin/env bash
set -euo pipefail

# Matched ordering experiment: same 5K facts, clustered vs dispersed.
#
# Runs either AlphaEdit or full-history MEMIT-Seq on the matched ordering
# streams. Designed for SkyPilot parallel execution.
#
# If all checkpoints already exist in the checkpoint directory, automatically
# runs post-hoc evaluation via eval_seqreg_checkpoints.py instead of editing.
#
# Usage:
#   bash scripts/run_matched_ordering.sh [SEED] [ALG] [ORDERING]
#   bash scripts/run_matched_ordering.sh 42 MEMIT-Seq-1-0 clustered
#   bash scripts/run_matched_ordering.sh 42 MEMIT-Seq-1-0 dispersed
#   bash scripts/run_matched_ordering.sh 42 AlphaEdit clustered
#   bash scripts/run_matched_ordering.sh 42 AlphaEdit dispersed
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
ALG="${2:-${ALG_NAME:-MEMIT-Seq-1-0}}"
ORDERING="${3:-${ORDERING:-clustered}}"
CUDA_DEVICE="${CUDA_DEVICE:-0}"
DATASET_SIZE_LIMIT="${TARGET_EDITS:-5000}"
SAVE_INTERVAL="${SAVE_INTERVAL:-10}"

# Resolve stream path: prefer S3 mount, fall back to local
S3_MATCHED="/s3-data/continual-learning/alphaedit/dsets/matched_ordering/orderings"
S3_MATCHED_LEGACY="/s3-data/continual-learning/alphaedit/dsets/matched_ordering"
LOCAL_MATCHED="$PROJECT_DIR/results/matched_ordering/orderings"

if [[ -f "$S3_MATCHED/${ORDERING}_seed${SEED}.json" ]]; then
    STREAM_PATH="$S3_MATCHED/${ORDERING}_seed${SEED}.json"
elif [[ -f "$S3_MATCHED_LEGACY/${ORDERING}_seed${SEED}.json" ]]; then
    STREAM_PATH="$S3_MATCHED_LEGACY/${ORDERING}_seed${SEED}.json"
elif [[ -f "$LOCAL_MATCHED/${ORDERING}_seed${SEED}.json" ]]; then
    STREAM_PATH="$LOCAL_MATCHED/${ORDERING}_seed${SEED}.json"
else
    echo "ERROR: Stream file not found for ordering=$ORDERING seed=$SEED"
    echo "  Tried: $S3_MATCHED/${ORDERING}_seed${SEED}.json"
    echo "  Tried: $LOCAL_MATCHED/${ORDERING}_seed${SEED}.json"
    echo ""
    echo "  Generate with: uv run python src/datasets/generate_orderings.py --seed $SEED"
    exit 1
fi

# Resolve checkpoint dir: prefer S3 for crash resilience
# Convention: checkpoints/matched_ordering/{ALG}/{ORDERING}/seed{N}
S3_CKPT="/s3-data/continual-learning/alphaedit/checkpoints/matched_ordering/${ALG}/${ORDERING}/seed${SEED}"
LOCAL_CKPT="$HOME/.cache/alphaedit_checkpoints/matched_ordering/${ALG}/${ORDERING}/seed${SEED}"

if [[ -d "/s3-data/continual-learning/alphaedit" ]]; then
    CKPT_DIR="$S3_CKPT"
    mkdir -p "$CKPT_DIR"
else
    CKPT_DIR="$LOCAL_CKPT"
    mkdir -p "$CKPT_DIR"
fi

# Results output (always to S3 if available)
S3_RESULTS="/s3-data/continual-learning/alphaedit/results/matched_ordering/${ALG}/${ORDERING}/seed${SEED}"
LOCAL_RESULTS="$PROJECT_DIR/results/matched_ordering/${ALG}/${ORDERING}/seed${SEED}"

if [[ -d "/s3-data/continual-learning/alphaedit" ]]; then
    RESULTS_DIR="$S3_RESULTS"
else
    RESULTS_DIR="$LOCAL_RESULTS"
fi
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
    uv run python scripts/eval_seqreg_checkpoints.py \
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

if [[ "$ALG" == "MEMIT-Seq-1-0" ]]; then
    # Full-history MEMIT-seq (λ_prev=1, λ_delta=0, unlimited cache)
    uv run python src/runners/memit_sequential_runner.py \
        --seed "$SEED" \
        --cuda_device "$CUDA_DEVICE" \
        --model_name "$MODEL_NAME" \
        --hparams_fname Llama3-8B.json \
        --ds_name mcf \
        --dataset_size_limit "$DATASET_SIZE_LIMIT" \
        --num_edits "$NUM_EDITS" \
        --downstream_eval_steps 10 \
        --conserve_memory \
        --lambda_prev 1.0 \
        --lambda_delta 0.0 \
        --cache_strategy all \
        --cache_max none \
        --save_interval "$SAVE_INTERVAL" \
        --checkpoint_dir "$CKPT_DIR" \
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
    echo "ERROR: Unknown algorithm '$ALG'. Use 'AlphaEdit' or 'MEMIT-Seq-1-0'."
    exit 1
fi

# Copy results to output dir (check hierarchical layout first, then flat legacy)
_SEQREG_BASE="$PROJECT_DIR/results/memit_seqreg"
_SEQREG_HIER="$_SEQREG_BASE/seed${SEED}/${DATASET_SIZE_LIMIT}edits/MEMIT-SEQ-lp1.0-ld0.0-cache0"
if [[ -d "$_SEQREG_HIER" ]]; then
    cp -r "$_SEQREG_HIER"/log_seed${SEED}_lp1.0_ld0.0_*.jsonl "$RESULTS_DIR/" 2>/dev/null || true
    cp -r "$_SEQREG_HIER"/metadata_seed${SEED}_*.json "$RESULTS_DIR/" 2>/dev/null || true
elif [[ -d "$_SEQREG_BASE" ]]; then
    cp -r "$_SEQREG_BASE"/log_seed${SEED}_lp1.0_ld0.0_*.jsonl "$RESULTS_DIR/" 2>/dev/null || true
    cp -r "$_SEQREG_BASE"/metadata_seed${SEED}_*.json "$RESULTS_DIR/" 2>/dev/null || true
fi

echo ""
echo "=== Matched Ordering complete ==="
echo "  Algorithm: $ALG"
echo "  Ordering:  $ORDERING"
echo "  Results:   $RESULTS_DIR"
echo "  Finished:  $(date -u +%Y-%m-%dT%H:%M:%SZ)"

#!/usr/bin/env bash
set -euo pipefail

# ZsRE Ordering Experiment: replicate key-geometry ordering effect on ZsRE.
#
# Same pipeline as run_matched_ordering.sh but using ZsRE dataset and streams.
# Demonstrates the memory-space mechanism generalizes beyond MultiCounterFact.
#
# Usage:
#   bash scripts/run_zsre_ordering_experiment.sh [SEED] [ALG] [ORDERING]
#   bash scripts/run_zsre_ordering_experiment.sh 42 AlphaEdit key_clustered
#   bash scripts/run_zsre_ordering_experiment.sh 42 AlphaEdit key_dispersed
#   bash scripts/run_zsre_ordering_experiment.sh 42 MEMIT-Seq-lp1.0-ld0.0-cache0 key_clustered
#   bash scripts/run_zsre_ordering_experiment.sh 42 MEMIT-Seq-lp1.0-ld0.0-cache0 key_dispersed
#
# Environment variables:
#   CUDA_DEVICE      - GPU device index (default: 0)
#   TARGET_EDITS     - Stream length (default: 5000)
#   SAVE_INTERVAL    - Checkpoint save interval (default: 10)
#   FAST_CHECKPOINT  - "true" for fast mode (default: true)

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

# Load environment config
if [[ -f "$PROJECT_DIR/.env" ]]; then
    set -a; source "$PROJECT_DIR/.env"; set +a
fi

MODEL_NAME="${MODEL_NAME:-meta-llama/Meta-Llama-3-8B-Instruct}"
SEED="${1:-42}"
ALG="${2:-${ALG_NAME:-AlphaEdit}}"
ORDERING="${3:-${ORDERING:-key_clustered}}"
CUDA_DEVICE="${CUDA_DEVICE:-0}"
DATASET_SIZE_LIMIT="${TARGET_EDITS:-10000}"
SAVE_INTERVAL="${SAVE_INTERVAL:-10}"

# Resolve stream path
RESULT_ROOT="${RESULT_ROOT:-$PROJECT_DIR/results}"
STREAM_DIR="$RESULT_ROOT/matched_ordering_zsre/orderings"
STREAM_FILE="${ORDERING}_seed${SEED}.json"

if [[ -f "$STREAM_DIR/$STREAM_FILE" ]]; then
    STREAM_PATH="$STREAM_DIR/$STREAM_FILE"
else
    # Auto-generate orderings if keys exist but streams don't
    KEYS_PATH="$RESULT_ROOT/key_vectors/zsre/keys_seed${SEED}_layer6.npz"
    if [[ -f "$KEYS_PATH" ]]; then
        echo "  Stream not found — generating orderings from existing keys..."
        uv run python src/datasets/generate_zsre_orderings.py \
            --seed "$SEED" \
            --keys_path "$KEYS_PATH" \
            --output_dir "$RESULT_ROOT/matched_ordering_zsre"
        if [[ -f "$STREAM_DIR/$STREAM_FILE" ]]; then
            STREAM_PATH="$STREAM_DIR/$STREAM_FILE"
        else
            echo "ERROR: Ordering generation did not produce expected file"
            exit 1
        fi
    else
        echo "ERROR: Stream file not found for ordering=$ORDERING seed=$SEED"
        echo "  Tried: $STREAM_DIR/$STREAM_FILE"
        echo "  Keys also missing: $KEYS_PATH"
        echo ""
        echo "  Run key extraction first, then generate orderings:"
        echo "    bash scripts/run_zsre_key_extraction.sh $SEED"
        echo "    uv run python src/datasets/generate_zsre_orderings.py --seed $SEED"
        exit 1
    fi
fi

# Resolve checkpoint and results dirs
CHECKPOINT_ROOT="${CHECKPOINT_ROOT:-${HOME}/.cache/alphaedit_checkpoints}"
CKPT_DIR="$CHECKPOINT_ROOT/matched_ordering_zsre/${ALG}/${ORDERING}/seed${SEED}"
mkdir -p "$CKPT_DIR"

RESULTS_DIR="$RESULT_ROOT/matched_ordering_zsre/${ALG}/${ORDERING}/seed${SEED}"
mkdir -p "$RESULTS_DIR"

echo "=== ZsRE Ordering Experiment ==="
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

# Check if all checkpoints exist — if so, run eval
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
    EVAL_BATCHES="${EVAL_CHECKPOINTS:-${AVAILABLE_BATCHES[*]}}"

    # eval_matched_ordering.py writes to $RESULT_ROOT/matched_ordering/{alg}/{ordering}/...
    # We use a temporary RESULT_ROOT so output lands under matched_ordering_zsre/ instead
    _EVAL_RESULT_ROOT="$RESULT_ROOT/_zsre_eval_tmp"
    RESULT_ROOT="$_EVAL_RESULT_ROOT" \
    uv run python scripts/eval_matched_ordering.py \
        --seed "$SEED" \
        --alg_name "$ALG" \
        --ordering "$ORDERING" \
        --model_name "$MODEL_NAME" \
        --checkpoint_dir "$CKPT_DIR" \
        --checkpoints $EVAL_BATCHES \
        --num_edits "$NUM_EDITS" \
        --dataset_path "$STREAM_PATH"

    # Move result to the correct ZsRE results directory
    _EVAL_OUT="$_EVAL_RESULT_ROOT/matched_ordering/${ALG}/${ORDERING}/seed${SEED}/full_eval_seed${SEED}.json"
    if [[ -f "$_EVAL_OUT" ]]; then
        cp "$_EVAL_OUT" "$RESULTS_DIR/full_eval_seed${SEED}.json"
        rm -rf "$_EVAL_RESULT_ROOT"
        echo "  Results: $RESULTS_DIR/full_eval_seed${SEED}.json"
    fi

    echo ""
    echo "=== ZsRE Ordering eval complete ==="
    echo "  Finished: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
    exit 0
fi

# --- Normal edit mode ---

FAST_FLAG=""
if [[ "${FAST_CHECKPOINT:-true}" == "true" ]]; then
    FAST_FLAG="--fast_checkpoint"
    echo "  FAST MODE: only evaluate edited batch"
fi

if [[ "$ALG" == MEMIT-Seq-* ]]; then
    LP=$(echo "$ALG" | sed -n 's/.*lp\([^-]*\).*/\1/p')
    LD=$(echo "$ALG" | sed -n 's/.*ld\([^-]*\).*/\1/p')
    CM=$(echo "$ALG" | sed -n 's/.*cache\(.*\)/\1/p')
    LP="${LP:-1.0}"
    LD="${LD:-0.0}"
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
        --ds_name zsre \
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
    # Use checkpoint_runner (not alphaedit_stream_runner) for ZsRE because:
    # - alphaedit_stream_runner doesn't support --ds_name
    # - ZsRE requires ds_name="zsre" for the correct eval function
    EVAL_FLAG=""
    if [[ "${FAST_CHECKPOINT:-true}" == "true" ]]; then
        EVAL_FLAG="--fast_checkpoint"
    fi

    uv run python src/runners/checkpoint_runner.py \
        --seed "$SEED" \
        --cuda_device "$CUDA_DEVICE" \
        --alg_name AlphaEdit \
        --model_name "$MODEL_NAME" \
        --hparams_fname Llama3-8B.json \
        --ds_name zsre \
        --dataset_size_limit "$DATASET_SIZE_LIMIT" \
        --num_edits "$NUM_EDITS" \
        --save_interval "$SAVE_INTERVAL" \
        --downstream_eval_steps 0 \
        --conserve_memory \
        --checkpoint_dir "$CKPT_DIR" \
        --dataset_override "$STREAM_PATH" \
        $EVAL_FLAG

else
    echo "ERROR: Unknown algorithm '$ALG'."
    exit 1
fi

# Copy results to RESULT_ROOT if different from local
_REMOTE_RESULTS="${RESULT_ROOT:-}/matched_ordering_zsre/${ALG}/${ORDERING}/seed${SEED}"
if [[ -n "${RESULT_ROOT:-}" ]] && [[ "$_REMOTE_RESULTS" != "$RESULTS_DIR" ]]; then
    mkdir -p "$_REMOTE_RESULTS"
    cp -r "$RESULTS_DIR"/* "$_REMOTE_RESULTS/" 2>/dev/null || true
fi

echo ""
echo "=== ZsRE Ordering experiment complete ==="
echo "  Algorithm: $ALG"
echo "  Ordering:  $ORDERING"
echo "  Results:   $RESULTS_DIR"
echo "  Finished:  $(date -u +%Y-%m-%dT%H:%M:%SZ)"

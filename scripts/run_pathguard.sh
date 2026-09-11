#!/usr/bin/env bash
set -euo pipefail

# PathGuard: Hazard-Gated Displacement-Constrained Sequential Editing
#
# Usage:
#   bash scripts/run_pathguard.sh SEED [VARIANT] [ORDERING]
#
# Variants:
#   PathGuard-E       — exposure covariance only (fixed lambda)
#   PathGuard-ED      — + adaptive displacement budget
#   PathGuard-EDS     — + signed margin shield (full method)
#   PathGuard-ED-random — random selection (negative control)
#
# Examples:
#   bash scripts/run_pathguard.sh 42 PathGuard-ED fb_high_exposure
#   bash scripts/run_pathguard.sh 42 PathGuard-EDS fb_random0
#   bash scripts/run_pathguard.sh 42 PathGuard-E   # canonical ordering

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

_PRESET_MODEL="${MODEL_NAME:-}"
_PRESET_HPARAMS="${HPARAMS_FNAME:-}"

if [[ -f "$PROJECT_DIR/.env" ]]; then
    set -a; source "$PROJECT_DIR/.env"; set +a
fi

if [[ -n "$_PRESET_MODEL" ]]; then MODEL_NAME="$_PRESET_MODEL"; fi
if [[ -n "$_PRESET_HPARAMS" ]]; then HPARAMS_FNAME="$_PRESET_HPARAMS"; fi

SEED="${1:?Usage: $0 SEED [VARIANT] [ORDERING]}"
VARIANT="${2:-PathGuard-ED}"
ORDERING="${3:-}"

MODEL_NAME="${MODEL_NAME:-meta-llama/Meta-Llama-3-8B-Instruct}"
HPARAMS_FNAME="${HPARAMS_FNAME:-Llama3-8B.json}"
TARGET_EDITS="${TARGET_EDITS:-10000}"
SAVE_INTERVAL="${SAVE_INTERVAL:-10}"
DEVICE="${CUDA_DEVICE:-0}"
RESULT_ROOT="${RESULT_ROOT:-$PROJECT_DIR/results}"
CHECKPOINT_ROOT="${CHECKPOINT_ROOT:-${HOME}/.cache/alphaedit_checkpoints}"

# PathGuard hyperparameters
PG_M="${PATHGUARD_M:-200}"
PG_Q="${PATHGUARD_Q:-10}"
PG_EPSILON="${PATHGUARD_EPSILON:-0.1}"
PG_LAMBDA_CANDIDATES="${PATHGUARD_LAMBDA_CANDIDATES:-0.1,0.5,1.0,5.0,10.0,50.0,100.0}"
PG_FIXED_LAMBDA="${PATHGUARD_FIXED_LAMBDA:-1.0}"
PG_KERNEL_DEGREE="${PATHGUARD_KERNEL_DEGREE:-0}"

# MEMIT-Seq base parameters
LAMBDA_PREV="${LAMBDA_PREV:-1.0}"
LAMBDA_DELTA="${LAMBDA_DELTA:-0.0}"

# Determine representative layer based on model
case "$MODEL_NAME" in
    *gpt-j*|*GPT-J*) PG_REPR_LAYER=5 ;;
    *)                PG_REPR_LAYER=6 ;;
esac

# Parse variant into CLI flags
PG_FLAGS=""
case "$VARIANT" in
    PathGuard-E)
        PG_FLAGS="--pathguard --pathguard_fixed_lambda $PG_FIXED_LAMBDA --pathguard_no_margin_shield"
        ;;
    PathGuard-ED)
        PG_FLAGS="--pathguard --pathguard_adaptive --pathguard_no_margin_shield"
        ;;
    PathGuard-EDS)
        PG_FLAGS="--pathguard --pathguard_adaptive"
        ;;
    PathGuard-ED-random)
        PG_FLAGS="--pathguard --pathguard_adaptive --pathguard_random --pathguard_no_margin_shield"
        ;;
    PathGuard-EDS-noadapt)
        PG_FLAGS="--pathguard --pathguard_fixed_lambda $PG_FIXED_LAMBDA"
        ;;
    *)
        echo "ERROR: Unknown variant '$VARIANT'"
        echo "Valid: PathGuard-E, PathGuard-ED, PathGuard-EDS, PathGuard-ED-random, PathGuard-EDS-noadapt"
        exit 1
        ;;
esac

# Signed diagnostic / signed-informed lambda (from env vars)
if [[ "${PATHGUARD_SIGNED_DIAG:-}" == "true" ]]; then
    PG_FLAGS="$PG_FLAGS --pathguard_signed_diag"
fi
if [[ "${PATHGUARD_SIGNED_LAMBDA:-}" == "true" ]]; then
    PG_FLAGS="$PG_FLAGS --pathguard_signed_lambda"
fi

# Resolve keys directory (check multiple conventions)
KEYS_DIR="${PATHGUARD_KEYS_DIR:-$RESULT_ROOT/key_vectors/full_mcf}"
if [[ ! -f "$KEYS_DIR/keys_seed${SEED}_layer${PG_REPR_LAYER}.npz" ]]; then
    # Try S3/alternate layout: key_vectors/layer{N}/seed{S}/keys_seed{S}.npz
    ALT_KEY="$RESULT_ROOT/key_vectors/layer${PG_REPR_LAYER}/seed${SEED}/keys_seed${SEED}.npz"
    if [[ -f "$ALT_KEY" ]]; then
        KEYS_DIR="$RESULT_ROOT/key_vectors/full_mcf"
        echo "INFO: Keys found in alternate layout (layer{N}/seed{S}/) — runner will resolve"
    else
        echo "WARNING: Precomputed keys not found"
        echo "  Tried: $KEYS_DIR/keys_seed${SEED}_layer${PG_REPR_LAYER}.npz"
        echo "  Tried: $ALT_KEY"
        echo "Run: bash scripts/run_multilayer_key_extraction.sh $SEED"
    fi
fi

# Resolve ordering
ORDERING_FLAGS=""
if [[ -n "$ORDERING" ]]; then
    STREAM_PATH="$RESULT_ROOT/matched_ordering/orderings/${ORDERING}_seed${SEED}.json"
    if [[ ! -f "$STREAM_PATH" ]]; then
        echo "ERROR: Ordering stream not found: $STREAM_PATH"
        echo "Run: uv run python src/datasets/generate_orderings.py --seed $SEED --fixed_batch"
        exit 1
    fi
    ORDERING_FLAGS="--ordering $ORDERING --dataset_override $STREAM_PATH"
fi

echo "═══════════════════════════════════════════════════════════════"
echo "PathGuard Runner"
echo "  Seed:        $SEED"
echo "  Variant:     $VARIANT"
echo "  Ordering:    ${ORDERING:-canonical}"
echo "  Model:       $MODEL_NAME"
echo "  Target:      $TARGET_EDITS edits"
echo "  PathGuard M: $PG_M"
echo "  PathGuard q: $PG_Q"
echo "  Epsilon:     $PG_EPSILON"
echo "  Lambda prev: $LAMBDA_PREV"
echo "  Keys dir:    $KEYS_DIR"
echo "═══════════════════════════════════════════════════════════════"

export RESULT_ROOT
export CHECKPOINT_ROOT

uv run python src/runners/pathguard_runner.py \
    --seed "$SEED" \
    --cuda_device "$DEVICE" \
    --model_name "$MODEL_NAME" \
    --hparams_fname "$HPARAMS_FNAME" \
    --ds_name mcf \
    --dataset_size_limit "$TARGET_EDITS" \
    --num_edits 100 \
    --downstream_eval_steps 10 \
    --conserve_memory \
    --lambda_prev "$LAMBDA_PREV" \
    --lambda_delta "$LAMBDA_DELTA" \
    --cache_strategy all \
    --cache_max none \
    --save_interval "$SAVE_INTERVAL" \
    --eval_at_checkpoints_only \
    --pathguard_M "$PG_M" \
    --pathguard_q "$PG_Q" \
    --pathguard_epsilon_init "$PG_EPSILON" \
    --pathguard_lambda_candidates "$PG_LAMBDA_CANDIDATES" \
    --pathguard_keys_dir "$KEYS_DIR" \
    --pathguard_representative_layer "$PG_REPR_LAYER" \
    --pathguard_kernel_degree "$PG_KERNEL_DEGREE" \
    $PG_FLAGS \
    $ORDERING_FLAGS \
    ${CONTINUE_FROM_RUN:+--continue_from_run "$CONTINUE_FROM_RUN"}

echo ""
echo "═══════════════════════════════════════════════════════════════"
echo "PathGuard complete: $VARIANT seed $SEED ${ORDERING:-canonical}"
echo "═══════════════════════════════════════════════════════════════"

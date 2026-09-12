#!/usr/bin/env bash
set -euo pipefail

# Run MEMIT+REVIVE baseline on fixed-batch orderings.
# REVIVE is a plugin that projects ΔW to remove components in the dominant
# singular directions of the current weights W. Applied post-hoc to MEMIT deltas.
#
# Two modes:
#   LAMBDA_PREV=0 LAMBDA_DELTA=0  → Plain MEMIT + REVIVE (matches REVIVE paper)
#   LAMBDA_PREV=1 LAMBDA_DELTA=0  → MEMIT-Seq + REVIVE (history-aware + spectral)
#
# Usage:
#   bash scripts/run_revive_baseline.sh SEED [ORDERING]
#   LAMBDA_PREV=0 LAMBDA_DELTA=0 bash scripts/run_revive_baseline.sh 42 fb_high_exposure

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

_PRESET_MODEL="${MODEL_NAME:-}"
if [[ -f "$PROJECT_DIR/.env" ]]; then
    set -a; source "$PROJECT_DIR/.env"; set +a
fi
if [[ -n "$_PRESET_MODEL" ]]; then MODEL_NAME="$_PRESET_MODEL"; fi

SEED="${1:?Usage: $0 SEED [ORDERING]}"
ORDERING="${2:-${ORDERING:-fb_random0}}"
MODEL_NAME="${MODEL_NAME:-meta-llama/Meta-Llama-3-8B-Instruct}"
case "$MODEL_NAME" in
    *gpt-j*|*gptj*|*EleutherAI*) HPARAMS_FNAME="EleutherAI_gpt-j-6B.json" ;;
    *Qwen*|*qwen*)               HPARAMS_FNAME="Qwen2.5-7B.json" ;;
    *)                           HPARAMS_FNAME="Llama3-8B.json" ;;
esac
TARGET_EDITS="${TARGET_EDITS:-10000}"
DEVICE="${CUDA_DEVICE:-0}"
RESULT_ROOT="${RESULT_ROOT:-$PROJECT_DIR/results}"
CHECKPOINT_ROOT="${CHECKPOINT_ROOT:-${HOME}/.cache/alphaedit_checkpoints}"
REVIVE_THRESH="${REVIVE_THRESH:-0.1}"
LAMBDA_PREV="${LAMBDA_PREV:-1.0}"
LAMBDA_DELTA="${LAMBDA_DELTA:-0.0}"
BASE_ALG="${BASE_ALG:?ERROR: BASE_ALG must be set (MEMIT or AlphaEdit)}"

# NSE needs precomputed KV caches (25 gradient steps per edit without them)
if [[ "$BASE_ALG" == "NSE" ]]; then
    _NSE_CACHE_S3="/s3-data/continual-learning/alphaedit/nse_kv_cache"
    _NSE_CACHE_LOCAL="$PROJECT_DIR/baselines/EvoEdit/share/projects/rewriting-knowledge/kvs"
    if [[ -d "$_NSE_CACHE_S3" ]]; then
        mkdir -p "$_NSE_CACHE_LOCAL"
        for _tar in "$_NSE_CACHE_S3"/*.tar; do
            [[ -f "$_tar" ]] || continue
            echo "  Extracting NSE kv cache from $(basename $_tar)..."
            tar xf "$_tar" -C "$_NSE_CACHE_LOCAL/"
        done
        _kv_count=$(find "$_NSE_CACHE_LOCAL" -name '*.npz' 2>/dev/null | wc -l)
        echo "  KV cache loaded: $_kv_count files"
        [[ "$_kv_count" -lt 100 ]] && { echo "ERROR: KV cache too small ($_kv_count files)"; exit 1; }
    else
        echo "ERROR: NSE KV cache not found at $_NSE_CACHE_S3"
        exit 1
    fi
fi

# Resolve ordering path — check GPT-J-specific orderings first
if [[ -n "${STREAM_PATH:-}" ]]; then
    :
elif [[ -n "${STREAM_DIR_OVERRIDE:-}" ]]; then
    STREAM_PATH="${STREAM_DIR_OVERRIDE}/${ORDERING}_seed${SEED}.json"
elif [[ "$MODEL_NAME" == *gpt-j* ]] || [[ "$MODEL_NAME" == *EleutherAI* ]]; then
    STREAM_PATH="$RESULT_ROOT/matched_ordering_gptj/orderings/${ORDERING}_seed${SEED}.json"
else
    STREAM_PATH="$RESULT_ROOT/matched_ordering/orderings/${ORDERING}_seed${SEED}.json"
fi

if [[ ! -f "$STREAM_PATH" ]]; then
    echo "ERROR: Ordering stream not found: $STREAM_PATH"
    exit 1
fi

case "$BASE_ALG" in
    AlphaEdit) LABEL="AlphaEdit" ;;
    NSE)       LABEL="NSE" ;;
    *)
        if [[ "$LAMBDA_PREV" != "0" ]]; then
            LABEL="MEMIT-Seq"
        else
            LABEL="MEMIT"
        fi
        ;;
esac

echo "═══════════════════════════════════════════════════════════════"
echo "${LABEL}+REVIVE Baseline"
echo "  Seed:        $SEED"
echo "  Ordering:    $ORDERING"
echo "  Threshold:   $REVIVE_THRESH"
echo "  Model:       $MODEL_NAME"
echo "  Hparams:     $HPARAMS_FNAME"
echo "  Lambda_prev: $LAMBDA_PREV"
echo "  Lambda_delta:$LAMBDA_DELTA"
echo "  Base alg:    $BASE_ALG"
echo "  Target:      $TARGET_EDITS edits"
echo "═══════════════════════════════════════════════════════════════"

uv run python src/polykernel/polykernel_seqreg_runner.py \
    --seed "$SEED" \
    --cuda_device "$DEVICE" \
    --model_name "$MODEL_NAME" \
    --hparams_fname "$HPARAMS_FNAME" \
    --ds_name mcf \
    --dataset_size_limit "$TARGET_EDITS" \
    --num_edits 100 \
    --downstream_eval_steps 0 \
    --conserve_memory \
    --lambda_prev "$LAMBDA_PREV" \
    --lambda_delta "$LAMBDA_DELTA" \
    --kernel_degree 1 \
    --cache_strategy all \
    --cache_max none \
    --save_interval 10 \
    --ordering "$ORDERING" \
    --dataset_override "$STREAM_PATH" \
    --eval_at_checkpoints_only \
    --base_alg "$BASE_ALG" \
    --revive \
    --revive_tau "$REVIVE_THRESH"

echo ""
echo "═══════════════════════════════════════════════════════════════"
echo "${LABEL}+REVIVE complete: seed $SEED ordering $ORDERING"
echo "═══════════════════════════════════════════════════════════════"

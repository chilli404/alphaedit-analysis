#!/usr/bin/env bash
set -euo pipefail

# Replicate REVIVE on first 10K MCF, default order.
# Uses polykernel_seqreg_runner with lambda_prev=0 (plain MEMIT base).

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

_PRESET_MODEL="${MODEL_NAME:-}"
if [[ -f "$PROJECT_DIR/.env" ]]; then
    set -a; source "$PROJECT_DIR/.env"; set +a
fi
if [[ -n "$_PRESET_MODEL" ]]; then MODEL_NAME="$_PRESET_MODEL"; fi

SEED="${1:-42}"
MODEL_NAME="${MODEL_NAME:-NousResearch/Meta-Llama-3-8B-Instruct}"
case "$MODEL_NAME" in
    *gpt-j*|*gptj*|*EleutherAI*) HPARAMS_FNAME="EleutherAI_gpt-j-6B.json" ;;
    *Qwen*|*qwen*)               HPARAMS_FNAME="Qwen2.5-7B.json" ;;
    *)                           HPARAMS_FNAME="Llama3-8B.json" ;;
esac
TARGET_EDITS="${TARGET_EDITS:-10000}"
DEVICE="${CUDA_DEVICE:-0}"
REVIVE_THRESH="${REVIVE_THRESH:-0.1}"
LAMBDA_PREV="${LAMBDA_PREV:-0}"
LAMBDA_DELTA="${LAMBDA_DELTA:-0}"
BASE_ALG="${BASE_ALG:-MEMIT}"
# Note: for RECT+REVIVE, use BASE_ALG=MEMIT_rect

echo "═══════════════════════════════════════════════════════════════"
echo "REVIVE Paper Replication (first ${TARGET_EDITS} MCF, default order)"
echo "  Seed:        $SEED"
echo "  Model:       $MODEL_NAME"
echo "  Base alg:    $BASE_ALG"
echo "  Lambda_prev: $LAMBDA_PREV (0=plain base)"
echo "  Threshold:   $REVIVE_THRESH"
echo "═══════════════════════════════════════════════════════════════"

cd "$PROJECT_DIR"

# NSE kv cache (needed if BASE_ALG=NSE)
if [[ "$BASE_ALG" == "NSE" ]]; then
    _NSE_CACHE_S3="/s3-data/continual-learning/alphaedit/nse_kv_cache"
    _NSE_CACHE_LOCAL="$PROJECT_DIR/baselines/EvoEdit/share/projects/rewriting-knowledge/kvs"
    mkdir -p "$_NSE_CACHE_LOCAL"
    if [[ -d "$_NSE_CACHE_S3" ]]; then
        for _tar in "$_NSE_CACHE_S3"/*.tar; do
            [ -f "$_tar" ] || continue
            echo "  Extracting NSE kv cache from $(basename $_tar)..."
            tar xf "$_tar" -C "$_NSE_CACHE_LOCAL/"
        done
        echo "  NSE cache ready: $(find "$_NSE_CACHE_LOCAL" -name '*.npz' | wc -l) files"
    fi
fi

# No dataset override — uses default first-10K MCF
uv run python src/polykernel/polykernel_seqreg_runner.py \
    --seed "$SEED" \
    --cuda_device "$DEVICE" \
    --model_name "$MODEL_NAME" \
    --hparams_fname "$HPARAMS_FNAME" \
    --base_alg "$BASE_ALG" \
    --ds_name mcf \
    --dataset_size_limit "$TARGET_EDITS" \
    --num_edits 100 \
    --downstream_eval_steps 0 \
    --conserve_memory \
    --lambda_prev "$LAMBDA_PREV" \
    --lambda_delta "$LAMBDA_DELTA" \
    --cache_strategy all \
    --cache_max none \
    --save_interval 10 \
    --eval_at_checkpoints_only \
    --revive \
    --revive_tau "$REVIVE_THRESH"

echo "REVIVE + ${BASE_ALG} paper replication complete"

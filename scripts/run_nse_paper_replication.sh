#!/usr/bin/env bash
set -euo pipefail

# Replicate NSE on first 10K MCF, default order — NO ordering override.

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
EVOEDIT_DIR="$PROJECT_DIR/baselines/EvoEdit"

_PRESET_MODEL="${MODEL_NAME:-}"
if [[ -f "$PROJECT_DIR/.env" ]]; then
    set -a; source "$PROJECT_DIR/.env"; set +a
fi
if [[ -n "$_PRESET_MODEL" ]]; then MODEL_NAME="$_PRESET_MODEL"; fi

SEED="${1:-42}"
MODEL_NAME="${MODEL_NAME:-meta-llama/Meta-Llama-3-8B-Instruct}"
case "$MODEL_NAME" in
    *gpt-j*|*gptj*|*EleutherAI*) HPARAMS_FNAME="EleutherAI_gpt-j-6B.json" ;;
    *Qwen*|*qwen*)               HPARAMS_FNAME="Qwen2.5-7B.json" ;;
    *)                           HPARAMS_FNAME="Llama3-8B.json" ;;
esac
TARGET_EDITS="${TARGET_EDITS:-10000}"
DEVICE="${CUDA_DEVICE:-0}"

echo "═══════════════════════════════════════════════════════════════"
echo "NSE Paper Replication (first ${TARGET_EDITS} MCF, default order)"
echo "  Seed:      $SEED"
echo "  Model:     $MODEL_NAME"
echo "  Hparams:   $HPARAMS_FNAME"
echo "═══════════════════════════════════════════════════════════════"

cd "$EVOEDIT_DIR"

# Link datasets
S3_DSETS="/s3-data/continual-learning/alphaedit/dsets"
LOCAL_DSETS="$PROJECT_DIR/data/dsets"
DSET_SRC="$LOCAL_DSETS"
[[ -d "$S3_DSETS" ]] && DSET_SRC="$S3_DSETS"
mkdir -p data
for f in multi_counterfact.json counterfact.json zsre_mend_eval.json \
         attribute_snippets.json tfidf_vocab.json idf.npy; do
    [[ -f "$DSET_SRC/$f" ]] && ln -sf "$DSET_SRC/$f" "data/$f" 2>/dev/null
done

# Link stats
bash "$PROJECT_DIR/scripts/link_stats.sh" 2>/dev/null || true

# NSE kv cache
_NSE_CACHE_S3="/s3-data/continual-learning/alphaedit/nse_kv_cache"
_NSE_CACHE_LOCAL="$EVOEDIT_DIR/share/projects/rewriting-knowledge/kvs"
mkdir -p "$_NSE_CACHE_LOCAL"
if [[ -d "$_NSE_CACHE_S3" ]]; then
    for _tar in "$_NSE_CACHE_S3"/*.tar; do
        [ -f "$_tar" ] || continue
        echo "  Extracting NSE kv cache from $(basename $_tar)..."
        tar xf "$_tar" -C "$_NSE_CACHE_LOCAL/"
    done
fi

python3 -c "import nltk; [nltk.download(pkg, quiet=True) for pkg in ('punkt', 'punkt_tab')]" 2>/dev/null || true

export CUDA_VISIBLE_DEVICES="$DEVICE"
export PYTHONHASHSEED="$SEED"
export TOKENIZERS_PARALLELISM=false

# Override results dir to use S3 FUSE mount if available
_RESULT_DIR="${RESULT_ROOT:-results}"
if [ -d "/s3-data" ]; then
    _RESULT_DIR="/s3-data/continual-learning/alphaedit/results/paper_replications"
fi
cat > globals.yml << GLOBALEOF
---
  RESULTS_DIR: "$_RESULT_DIR"
  DATA_DIR: "data"
  STATS_DIR: "data/stats"
  KV_DIR: "share/projects/rewriting-knowledge/kvs"
  HPARAMS_DIR: "hparams"
  REMOTE_ROOT_URL: "https://memit.baulab.info"
GLOBALEOF
echo "  Results dir: $_RESULT_DIR"

# Resolve model path to S3 FUSE cache if available
MODEL_NAME=$(python3 -c "
import sys; sys.path.insert(0, '$PROJECT_DIR/src/util')
from model_resolve import resolve_model_path
print(resolve_model_path('$MODEL_NAME'))
" 2>/dev/null || echo "$MODEL_NAME")
echo "  Resolved model: $MODEL_NAME"

# Patch checkpoint save for S3 FUSE compatibility
# Patches applied by scripts/patches/apply_all.py at cluster startup

PYTHONPATH=. uv run python experiments/evaluate.py \
    --alg_name NSE \
    --model_name "$MODEL_NAME" \
    --hparams_fname "$HPARAMS_FNAME" \
    --ds_name mcf \
    --dataset_size_limit "$TARGET_EDITS" \
    --num_edits 100 \
    --downstream_eval_steps 0 \
    --save_every 1000 \
    --conserve_memory

echo "NSE paper replication complete"

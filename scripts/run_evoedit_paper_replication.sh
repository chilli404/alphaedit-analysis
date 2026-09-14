#!/usr/bin/env bash
set -euo pipefail

# Replicate EvoEdit paper's exact 10K experiment.
# Uses first 10K MCF records in default order — NO ordering override.
# This tests whether our EvoEdit integration matches published results.
#
# Usage:
#   bash scripts/run_evoedit_paper_replication.sh [SEED]

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
RESULT_ROOT="${RESULT_ROOT:-$PROJECT_DIR/results}"

echo "═══════════════════════════════════════════════════════════════"
echo "EvoEdit Paper Replication (first 10K MCF, default order)"
echo "  Seed:      $SEED"
echo "  Model:     $MODEL_NAME"
echo "  Hparams:   $HPARAMS_FNAME"
echo "  Target:    $TARGET_EDITS edits"
echo "  NO ordering override — uses MCF default order"
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

# NLTK
python3 -c "import nltk; [nltk.download(pkg, quiet=True) for pkg in ('punkt', 'punkt_tab')]" 2>/dev/null || true

export CUDA_VISIBLE_DEVICES="$DEVICE"
export SKIP_MEGA_BATCH_EVAL=1
export PYTHONHASHSEED="$SEED"
export TOKENIZERS_PARALLELISM=false

# Override results dir to use S3 FUSE mount if available
_RESULT_DIR="${RESULT_ROOT:-results}/failure_curve_checkpointed"
if [ -d "/s3-data" ]; then
    _RESULT_DIR="/s3-data/continual-learning/alphaedit/results/failure_curve_checkpointed"
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

# Run EvoEdit with NO dataset override — just the standard first-10K MCF
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
    --alg_name EvoEdit \
    --model_name "$MODEL_NAME" \
    --hparams_fname "$HPARAMS_FNAME" \
    --ds_name mcf \
    --dataset_size_limit "$TARGET_EDITS" \
    --num_edits 100 \
    --downstream_eval_steps 0 \
    --save_every 1000 \
    --conserve_memory

echo "═══════════════════════════════════════════════════════════════"
echo "EvoEdit paper replication complete"
echo "═══════════════════════════════════════════════════════════════"

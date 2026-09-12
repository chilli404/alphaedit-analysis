#!/usr/bin/env bash
set -euo pipefail

# Generic baseline runner for any algorithm in baselines/EvoEdit/experiments/evaluate.py
# Supports: MEMIT_seq_rect, MEMIT_rect, MEMIT_seq, or any ALG_DICT entry
#
# Usage:
#   ALG_NAME=MEMIT_seq_rect bash scripts/run_generic_baseline.sh SEED [ORDERING]

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
EVOEDIT_DIR="$PROJECT_DIR/baselines/EvoEdit"

_PRESET_MODEL="${MODEL_NAME:-}"
if [[ -f "$PROJECT_DIR/.env" ]]; then
    set -a; source "$PROJECT_DIR/.env"; set +a
fi
if [[ -n "$_PRESET_MODEL" ]]; then MODEL_NAME="$_PRESET_MODEL"; fi

SEED="${1:?Usage: ALG_NAME=... $0 SEED [ORDERING]}"
ORDERING="${2:-${ORDERING:-fb_random0}}"
ALG_NAME="${ALG_NAME:?ALG_NAME must be set}"
MODEL_NAME="${MODEL_NAME:-meta-llama/Meta-Llama-3-8B-Instruct}"
case "$MODEL_NAME" in
    *gpt-j*|*gptj*|*EleutherAI*) HPARAMS_FNAME="EleutherAI_gpt-j-6B.json" ;;
    *Qwen*|*qwen*)               HPARAMS_FNAME="Qwen2.5-7B.json" ;;
    *)                           HPARAMS_FNAME="Llama3-8B.json" ;;
esac
TARGET_EDITS="${TARGET_EDITS:-10000}"
DEVICE="${CUDA_DEVICE:-0}"
RESULT_ROOT="${RESULT_ROOT:-$PROJECT_DIR/results}"

# Resolve ordering path
if [[ "$MODEL_NAME" == *gpt-j* ]] || [[ "$MODEL_NAME" == *EleutherAI* ]]; then
    STREAM_PATH="$RESULT_ROOT/matched_ordering_gptj/orderings/${ORDERING}_seed${SEED}.json"
else
    STREAM_PATH="$RESULT_ROOT/matched_ordering/orderings/${ORDERING}_seed${SEED}.json"
fi
if [[ ! -f "$STREAM_PATH" ]]; then
    echo "ERROR: Ordering stream not found: $STREAM_PATH"
    exit 1
fi

echo "═══════════════════════════════════════════════════════════════"
echo "Generic Baseline: $ALG_NAME"
echo "  Seed:      $SEED"
echo "  Ordering:  $ORDERING"
echo "  Stream:    $STREAM_PATH"
echo "  Model:     $MODEL_NAME"
echo "  Hparams:   $HPARAMS_FNAME"
echo "  Target:    $TARGET_EDITS edits"
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
export PYTHONHASHSEED="$SEED"
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

# Patch checkpoint save for S3 FUSE compatibility
# Patches applied by scripts/patches/apply_all.py at cluster startup

# Run with dataset override for ordering

# Resolve model path to S3 FUSE cache if available
MODEL_NAME=$(python3 -c "
import sys; sys.path.insert(0, '$PROJECT_DIR/src/util')
from model_resolve import resolve_model_path
print(resolve_model_path('$MODEL_NAME'))
" 2>/dev/null || echo "$MODEL_NAME")
echo "  Resolved model: $MODEL_NAME"

uv run python -c "
import os, sys, json, random
sys.path.insert(0, '.')
import numpy as np
import torch

seed = $SEED
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

sys.argv = [
    'experiments.evaluate',
    '--alg_name=$ALG_NAME',
    '--model_name=$MODEL_NAME',
    '--hparams_fname=$HPARAMS_FNAME',
    '--ds_name=mcf',
    '--dataset_size_limit=$TARGET_EDITS',
    '--num_edits=100',
    '--downstream_eval_steps=0',
    '--save_every=1000',
    '--conserve_memory',
    '--forgetting_eval_interval=0',
]

with open('experiments/evaluate.py', 'r') as f:
    source = f.read()

# Dataset override
loop_anchor = '    for record_chunks in chunks(ds, num_edits):'
override_code = '''    # === DATASET OVERRIDE ===
    import json as _ov_json
    with open(\"$STREAM_PATH\", \"r\") as _ov_f:
        _ov_stream = _ov_json.load(_ov_f)
    _ov_attr = \"_data\" if hasattr(ds, \"_data\") else \"data\"
    _ov_existing = getattr(ds, _ov_attr)
    _ov_id_map = {r.get(\"case_id\", i): r for i, r in enumerate(_ov_existing)}
    _ov_stream_ids = [r[\"case_id\"] for r in _ov_stream]
    _ov_matched = [_ov_id_map[cid] for cid in _ov_stream_ids if cid in _ov_id_map]
    if len(_ov_matched) >= len(_ov_stream_ids) * 0.95:
        setattr(ds, _ov_attr, _ov_matched)
        print(f\"  [OVERRIDE] Reordered {len(_ov_matched)} records from stream\")
    else:
        setattr(ds, _ov_attr, _ov_stream)
        print(f\"  [OVERRIDE] Replaced with {len(_ov_stream)} records from stream\")
    # === END OVERRIDE ===
'''
source = source.replace(loop_anchor, override_code + loop_anchor, 1)

# Result dir override — modify globals.yml before exec
import yaml
from pathlib import Path
_result_dir = '$RESULT_ROOT/matched_ordering/$ALG_NAME/$ORDERING/seed$SEED/${TARGET_EDITS}edits/$ALG_NAME'
_results_root = str(Path(_result_dir).parent.parent)
try:
    with open('globals.yml') as _gf:
        _gdata = yaml.safe_load(_gf)
    _gdata['RESULTS_DIR'] = _results_root
    with open('globals.yml', 'w') as _gf:
        yaml.dump(_gdata, _gf)
except Exception:
    pass

exec(compile(source, 'experiments/evaluate.py', 'exec'))
"

echo "═══════════════════════════════════════════════════════════════"
echo "$ALG_NAME complete: seed $SEED ordering $ORDERING"
echo "═══════════════════════════════════════════════════════════════"

#!/usr/bin/env bash
set -euo pipefail

# Build NSE kv cache (v_star) for MCF records.
# Supports sharding: split the 21K records across N workers for parallel builds.
#
# Usage:
#   bash scripts/build_nse_cache.sh                     # All records, 1 worker
#   SHARD=0 NUM_SHARDS=4 bash scripts/build_nse_cache.sh  # Shard 0 of 4
#   MODEL_NAME=EleutherAI/gpt-j-6b SHARD=2 NUM_SHARDS=4 bash scripts/build_nse_cache.sh

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

if [[ -f "$PROJECT_DIR/.env" ]]; then
    set -a; source "$PROJECT_DIR/.env"; set +a
fi

MODEL_NAME="${MODEL_NAME:-meta-llama/Meta-Llama-3-8B-Instruct}"
# Always derive HPARAMS from MODEL_NAME (don't use :- which keeps stale values)
case "$MODEL_NAME" in
    *gpt-j*|*gptj*|*EleutherAI*) HPARAMS_FNAME="EleutherAI_gpt-j-6B.json" ;;
    *Qwen*|*qwen*)               HPARAMS_FNAME="Qwen2.5-7B.json" ;;
    *)                           HPARAMS_FNAME="Llama3-8B.json" ;;
esac
DEVICE="${CUDA_DEVICE:-0}"
SHARD="${SHARD:-0}"
NUM_SHARDS="${NUM_SHARDS:-1}"

# Extract existing caches from S3 tar files (if on SkyPilot)
_NSE_CACHE_S3="/s3-data/continual-learning/alphaedit/nse_kv_cache"
_NSE_CACHE_LOCAL="$PROJECT_DIR/data/nse_kv_cache"
mkdir -p "$_NSE_CACHE_LOCAL"

_MODEL_TAG=$(echo "$MODEL_NAME" | tr '/' '_')

if [[ -d "$_NSE_CACHE_S3" ]]; then
    for _tar in "$_NSE_CACHE_S3"/*.tar; do
        [ -f "$_tar" ] || continue
        echo "  Extracting existing cache from $(basename $_tar)..."
        tar xf "$_tar" -C "$_NSE_CACHE_LOCAL/" 2>/dev/null || true
    done
    _S3_DIR="$_NSE_CACHE_S3/${_MODEL_TAG}_NSE"
    _LOCAL_DIR="$_NSE_CACHE_LOCAL/${_MODEL_TAG}_NSE"
    if [[ -d "$_S3_DIR" ]]; then
        echo "  Syncing loose cache files from S3..."
        mkdir -p "$_LOCAL_DIR"
        cp -n "$_S3_DIR"/*.npz "$_LOCAL_DIR/" 2>/dev/null || true
    fi
fi

# Use S3 dir as output if available (writes go to S3 via FUSE)
if [[ -d "$_NSE_CACHE_S3" ]]; then
    CACHE_OUT="$_NSE_CACHE_S3/${_MODEL_TAG}_NSE"
else
    CACHE_OUT="$_NSE_CACHE_LOCAL/${_MODEL_TAG}_NSE"
fi
mkdir -p "$CACHE_OUT"

# Also check local for existing entries
_EXISTING=$(find "$_NSE_CACHE_LOCAL/${_MODEL_TAG}_NSE" "$CACHE_OUT" -name "*.npz" 2>/dev/null | wc -l | tr -d ' ')
echo "  Existing cached entries: $_EXISTING"

echo "═══════════════════════════════════════════════════════════════"
echo "NSE Cache Build"
echo "  Model:       $MODEL_NAME"
echo "  Hparams:     $HPARAMS_FNAME"
echo "  Shard:       $SHARD / $NUM_SHARDS"
echo "  Cache out:   $CACHE_OUT"
echo "═══════════════════════════════════════════════════════════════"

# Link datasets into baselines/EvoEdit/data/ (same as run_nse_baseline.sh)
EVOEDIT_DIR="$PROJECT_DIR/baselines/EvoEdit"
S3_DSETS="/s3-data/continual-learning/alphaedit/dsets"
LOCAL_DSETS="$PROJECT_DIR/data/dsets"
DSET_SRC="$LOCAL_DSETS"
[[ -d "$S3_DSETS" ]] && DSET_SRC="$S3_DSETS"
mkdir -p "$EVOEDIT_DIR/data"
for f in multi_counterfact.json counterfact.json zsre_mend_eval.json \
         attribute_snippets.json tfidf_vocab.json idf.npy; do
    if [[ -f "$DSET_SRC/$f" ]] && [[ ! -f "$EVOEDIT_DIR/data/$f" ]]; then
        ln -sf "$DSET_SRC/$f" "$EVOEDIT_DIR/data/$f"
    fi
done
echo "  Datasets linked from: $DSET_SRC"

cd "$PROJECT_DIR"

uv run python -c "
import sys, os, json, time
sys.path.insert(0, 'baselines/EvoEdit')
os.chdir('baselines/EvoEdit')

from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import numpy as np
from pathlib import Path

import sys
sys.path.insert(0, '$PROJECT_DIR/src/util')
from model_resolve import resolve_model_path

model_name = '$MODEL_NAME'
resolved = resolve_model_path(model_name)
print(f'Resolved: {model_name} -> {resolved}')

shard = $SHARD
num_shards = $NUM_SHARDS
cache_out = Path('$CACHE_OUT')

print(f'Loading {resolved}...')
model = AutoModelForCausalLM.from_pretrained(resolved).cuda()  # no dtype override — use model config (bfloat16 for Llama-3)
tok = AutoTokenizer.from_pretrained(resolved)
tok.pad_token = tok.eos_token

from nse import NSEHyperParams
hparams = NSEHyperParams.from_json('hparams/NSE/$HPARAMS_FNAME')

from util.globals import DATA_DIR
with open(DATA_DIR / 'multi_counterfact.json') as f:
    all_records = json.load(f)
print(f'Total MCF records: {len(all_records)}')

# Shard the records
shard_records = [r for i, r in enumerate(all_records) if i % num_shards == shard]
print(f'Shard {shard}/{num_shards}: {len(shard_records)} records')

# Also check local cache dir for existing entries
local_cache = Path('../../data/nse_kv_cache') / f'{model_name.replace(chr(47), chr(95))}_NSE'

from nse.compute_z import compute_z
from nse.nse_main import get_context_templates

context_templates = get_context_templates(model, tok)

cached = 0
computed = 0
errors = 0
t0 = time.time()

for i, record in enumerate(shard_records):
    case_id = record['case_id']
    fname = f'mcf_layer_{hparams.layers[-1]}_clamp_{hparams.clamp_norm_factor}_case_{case_id}.npz'
    cache_path = cache_out / fname
    local_path = local_cache / fname if local_cache.exists() else None

    if cache_path.exists() or (local_path and local_path.exists()):
        cached += 1
        continue

    try:
        cur_z = compute_z(
            model, tok,
            record['requested_rewrite'],
            hparams,
            hparams.layers[-1],
            context_templates,
        )
        # Save to /tmp first then copy — S3 FUSE doesn't support np.savez's atomic rename
        import tempfile, shutil
        tmp_path = Path(tempfile.gettempdir()) / fname
        np.savez(tmp_path, v_star=cur_z.detach().cpu().numpy())
        shutil.copyfileobj(open(tmp_path, 'rb'), open(cache_path, 'wb'))
        tmp_path.unlink()
        computed += 1
    except Exception as e:
        errors += 1
        if errors <= 5:
            print(f'  ERROR case {case_id}: {e}')

    if (i+1) % 100 == 0:
        elapsed = time.time() - t0
        rate = computed / elapsed if elapsed > 0 else 0
        remaining = (len(shard_records) - i - 1 - cached) / rate / 3600 if rate > 0 else 0
        print(f'  [{i+1}/{len(shard_records)}] cached={cached} computed={computed} errors={errors} '
              f'rate={rate:.1f}/s est_remaining={remaining:.1f}h')

elapsed = time.time() - t0
print(f'Done shard {shard}. cached={cached} computed={computed} errors={errors} time={elapsed/3600:.1f}h')
"

echo "Cache build shard $SHARD complete."

# Upload tar to S3 so other clusters can use it
if [[ -d "$_NSE_CACHE_S3" ]]; then
    if [[ "$NUM_SHARDS" -gt 1 ]]; then
        _TAR_NAME="${_MODEL_TAG}_nse_cache_shard${SHARD}.tar"
    else
        _TAR_NAME="${_MODEL_TAG}_nse_cache.tar"
    fi
    _LOCAL_DIR="$_NSE_CACHE_LOCAL/${_MODEL_TAG}_NSE"
    _CACHE_COUNT=$(find "$_LOCAL_DIR" -name '*.npz' 2>/dev/null | wc -l | tr -d ' ')
    if [[ "$_CACHE_COUNT" -gt 0 ]]; then
        echo "Uploading cache tar to S3 (${_CACHE_COUNT} files)..."
        tar cf "/tmp/${_TAR_NAME}" -C "$_NSE_CACHE_LOCAL" "${_MODEL_TAG}_NSE"
        cp "/tmp/${_TAR_NAME}" "$_NSE_CACHE_S3/${_TAR_NAME}"
        rm -f "/tmp/${_TAR_NAME}"
        echo "  Uploaded: ${_NSE_CACHE_S3}/${_TAR_NAME}"
    fi
fi

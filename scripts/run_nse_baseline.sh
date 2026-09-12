#!/usr/bin/env bash
set -euo pipefail

# Run NSE (Neuron-Level Sequential Editing) baseline on fixed-batch orderings.
# Uses the EvoEdit repo (baselines/EvoEdit/) which already contains NSE.
#
# NSE is methodologically distinct from null-space projection methods:
# it selects neurons based on activation magnitudes and optimizes target states
# using original model weights, rather than projecting into a null space.
#
# Usage:
#   bash scripts/run_nse_baseline.sh SEED [ORDERING]
#   ORDERING=fb_high_exposure bash scripts/run_nse_baseline.sh 42

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
EVOEDIT_DIR="$PROJECT_DIR/baselines/EvoEdit"

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
NUM_EDITS="${NUM_EDITS:-100}"
DEVICE="${CUDA_DEVICE:-0}"
RESULT_ROOT="${RESULT_ROOT:-$PROJECT_DIR/results}"
CHECKPOINT_ROOT="${CHECKPOINT_ROOT:-${HOME}/.cache/alphaedit_checkpoints}"

# Resolve ordering path — GPT-J uses matched_ordering_gptj/
if [[ -n "${STREAM_PATH:-}" ]]; then
    : # explicit override
elif [[ "$MODEL_NAME" == *gpt-j* ]] || [[ "$MODEL_NAME" == *gptj* ]]; then
    STREAM_PATH="$RESULT_ROOT/matched_ordering_gptj/orderings/${ORDERING}_seed${SEED}.json"
else
    STREAM_PATH="$RESULT_ROOT/matched_ordering/orderings/${ORDERING}_seed${SEED}.json"
fi
if [[ ! -f "$STREAM_PATH" ]]; then
    echo "ERROR: Ordering stream not found: $STREAM_PATH"
    exit 1
fi

# Pre-populate NSE kv cache from S3 FUSE mount if available (avoids 6h re-computation)
_NSE_CACHE_S3="/s3-data/continual-learning/alphaedit/nse_kv_cache"
_NSE_CACHE_LOCAL="$EVOEDIT_DIR/share/projects/rewriting-knowledge/kvs"
if [ -d "$_NSE_CACHE_S3" ]; then
    mkdir -p "$_NSE_CACHE_LOCAL"
    # Try tar archives first (faster than many small files over FUSE)
    for _tar in "$_NSE_CACHE_S3"/*.tar; do
        [ -f "$_tar" ] || continue
        echo "  Extracting NSE kv cache from $(basename $_tar)..."
        tar xf "$_tar" -C "$_NSE_CACHE_LOCAL/"
        echo "  Extracted."
    done
    # Then try directories
    for _kv_dir in "$_NSE_CACHE_S3"/*/; do
        [ -d "$_kv_dir" ] || continue
        _name=$(basename "$_kv_dir")
        if [ ! -d "$_NSE_CACHE_LOCAL/$_name" ] || [ "$(ls "$_NSE_CACHE_LOCAL/$_name" 2>/dev/null | wc -l)" -lt 1000 ]; then
            echo "  Pre-populating NSE kv cache: $_name"
            mkdir -p "$_NSE_CACHE_LOCAL/$_name"
            cp "$_kv_dir"/*.npz "$_NSE_CACHE_LOCAL/$_name/" 2>/dev/null
            echo "  Done: $(ls "$_NSE_CACHE_LOCAL/$_name" | wc -l) files cached"
        else
            echo "  NSE kv cache already populated: $_name ($(ls "$_NSE_CACHE_LOCAL/$_name" | wc -l) files)"
        fi
    done
    echo "  Total cached: $(find "$_NSE_CACHE_LOCAL" -name '*.npz' | wc -l) files"
else
    echo "  No S3 NSE kv cache found at $_NSE_CACHE_S3 — will compute from scratch"
fi

echo "═══════════════════════════════════════════════════════════════"
echo "NSE (Neuron-Level Sequential Editing) Baseline"
echo "  Seed:      $SEED"
echo "  Ordering:  $ORDERING"
echo "  Stream:    $STREAM_PATH"
echo "  Model:     $MODEL_NAME"
echo "  Target:    $TARGET_EDITS edits"
echo "═══════════════════════════════════════════════════════════════"

# --- Data setup ---
# Datasets and stats are linked by link_dsets.sh + link_stats.sh (called from YAML or apply_all.py).
# Patches applied by apply_all.py. Verify they're present; if not, set up (for local runs).
cd "$EVOEDIT_DIR"
if [[ ! -f "data/multi_counterfact.json" ]]; then
    echo "  Linking data (not yet done by apply_all.py)..."
    bash "$PROJECT_DIR/scripts/link_dsets.sh"
    bash "$PROJECT_DIR/scripts/link_stats.sh"
    uv run python "$PROJECT_DIR/scripts/patches/apply_all.py" --baselines-only
fi

# NLTK data
python3 -c "import nltk; [nltk.download(pkg, quiet=True) for pkg in ('punkt', 'punkt_tab')]" 2>/dev/null || true

export CUDA_VISIBLE_DEVICES="$DEVICE"
export PYTHONHASHSEED="$SEED"
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

# --- Run NSE via source injection ---
uv run python -c "
import os, sys, random, json
import numpy as np
import torch

seed = $SEED
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

sys.argv = [
    'experiments.evaluate',
    '--alg_name=NSE',
    '--model_name=$MODEL_NAME',
    '--hparams_fname=$HPARAMS_FNAME',
    '--ds_name=mcf',
    '--dataset_size_limit=$TARGET_EDITS',
    '--num_edits=$NUM_EDITS',
    '--downstream_eval_steps=0',
    '--save_every=1000',
    '--conserve_memory',
    '--forgetting_eval_interval=0',
]

# Auto-detect existing run directory for resume
sys.path.insert(0, '$PROJECT_DIR/src/util')
from evoedit_resume import find_continue_run
_result_base = '$RESULT_ROOT/matched_ordering/NSE/${ORDERING}/seed${SEED}/${TARGET_EDITS}edits/NSE'
_continue_run = find_continue_run(_result_base)
if _continue_run:
    sys.argv.append(f'--continue_from_run={_continue_run}')
    print(f'  [RESUME] Continuing from {_continue_run}')

with open('experiments/evaluate.py', 'r') as f:
    source = f.read()

# --- Patch 1: Model loading — resolve to S3 FUSE or HF cache ---
_model_load_original = '        model = AutoModelForCausalLM.from_pretrained(model_name).cuda()\n        tok = AutoTokenizer.from_pretrained(model_name)'
_model_load_patched = '        import sys as _sys; _sys.path.insert(0, \"$PROJECT_DIR/src/util\")\n        from model_resolve import resolve_model_path as _resolve_model\n        _resolved = _resolve_model(model_name)\n        print(f\"  [MODEL] Resolved: {model_name} -> {_resolved}\")\n        model = AutoModelForCausalLM.from_pretrained(_resolved).cuda()\n        tok = AutoTokenizer.from_pretrained(_resolved)'
assert _model_load_original in source, 'Model load anchor not found in evaluate.py'
source = source.replace(_model_load_original, _model_load_patched, 1)

# --- Patch 2: CUDA line ---
source = source.replace(
    '#os.environ[\"CUDA_VISIBLE_DEVICES\"] = \"1\"',
    '# CUDA managed by runner',
)
source = source.replace(
    'os.environ[\"CUDA_VISIBLE_DEVICES\"] = \"1\"',
    '# CUDA managed by runner',
)

# --- Patch 3: Override RESULTS_DIR ---
_globals_import = 'from util.globals import *'
if _globals_import in source:
    source = source.replace(
        _globals_import,
        _globals_import + '\nRESULTS_DIR = Path(\"$RESULT_ROOT/matched_ordering/NSE/${ORDERING}/seed${SEED}/${TARGET_EDITS}edits\")\n',
        1,
    )

# --- Patch 4: Resume + Dataset override ---
from evoedit_resume import patch_resume
source, _ = patch_resume(source, _result_base, alg_name='NSE')

loop_anchor = '    for record_chunks in chunks(ds, num_edits):'
assert loop_anchor in source, 'Loop anchor not found in evaluate.py'

override_code = '''    # === DATASET OVERRIDE (injected by runner) ===
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

# --- Patch 5: Lightweight checkpoint save ---
_save_original = '''                try:
                    model.save_pretrained(
                        ckpt_dir,
                        safe_serialization=True,
                        max_shard_size=\"5GB\"
                    )
                    tok.save_pretrained(ckpt_dir)'''
_save_patched = '''                try:
                    _layer_weights = {}
                    _all_params = dict(model.named_parameters())
                    for _layer_idx in hparams.layers:
                        _rkey = hparams.rewrite_module_tmp.format(_layer_idx) + \".weight\"
                        if _rkey in _all_params:
                            _layer_weights[_rkey] = _all_params[_rkey].data.cpu()
                    torch.save(_layer_weights, str(ckpt_dir / \"model_weights.pt\"))
                    if \"cache_c\" in dir() or \"cache_c\" in globals():
                        torch.save(cache_c, str(ckpt_dir / \"cache_c.pt\"))
                    print(f\"  Saved {len(_layer_weights)} layer weights + cache_c to {ckpt_dir}\")'''
if _save_original in source:
    source = source.replace(_save_original, _save_patched, 1)
    print('  [PATCH] Checkpoint save: lightweight layer-weights-only')
else:
    print('  [PATCH] WARNING: save_pretrained anchor not found')

# --- Patch 6: Print edit count every batch ---
_cnt_anchor = '        cnt+=1\n        print(\"Execution took\", exec_time)'
_cnt_patched = '        cnt+=1\n        _total_edits = cnt * num_edits\n        print(f\"=================================================================={_total_edits}_edit==================================================================\", flush=True)\n        print(\"Execution took\", exec_time)'
if _cnt_anchor in source:
    source = source.replace(_cnt_anchor, _cnt_patched, 1)
    print('  [PATCH] Edit count print injected')
else:
    print('  [PATCH] WARNING: cnt anchor not found for edit print')

print('[NSE] evaluate.py patched: model resolve, results dir, dataset override, checkpoints')

exec(compile(source, 'experiments/evaluate.py', 'exec'),
     {'__name__': '__main__', '__file__': 'experiments/evaluate.py'})
"

echo ""
echo "═══════════════════════════════════════════════════════════════"
echo "NSE complete: seed $SEED ordering $ORDERING"
echo "═══════════════════════════════════════════════════════════════"

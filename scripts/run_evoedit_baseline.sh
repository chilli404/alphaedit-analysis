#!/usr/bin/env bash
set -euo pipefail

# Run EvoEdit baseline on fixed-batch orderings.
# Uses the EvoEdit repo (baselines/EvoEdit/) with our ordering streams.
#
# EvoEdit is built on the same AlphaEdit codebase, so it shares the same
# evaluate.py structure, data layout, and hparams format. We run it via
# source injection (same pattern as our other runners).
#
# Usage:
#   bash scripts/run_evoedit_baseline.sh SEED [ORDERING]
#   ORDERING=fb_high_exposure bash scripts/run_evoedit_baseline.sh 42

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
# Always derive hparams from model name — never from env
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

echo "═══════════════════════════════════════════════════════════════"
echo "EvoEdit Baseline"
echo "  Seed:      $SEED"
echo "  Ordering:  $ORDERING"
echo "  Stream:    $STREAM_PATH"
echo "  Model:     $MODEL_NAME"
echo "  Target:    $TARGET_EDITS edits"
echo "═══════════════════════════════════════════════════════════════"

# --- Data setup ---
# Datasets and stats are linked by link_dsets.sh + link_stats.sh (called from YAML or apply_all.py).
# Verify they're present; if not, link them (for local runs).
cd "$EVOEDIT_DIR"
if [[ ! -f "data/multi_counterfact.json" ]]; then
    echo "  Linking data (not yet done by apply_all.py)..."
    bash "$PROJECT_DIR/scripts/link_dsets.sh"
    bash "$PROJECT_DIR/scripts/link_stats.sh"
fi

# Patch EvoEdit to accept extra kwargs (return_orig_weights_device passed by evaluate.py)
# Idempotent: skip if already patched
grep -q '_kwargs' EvoEdit/EvoEdit_main.py 2>/dev/null || \
    sed -i.bak 's/    z_error_threshold = 1e-2,/    z_error_threshold = 1e-2, **_kwargs,/' EvoEdit/EvoEdit_main.py 2>/dev/null || true

# Fix deprecated Wikipedia dataset config
sed -i.bak 's/20200501.en/20220301.en/' rome/layer_stats.py 2>/dev/null || true

# Patch checkpoint save: replace save_pretrained with lightweight layer-weights-only save
# This avoids the S3 FUSE sharding issue where only 1 of 4 shards gets written
python3 -c "
import re
with open('experiments/evaluate.py', 'r') as f:
    src = f.read()
old = '''                    model.save_pretrained(
                        ckpt_dir,
                        safe_serialization=True,
                        max_shard_size=\"5GB\"
                    )
                    tok.save_pretrained(ckpt_dir)'''
new = '''                    _lw = {}
                    _ap = dict(model.named_parameters())
                    for _li in hparams.layers:
                        _rk = hparams.rewrite_module_tmp.format(_li) + '.weight'
                        if _rk in _ap:
                            _lw[_rk] = _ap[_rk].data.cpu()
                    torch.save(_lw, str(ckpt_dir / 'model_weights.pt'))
                    print(f'  Saved {len(_lw)} layer weights to {ckpt_dir / \"model_weights.pt\"}')'''
if old in src:
    src = src.replace(old, new, 1)
    with open('experiments/evaluate.py', 'w') as f:
        f.write(src)
    print('  [PATCH] Checkpoint save: lightweight layer-weights-only')
else:
    print('  [PATCH] WARNING: save_pretrained anchor not found in evaluate.py')
"

# NLTK data
python3 -c "import nltk; [nltk.download(pkg, quiet=True) for pkg in ('punkt', 'punkt_tab')]" 2>/dev/null || true

export CUDA_VISIBLE_DEVICES="$DEVICE"
export PYTHONHASHSEED="$SEED"
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

# --- Run EvoEdit via source injection ---
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
    '--alg_name=EvoEdit',
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
_model_tag = 'gptj' if 'gpt-j' in '$MODEL_NAME'.lower() or 'gptj' in '$MODEL_NAME'.lower() else ''
_evo_subdir = 'evoedit_gptj' if _model_tag else 'evoedit'
_result_base = f'$RESULT_ROOT/{_evo_subdir}/${ORDERING}/seed${SEED}/${TARGET_EDITS}edits/EvoEdit'
_continue_run = find_continue_run(_result_base)
if _continue_run:
    sys.argv.append(f'--continue_from_run={_continue_run}')
    print(f'  [RESUME] Continuing from {_continue_run}')

with open('experiments/evaluate.py', 'r') as f:
    source = f.read()

# --- Patch 1: Model loading — resolve to S3 FUSE or HF cache ---
_model_load_original = '        model = AutoModelForCausalLM.from_pretrained(model_name).cuda()\n        tok = AutoTokenizer.from_pretrained(model_name)'
_model_load_patched = '        import sys as _sys; _sys.path.insert(0, \"$PROJECT_DIR/src/util\")\n        from model_resolve import resolve_model_path as _resolve_model\n        _resolved = _resolve_model(model_name)\n        print(f\"  [MODEL] Resolved: {model_name} -> {_resolved}\")\n        model = AutoModelForCausalLM.from_pretrained(_resolved).cuda()\n        tok = AutoTokenizer.from_pretrained(_resolved)'
assert _model_load_original in source, 'Model load anchor not found in EvoEdit evaluate.py'
source = source.replace(_model_load_original, _model_load_patched, 1)

# --- Patch 3: CUDA line (already commented out in EvoEdit, but be safe) ---
source = source.replace(
    '#os.environ[\"CUDA_VISIBLE_DEVICES\"] = \"1\"',
    '# CUDA managed by runner',
)
source = source.replace(
    'os.environ[\"CUDA_VISIBLE_DEVICES\"] = \"1\"',
    '# CUDA managed by runner',
)

# --- Patch 4: P-cache — load null_space_project.pt if available (skips 20 min SVD) ---
_p_compute_original = '''    if alg_name == \"AlphaEdit\" or alg_name == \"EvoEdit\":
        for i, layer in enumerate(hparams.layers):
            P[i,:,:] = get_project(model,tok,layer,hparams)
        torch.save(P, \"null_space_project.pt\")'''
_p_compute_patched = '''    if alg_name == \"AlphaEdit\" or alg_name == \"EvoEdit\":
        import os as _pos
        if _pos.path.exists(\"null_space_project.pt\"):
            P = torch.load(\"null_space_project.pt\", map_location=\"cpu\")
            print(f\"  [P-CACHE] Loaded null_space_project.pt: {P.shape}\")
        else:
            for i, layer in enumerate(hparams.layers):
                P[i,:,:] = get_project(model,tok,layer,hparams)
            torch.save(P, \"null_space_project.pt\")
            print(\"  [P-CACHE] Computed and saved null_space_project.pt\")'''
if _p_compute_original in source:
    source = source.replace(_p_compute_original, _p_compute_patched, 1)
    print('  [PATCH] P-cache enabled')
else:
    print('  [PATCH] WARNING: P-cache anchor not found')

# --- Patch 5: Override RESULTS_DIR to project-level results ---
_globals_import = 'from util.globals import *'
if _globals_import in source:
    source = source.replace(
        _globals_import,
        _globals_import + '\nRESULTS_DIR = Path(\"$RESULT_ROOT/' + _evo_subdir + '/${ORDERING}/seed${SEED}/${TARGET_EDITS}edits\")\n',
        1,
    )

# --- Patch 5: Resume + Dataset override ---
from evoedit_resume import patch_resume
source, _ = patch_resume(source, _result_base, alg_name='EvoEdit')

loop_anchor = '    for record_chunks in chunks(ds, num_edits):'
assert loop_anchor in source, 'Loop anchor not found in EvoEdit evaluate.py'

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

# --- Patch 7: Replace save_pretrained with lightweight layer-weights-only save ---
# EvoEdit's save_pretrained creates 4 shards (16GB) which S3 FUSE loses.
# Save only the 5 edited layer weights (~560MB) in our checkpoint format.
_save_original = '''                try:
                    model.save_pretrained(
                        ckpt_dir,
                        safe_serialization=True,
                        max_shard_size=\"5GB\"
                    )
                    tok.save_pretrained(ckpt_dir)'''
_save_patched = '''                try:
                    # Lightweight save: only edited layer weights (S3 FUSE compatible)
                    _layer_weights = {}
                    _all_params = dict(model.named_parameters())
                    for _layer_idx in hparams.layers:
                        _rkey = hparams.rewrite_module_tmp.format(_layer_idx) + \".weight\"
                        if _rkey in _all_params:
                            _layer_weights[_rkey] = _all_params[_rkey].data.cpu()
                    torch.save(_layer_weights, str(ckpt_dir / \"model_weights.pt\"))
                    # Also save cache_c for resume
                    if \"cache_c\" in dir() or \"cache_c\" in globals():
                        torch.save(cache_c, str(ckpt_dir / \"cache_c.pt\"))
                    print(f\"  Saved {len(_layer_weights)} layer weights + cache_c to {ckpt_dir}\")'''
if _save_original in source:
    source = source.replace(_save_original, _save_patched, 1)
    print('  [PATCH] Checkpoint save: lightweight layer-weights-only')
else:
    print('  [PATCH] WARNING: save_pretrained anchor not found')

print('[EvoEdit] evaluate.py patched: model resolve, results dir, dataset override, checkpoints')

exec(compile(source, 'experiments/evaluate.py', 'exec'),
     {'__name__': '__main__', '__file__': 'experiments/evaluate.py'})
"

echo ""
echo "═══════════════════════════════════════════════════════════════"
echo "EvoEdit complete: seed $SEED ordering $ORDERING"
echo "═══════════════════════════════════════════════════════════════"

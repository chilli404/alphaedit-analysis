#!/usr/bin/env bash
set -euo pipefail

# ZsRE Key Extraction: compute base-model key vectors for all ZsRE records.
#
# Extracts input representations to model.layers.6.mlp.down_proj at the
# subject's last token position — the same keys AlphaEdit uses during editing.
# Output is compatible with generate_zsre_orderings.py.
#
# Usage:
#   bash scripts/run_zsre_key_extraction.sh [SEED]
#   bash scripts/run_zsre_key_extraction.sh 42
#
# Output:
#   results/key_vectors/zsre/keys_seed{SEED}_layer6.npz
#
# Requires GPU (~16GB VRAM for Llama-3-8B in float16, ~30 min)

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

# Load environment config
if [[ -f "$PROJECT_DIR/.env" ]]; then
    set -a; source "$PROJECT_DIR/.env"; set +a
fi

MODEL_NAME="${MODEL_NAME:-meta-llama/Meta-Llama-3-8B-Instruct}"
SEED="${1:-42}"
CUDA_DEVICE="${CUDA_DEVICE:-0}"
LAYER="${LAYER:-6}"

RESULT_ROOT="${RESULT_ROOT:-$PROJECT_DIR/results}"
OUTPUT_DIR="${RESULT_ROOT}/key_vectors/zsre"
mkdir -p "$OUTPUT_DIR"

echo "=== ZsRE Key Extraction ==="
echo "  Seed:   $SEED"
echo "  Model:  $MODEL_NAME"
echo "  Layer:  $LAYER"
echo "  Output: $OUTPUT_DIR/keys_seed${SEED}_layer${LAYER}.npz"
echo "  Started: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo ""

cd "$PROJECT_DIR"

# Use the mechanism/compute_keys approach adapted for ZsRE
uv run python -c "
import os, sys, json, time
import numpy as np
import torch
from pathlib import Path
from datetime import datetime, timezone

sys.path.insert(0, 'src/util')
from model_resolve import resolve_model_path

PROJECT = Path('.')
LAYER = ${LAYER}
SEED = ${SEED}
OUTPUT_DIR = Path('${OUTPUT_DIR}')

# Load ZsRE dataset
data_candidates = [
    PROJECT / 'data' / 'dsets' / 'zsre_mend_eval.json',
    PROJECT / 'vendor' / 'AlphaEdit' / 'data' / 'zsre_mend_eval.json',
    Path(os.environ.get('DSETS_ROOT', ''), 'zsre_mend_eval.json'),
]
zsre_path = None
for p in data_candidates:
    if p.exists():
        zsre_path = p
        break
if zsre_path is None:
    print('ERROR: zsre_mend_eval.json not found')
    sys.exit(1)

print(f'Loading ZsRE from {zsre_path}...')
with open(zsre_path) as f:
    raw = json.load(f)
print(f'  Total records: {len(raw)}')

# Load model
model_id = resolve_model_path('${MODEL_NAME}')
print(f'Loading model: {model_id}')
token = os.environ.get('HF_TOKEN')

from transformers import AutoModelForCausalLM, AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained(model_id, token=token)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    model_id, token=token, torch_dtype=torch.float16, device_map='cuda:${CUDA_DEVICE}',
)
model.eval()
print(f'Model loaded on cuda:${CUDA_DEVICE}')

# Register hook on target layer
captured = [None]
def hook_fn(module, input, output):
    captured[0] = input[0].detach()

target_module = model.model.layers[LAYER].mlp.down_proj
target_module.register_forward_hook(hook_fn)

def find_subject_last_token(text, subject):
    subj_start = text.find(subject)
    if subj_start == -1:
        return None
    prefix = text[:subj_start + len(subject)]
    prefix_ids = tokenizer(prefix, return_tensors='pt', padding=False)['input_ids'][0]
    return len(prefix_ids) - 1

# Extract keys for all records
case_ids = []
keys = []
failed = 0
t0 = time.time()

for i, rec in enumerate(raw):
    subject = rec['subject']
    src = rec['src']

    pos = find_subject_last_token(src, subject)
    if pos is None:
        failed += 1
        continue

    inputs = tokenizer(src, return_tensors='pt', padding=False).to(model.device)
    captured[0] = None
    with torch.no_grad():
        model(**inputs)

    if captured[0] is None:
        failed += 1
        continue

    if pos >= captured[0].shape[1]:
        failed += 1
        continue

    key = captured[0][0, pos].cpu().numpy().astype(np.float32)
    case_ids.append(i)
    keys.append(key)

    if (i + 1) % 1000 == 0:
        elapsed = time.time() - t0
        rate = (i + 1) / elapsed
        eta = (len(raw) - i - 1) / rate
        print(f'  [{i+1}/{len(raw)}] {rate:.1f} rec/s, ETA {eta:.0f}s ({failed} failed)')

elapsed = time.time() - t0
print(f'  Done: {len(keys)} keys in {elapsed:.1f}s ({failed} failed)')

# Save locally first, then copy to final destination (FUSE mounts can't handle large temp files)
import shutil, tempfile
_tmp_dir = Path(tempfile.mkdtemp())
_tmp_npz = _tmp_dir / f'keys_seed${SEED}_layer${LAYER}.npz'
_tmp_meta = _tmp_dir / f'keys_seed${SEED}_layer${LAYER}_meta.json'

np.savez_compressed(
    _tmp_npz,
    case_ids=np.array(case_ids, dtype=np.int32),
    keys=np.stack(keys, axis=0),
    layer=np.array(LAYER),
)

with open(_tmp_meta, 'w') as f:
    json.dump({
        'seed': ${SEED},
        'layer': LAYER,
        'model_name': '${MODEL_NAME}',
        'dataset': 'zsre',
        'n_cases': len(keys),
        'n_failed': failed,
        'n_total': len(raw),
        'hidden_dim': keys[0].shape[0] if keys else 0,
        'timestamp': datetime.now(timezone.utc).isoformat(),
        'elapsed_seconds': elapsed,
    }, f, indent=2)

# Copy to final output (S3 FUSE or local)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
out_path = OUTPUT_DIR / f'keys_seed${SEED}_layer${LAYER}.npz'
meta_path = OUTPUT_DIR / f'keys_seed${SEED}_layer${LAYER}_meta.json'
shutil.copyfile(str(_tmp_npz), str(out_path))
shutil.copyfile(str(_tmp_meta), str(meta_path))
shutil.rmtree(_tmp_dir)

print(f'  Saved: {out_path} ({len(keys)} x {keys[0].shape[0]})')
print(f'  Metadata: {meta_path}')
"

echo ""
echo "=== ZsRE Key Extraction complete ==="
echo "  Output: $OUTPUT_DIR/keys_seed${SEED}_layer${LAYER}.npz"
echo "  Finished: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo ""
echo "Next: uv run python src/datasets/generate_zsre_orderings.py --seed $SEED"

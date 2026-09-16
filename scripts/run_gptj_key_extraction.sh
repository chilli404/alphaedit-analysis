#!/usr/bin/env bash
set -euo pipefail

# GPT-J Key Extraction for Matched Ordering
#
# Extracts key vectors from GPT-J-6B for the full MCF dataset via a single
# forward pass with a hook on the MLP down-projection input. Fast (~30 min)
# compared to the full polykernel extractor which runs optimization per record.
#
# Output: results/key_vectors/gptj_full_mcf/keys_seed{SEED}_layer{LAYER}.npz
#
# Usage:
#   bash scripts/run_gptj_key_extraction.sh [SEED]
#   bash scripts/run_gptj_key_extraction.sh 2024

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

# Load environment config
if [[ -f "$PROJECT_DIR/.env" ]]; then
    set -a; source "$PROJECT_DIR/.env"; set +a
fi

export MODEL_NAME="EleutherAI/gpt-j-6b"
SEED="${1:-42}"
CUDA_DEVICE="${CUDA_DEVICE:-0}"
# GPT-J AlphaEdit edits layers 3-8; extract keys from layer 5 (middle of edited range)
LAYER="${KEY_LAYER:-5}"

RESULT_ROOT="${RESULT_ROOT:-$PROJECT_DIR/results}"
OUTPUT_DIR="$RESULT_ROOT/key_vectors/gptj_full_mcf"
mkdir -p "$OUTPUT_DIR"

echo "=== GPT-J Key Extraction (forward-pass hook) ==="
echo "  Seed: $SEED"
echo "  Model: $MODEL_NAME"
echo "  Layer: $LAYER"
echo "  Output: $OUTPUT_DIR/keys_seed${SEED}_layer${LAYER}.npz"
echo "  Started: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo ""

cd "$PROJECT_DIR"

uv run python -c "
import os, sys, json, time
import numpy as np
import torch
from pathlib import Path
from datetime import datetime, timezone

sys.path.insert(0, 'src/util')
from model_resolve import resolve_model_path

LAYER = ${LAYER}
SEED = ${SEED}
OUTPUT_DIR = Path('${OUTPUT_DIR}')

# Load MCF dataset
data_candidates = [
    Path('data/dsets/multi_counterfact.json'),
    Path('vendor/AlphaEdit/data/multi_counterfact.json'),
    Path(os.environ.get('DSETS_ROOT', ''), 'multi_counterfact.json'),
]
mcf_path = None
for p in data_candidates:
    if p.exists():
        mcf_path = p
        break
if mcf_path is None:
    print('ERROR: multi_counterfact.json not found')
    sys.exit(1)

print(f'Loading MCF from {mcf_path}...')
with open(mcf_path) as f:
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

# GPT-J architecture: transformer.h[layer].mlp.fc_out is the down projection
# The input to fc_out is the key vector (after fc_in + activation)
captured = [None]
def hook_fn(module, input, output):
    captured[0] = input[0].detach()

target_module = model.transformer.h[LAYER].mlp.fc_out
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
    subject = rec['requested_rewrite']['subject']
    # Build the prompt the same way AlphaEdit does: subject + relation prompt
    prompt = rec['requested_rewrite']['prompt'].format(rec['requested_rewrite']['subject'])

    pos = find_subject_last_token(prompt, subject)
    if pos is None:
        failed += 1
        continue

    inputs = tokenizer(prompt, return_tensors='pt', padding=False).to(model.device)
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
    case_ids.append(rec['case_id'])
    keys.append(key)

    if (i + 1) % 1000 == 0:
        elapsed = time.time() - t0
        rate = (i + 1) / elapsed
        eta = (len(raw) - i - 1) / rate
        print(f'  [{i+1}/{len(raw)}] {rate:.1f} rec/s, ETA {eta:.0f}s ({failed} failed)')

elapsed = time.time() - t0
print(f'  Done: {len(keys)} keys in {elapsed:.1f}s ({failed} failed)')

# Save locally first, then copy (FUSE safety)
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
        'dataset': 'multi_counterfact',
        'n_cases': len(keys),
        'n_failed': failed,
        'n_total': len(raw),
        'hidden_dim': keys[0].shape[0] if keys else 0,
        'timestamp': datetime.now(timezone.utc).isoformat(),
        'elapsed_seconds': elapsed,
    }, f, indent=2)

# Copy to final output (S3 FUSE or local)
# Use copyfile (not copy) because FUSE mounts don't support chmod
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
echo "=== Key extraction complete ==="
echo "  Output: $OUTPUT_DIR/keys_seed${SEED}_layer${LAYER}.npz"
echo "  Finished: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo ""
echo "Next: generate GPT-J orderings with:"
echo "  uv run python src/datasets/generate_orderings.py --seed $SEED --keys_path $OUTPUT_DIR/keys_seed${SEED}_layer${LAYER}.npz --output_dir results/matched_ordering_gptj"

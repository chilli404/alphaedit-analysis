#!/usr/bin/env python3
"""Quick NSE validation: compare cached v_star, run 200 edits, check efficacy.

Run on a GPU cluster:
    uv run python scripts/test_nse_validation.py

This script:
1. Loads Llama-3-8B in float32 (matching vendor)
2. Computes v_star for case_id=0 fresh and compares to cached
3. Runs NSE on 200 edits (2 batches of 100)
4. Evaluates efficacy of those 200 edits
5. Prints diagnostic comparison
"""
import json
import os
import sys
import numpy as np
import torch
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer

PROJECT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT / "vendor" / "AlphaEdit"))
sys.path.insert(0, str(PROJECT / "src" / "util"))

from model_resolve import resolve_model_path
from dsets import MultiCounterFactDataset
from nse.nse_main import apply_nse_to_model
from nse import NSEHyperParams
from nse.compute_z import compute_z
from util.generate import generate_fast
from util import nethook

# ---- Config ----
MODEL_NAME = "meta-llama/Meta-Llama-3-8B-Instruct"
HPARAMS_PATH = PROJECT / "vendor" / "AlphaEdit" / "hparams" / "NSE" / "Llama3-8B.json"
DATA_DIR = PROJECT / "vendor" / "AlphaEdit" / "data"
CACHE_DIR = None  # Will search for cached v_star

# Find cache
for candidate in [
    PROJECT / "baselines" / "EvoEdit" / "share" / "projects" / "rewriting-knowledge" / "kvs",
    Path("/s3-data/continual-learning/alphaedit/nse_kv_cache"),
]:
    nse_dir = candidate / f"{MODEL_NAME.replace('/', '_')}_NSE"
    if not nse_dir.exists():
        nse_dir = candidate / "meta-llama_Meta-Llama-3-8B-Instruct_NSE"
    if nse_dir.exists():
        CACHE_DIR = nse_dir
        break

print("=" * 70)
print("NSE VALIDATION TEST")
print("=" * 70)

# ---- 1. Load model ----
model_path = resolve_model_path(MODEL_NAME)
print(f"\n[1/5] Loading model: {model_path}")
print(f"  torch_dtype: None (float32 default)")
model = AutoModelForCausalLM.from_pretrained(model_path).cuda()
tok = AutoTokenizer.from_pretrained(model_path)
tok.pad_token = tok.eos_token
model.config._name_or_path = "llama3-8b-instruct"

model_dtype = next(model.parameters()).dtype
print(f"  Model dtype: {model_dtype}")
assert model_dtype == torch.float32, f"Expected float32, got {model_dtype}"

# ---- 2. Load hparams ----
hparams = NSEHyperParams.from_json(HPARAMS_PATH)
print(f"\n[2/5] Hparams: layers={hparams.layers}, neuron_threshold={hparams.neuron_threshold}")

# ---- 3. Compare v_star ----
print(f"\n[3/5] Computing v_star for case_id=0 (fresh vs cached)")

ds = MultiCounterFactDataset(str(DATA_DIR), tok=tok, size=200)
record = ds[0]

# Get context templates
CONTEXT_TEMPLATES_CACHE = [["{}"]] + [
    [
        f.replace("{", " ").replace("}", " ") + ". {}"
        for f in generate_fast(
            model, tok,
            ["The", "Therefore", "Because", "I", "You"],
            n_gen_per_prompt=1,
            max_out_len=10,
        )
    ]
]

request = {"case_id": record["case_id"], **record["requested_rewrite"]}
if request["target_new"]["str"][0] != " ":
    request["target_new"]["str"] = " " + request["target_new"]["str"]

z_layer = hparams.layers[-1]
fresh_z = compute_z(model, tok, request, hparams, z_layer, CONTEXT_TEMPLATES_CACHE)
fresh_z_np = fresh_z.detach().cpu().numpy()

print(f"  Fresh v_star: shape={fresh_z_np.shape}, dtype={fresh_z_np.dtype}, norm={np.linalg.norm(fresh_z_np):.4f}")
print(f"  First 5: {fresh_z_np[:5]}")

if CACHE_DIR:
    cache_file = CACHE_DIR / f"mcf_layer_{z_layer}_clamp_{hparams.clamp_norm_factor}_case_{record['case_id']}.npz"
    if cache_file.exists():
        cached = np.load(cache_file)["v_star"]
        diff = np.linalg.norm(fresh_z_np - cached)
        rel_diff = diff / max(np.linalg.norm(cached), 1e-10)
        print(f"  Cached v_star: norm={np.linalg.norm(cached):.4f}")
        print(f"  Cached first 5: {cached[:5]}")
        print(f"  Absolute diff: {diff:.6f}")
        print(f"  Relative diff: {rel_diff:.6f}")
        if rel_diff < 1e-4:
            print(f"  ✅ MATCH (rel_diff < 1e-4)")
        elif rel_diff < 1e-2:
            print(f"  ⚠️ CLOSE (rel_diff < 1e-2)")
        else:
            print(f"  ❌ MISMATCH (rel_diff = {rel_diff:.4f})")
    else:
        print(f"  Cache file not found: {cache_file}")
else:
    print(f"  No cache directory found")

# ---- 4. Run 200 edits ----
print(f"\n[4/5] Running NSE on 200 edits (2 batches of 100)")

# Initialize cache_c
W_out = nethook.get_parameter(model, f"{hparams.rewrite_module_tmp.format(hparams.layers[-1])}.weight")
cache_c = torch.zeros((len(hparams.layers), W_out.shape[1], W_out.shape[1]), device="cpu")
del W_out

# Set up cache template for v_star
cache_template = None
if CACHE_DIR:
    cache_template = str(CACHE_DIR / f"mcf_layer_{{}}_clamp_{{}}_case_{{}}.npz")
    print(f"  Using cached v_star from: {CACHE_DIR}")
else:
    print(f"  Computing v_star fresh (no cache)")

records = list(ds)
batch_size = 100

for batch_idx in range(2):
    batch = records[batch_idx * batch_size : (batch_idx + 1) * batch_size]
    requests = [{"case_id": r["case_id"], **r["requested_rewrite"]} for r in batch]

    print(f"\n  Batch {batch_idx}: {len(requests)} edits")

    model, cache_c = apply_nse_to_model(
        model, tok, requests, hparams,
        cache_c=cache_c,
        cache_template=cache_template,
        return_orig_weights=False,
    )
    print(f"  cache_c norm: {cache_c.norm():.2f}")

# ---- 5. Evaluate ----
print(f"\n[5/5] Evaluating 200 edited records")

correct_pp = 0
correct_am = 0
total = 0

for record in records[:200]:
    subject = record["requested_rewrite"]["subject"]
    prompt = record["requested_rewrite"]["prompt"].format(subject)
    target_new = record["requested_rewrite"]["target_new"]["str"]
    target_true = record["requested_rewrite"]["target_true"]["str"]

    if target_new[0] != " ":
        target_new = " " + target_new
    if target_true[0] != " ":
        target_true = " " + target_true

    # Tokenize
    new_toks = tok(prompt + target_new, return_tensors="pt").input_ids.cuda()
    true_toks = tok(prompt + target_true, return_tensors="pt").input_ids.cuda()
    prefix_len = tok(prompt, return_tensors="pt").input_ids.shape[1]

    with torch.no_grad():
        new_logits = model(new_toks).logits[0]
        true_logits = model(true_toks).logits[0]

    # Prob-pref: NLL comparison
    new_nll = 0
    for j in range(prefix_len, new_toks.shape[1]):
        new_nll -= torch.log_softmax(new_logits[j-1], dim=-1)[new_toks[0, j]].item()
    new_nll /= max(new_toks.shape[1] - prefix_len, 1)

    true_nll = 0
    for j in range(prefix_len, true_toks.shape[1]):
        true_nll -= torch.log_softmax(true_logits[j-1], dim=-1)[true_toks[0, j]].item()
    true_nll /= max(true_toks.shape[1] - prefix_len, 1)

    if new_nll < true_nll:
        correct_pp += 1

    # Argmax
    all_correct = True
    for j in range(prefix_len, new_toks.shape[1]):
        if new_logits[j-1].argmax().item() != new_toks[0, j].item():
            all_correct = False
            break
    if all_correct:
        correct_am += 1

    total += 1

    if total % 50 == 0:
        print(f"  [{total}/200] PP: {correct_pp/total*100:.1f}%  AM: {correct_am/total*100:.1f}%")

print(f"\n{'='*70}")
print(f"RESULTS: 200 edits (2 batches of 100)")
print(f"  Prob-pref efficacy: {correct_pp/total*100:.1f}%")
print(f"  Argmax efficacy:    {correct_am/total*100:.1f}%")
print(f"{'='*70}")
print(f"\nExpected (published NSE at 2K): ~99% prob-pref")
print(f"If we get >>90%, issue is degradation at scale.")
print(f"If we get <<90%, issue is in our implementation.")

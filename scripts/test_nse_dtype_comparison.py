#!/usr/bin/env python3
"""Compare NSE with float32 vs bfloat16 v_star cache.

Builds 100 v_star in each dtype, runs 100 edits with each, compares efficacy.
"""
import json
import os
import sys
import numpy as np
import torch
import time
from copy import deepcopy
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
from nse.nse_main import get_context_templates
from util import nethook

MODEL_NAME = "meta-llama/Meta-Llama-3-8B-Instruct"
HPARAMS_PATH = PROJECT / "vendor" / "AlphaEdit" / "hparams" / "NSE" / "Llama3-8B.json"
N_EDITS = 100

model_path = resolve_model_path(MODEL_NAME)
hparams = NSEHyperParams.from_json(HPARAMS_PATH)
z_layer = hparams.layers[-1]

print("=" * 70)
print("NSE DTYPE COMPARISON TEST")
print("=" * 70)

# Load dataset
tok = AutoTokenizer.from_pretrained(model_path)
tok.pad_token = tok.eos_token
ds = MultiCounterFactDataset(str(PROJECT / "vendor" / "AlphaEdit" / "data"), tok=tok, size=N_EDITS)
records = list(ds)
requests = [{"case_id": r["case_id"], **r["requested_rewrite"]} for r in records]
for req in requests:
    if req["target_new"]["str"][0] != " ":
        req["target_new"]["str"] = " " + req["target_new"]["str"]


def build_cache(model, tok, dtype_label):
    """Build v_star for N_EDITS records."""
    ctx = get_context_templates(model, tok)
    cache = {}
    t0 = time.time()
    for i, req in enumerate(requests):
        z = compute_z(model, tok, req, hparams, z_layer, ctx)
        cache[req["case_id"]] = z.detach().cpu().numpy()
        if (i + 1) % 25 == 0:
            print(f"  [{dtype_label}] {i+1}/{N_EDITS} v_star computed")
    elapsed = time.time() - t0
    norms = [np.linalg.norm(v) for v in cache.values()]
    print(f"  [{dtype_label}] Done in {elapsed:.0f}s. Mean norm: {np.mean(norms):.4f}")
    return cache


def write_cache_files(cache, cache_dir):
    """Write v_star cache to npz files."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    for case_id, z in cache.items():
        fname = f"mcf_layer_{z_layer}_clamp_{hparams.clamp_norm_factor}_case_{case_id}.npz"
        np.savez(cache_dir / fname, v_star=z)


def run_nse(dtype_label, cache_dir):
    """Load model in float32, run NSE with given cache, evaluate."""
    print(f"\n--- Running NSE with {dtype_label} cache ---")
    model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.float32).cuda()
    model.config._name_or_path = "llama3-8b-instruct"
    print(f"  Model dtype: {next(model.parameters()).dtype}")

    cache_template = str(cache_dir / f"mcf_layer_{{}}_clamp_{{}}_case_{{}}.npz")

    W_out = nethook.get_parameter(model, f"{hparams.rewrite_module_tmp.format(hparams.layers[-1])}.weight")
    cache_c = torch.zeros((len(hparams.layers), W_out.shape[1], W_out.shape[1]), device="cpu")
    del W_out

    t0 = time.time()
    model, cache_c = apply_nse_to_model(
        model, tok, requests, hparams,
        cache_c=cache_c,
        cache_template=cache_template,
        return_orig_weights=False,
    )
    edit_time = time.time() - t0
    print(f"  Editing took {edit_time:.0f}s")

    # Evaluate
    correct_pp = 0
    correct_am = 0
    total = 0

    for record in records:
        subject = record["requested_rewrite"]["subject"]
        prompt = record["requested_rewrite"]["prompt"].format(subject)
        target_new = record["requested_rewrite"]["target_new"]["str"]
        target_true = record["requested_rewrite"]["target_true"]["str"]
        if target_new[0] != " ":
            target_new = " " + target_new
        if target_true[0] != " ":
            target_true = " " + target_true

        new_toks = tok(prompt + target_new, return_tensors="pt").input_ids.cuda()
        true_toks = tok(prompt + target_true, return_tensors="pt").input_ids.cuda()
        prefix_len = tok(prompt, return_tensors="pt").input_ids.shape[1]

        with torch.no_grad():
            new_logits = model(new_toks).logits[0]
            true_logits = model(true_toks).logits[0]

        new_nll = 0
        for j in range(prefix_len, new_toks.shape[1]):
            new_nll -= torch.log_softmax(new_logits[j - 1], dim=-1)[new_toks[0, j]].item()
        new_nll /= max(new_toks.shape[1] - prefix_len, 1)

        true_nll = 0
        for j in range(prefix_len, true_toks.shape[1]):
            true_nll -= torch.log_softmax(true_logits[j - 1], dim=-1)[true_toks[0, j]].item()
        true_nll /= max(true_toks.shape[1] - prefix_len, 1)

        if new_nll < true_nll:
            correct_pp += 1

        all_correct = True
        for j in range(prefix_len, new_toks.shape[1]):
            if new_logits[j - 1].argmax().item() != new_toks[0, j].item():
                all_correct = False
                break
        if all_correct:
            correct_am += 1

        total += 1

    del model
    torch.cuda.empty_cache()

    pp = correct_pp / total * 100
    am = correct_am / total * 100
    print(f"  [{dtype_label}] Prob-pref: {pp:.1f}%  Argmax: {am:.1f}%")
    return pp, am


# --- Step 1: Build v_star caches in each dtype ---
print("\n[1/4] Building float32 v_star cache")
model_fp32 = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.float32).cuda()
model_fp32.config._name_or_path = "llama3-8b-instruct"
print(f"  Model dtype: {next(model_fp32.parameters()).dtype}")
cache_fp32 = build_cache(model_fp32, tok, "fp32")
del model_fp32
torch.cuda.empty_cache()

print("\n[2/4] Building bfloat16 v_star cache")
model_bf16 = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.bfloat16).cuda()
model_bf16.config._name_or_path = "llama3-8b-instruct"
print(f"  Model dtype: {next(model_bf16.parameters()).dtype}")
cache_bf16 = build_cache(model_bf16, tok, "bf16")
del model_bf16
torch.cuda.empty_cache()

# Compare the caches
print("\n--- Cache comparison ---")
diffs = []
for cid in cache_fp32:
    z32 = cache_fp32[cid]
    z16 = cache_bf16[cid]
    rel = np.linalg.norm(z32 - z16) / max(np.linalg.norm(z32), 1e-10)
    diffs.append(rel)
print(f"  Mean relative diff: {np.mean(diffs):.4f}")
print(f"  Max relative diff:  {np.max(diffs):.4f}")
print(f"  Min relative diff:  {np.min(diffs):.4f}")
print(f"  fp32 mean norm: {np.mean([np.linalg.norm(v) for v in cache_fp32.values()]):.4f}")
print(f"  bf16 mean norm: {np.mean([np.linalg.norm(v) for v in cache_bf16.values()]):.4f}")

# Write cache files
fp32_dir = Path("/tmp/nse_cache_fp32")
bf16_dir = Path("/tmp/nse_cache_bf16")
write_cache_files(cache_fp32, fp32_dir)
write_cache_files(cache_bf16, bf16_dir)

# --- Step 2: Run NSE with each cache ---
print("\n[3/4] NSE with float32 cache")
pp_fp32, am_fp32 = run_nse("fp32", fp32_dir)

print("\n[4/4] NSE with bfloat16 cache")
pp_bf16, am_bf16 = run_nse("bf16", bf16_dir)

# --- Summary ---
print("\n" + "=" * 70)
print("SUMMARY")
print("=" * 70)
print(f"  {'Metric':<25} {'fp32 cache':<15} {'bf16 cache':<15}")
print(f"  {'─' * 25} {'─' * 15} {'─' * 15}")
print(f"  {'Prob-pref efficacy':<25} {pp_fp32:<15.1f} {pp_bf16:<15.1f}")
print(f"  {'Argmax efficacy':<25} {am_fp32:<15.1f} {am_bf16:<15.1f}")
print(f"  {'v_star mean rel diff':<25} {np.mean(diffs):.4f}")
print("=" * 70)

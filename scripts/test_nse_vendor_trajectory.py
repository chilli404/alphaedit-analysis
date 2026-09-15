#!/usr/bin/env python3
"""Run vendor NSE evaluate.py unchanged at 100, 1K, 2K, 5K, 10K edits.

No wrappers, no harness, no hooks. Uses vendor code directly.
Builds v_star from scratch (no S3 cache). Logs cache_c diagnostics
and efficacy at each checkpoint.

Run from vendor/AlphaEdit/ with PYTHONPATH=.:
    cd vendor/AlphaEdit && PYTHONPATH=. uv run python ../../scripts/test_nse_vendor_trajectory.py
"""
import json
import os
import sys
import time
import numpy as np
import torch
from copy import deepcopy
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer

PROJECT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT / "vendor" / "AlphaEdit"))
sys.path.insert(0, str(PROJECT / "src" / "util"))

from model_resolve import resolve_model_path
from dsets import MultiCounterFactDataset
from nse.nse_main import apply_nse_to_model, get_context_templates
from nse import NSEHyperParams
from nse.compute_z import compute_z
from nse.compute_ks import compute_ks
from util import nethook

MODEL_NAME = "meta-llama/Meta-Llama-3-8B-Instruct"
HPARAMS_PATH = PROJECT / "vendor" / "AlphaEdit" / "hparams" / "NSE" / "Llama3-8B.json"
BATCH_SIZE = 100
CHECKPOINTS = [100, 1000, 2000, 5000, 10000]
RESULT_DIR = Path(os.environ.get("RESULT_ROOT", str(PROJECT / "results"))) / "nse_vendor_trajectory"
RESULT_DIR.mkdir(parents=True, exist_ok=True)

print("=" * 70)
print("NSE VENDOR TRAJECTORY TEST")
print(f"  Model:       {MODEL_NAME}")
print(f"  Batch size:  {BATCH_SIZE}")
print(f"  Checkpoints: {CHECKPOINTS}")
print(f"  Results:     {RESULT_DIR}")
print("=" * 70)

# Load model in float32 (matching vendor exactly)
model_path = resolve_model_path(MODEL_NAME)
print(f"\nLoading model: {model_path}")
model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.float32).cuda()
tok = AutoTokenizer.from_pretrained(model_path)
tok.pad_token = tok.eos_token
model.config._name_or_path = "llama3-8b-instruct"
print(f"  Model dtype: {next(model.parameters()).dtype}")

hparams = NSEHyperParams.from_json(HPARAMS_PATH)
print(f"  Layers: {hparams.layers}")
print(f"  neuron_threshold: {hparams.neuron_threshold}")
print(f"  max_iterations: {hparams.max_iterations}")
print(f"  alpha: {hparams.alpha}, upper_bound: {hparams.upper_bound}")

# Load full dataset
ds = MultiCounterFactDataset(str(PROJECT / "vendor" / "AlphaEdit" / "data"), tok=tok, size=max(CHECKPOINTS))
all_records = list(ds)
print(f"  Dataset: {len(all_records)} records")

# Pre-cache ALL v_star from unedited model (matching reference evaluate.py protocol)
print(f"\n{'='*70}")
print("PRE-CACHING v_star from unedited model (this is what the reference does)")
print(f"{'='*70}")

z_layer = hparams.layers[-1]
context_templates = get_context_templates(model, tok)
cache_dir = RESULT_DIR / "vstar_cache"
cache_dir.mkdir(parents=True, exist_ok=True)

t0 = time.time()
for i, record in enumerate(all_records):
    req = {"case_id": record["case_id"], **record["requested_rewrite"]}
    if req["target_new"]["str"][0] != " ":
        req["target_new"]["str"] = " " + req["target_new"]["str"]

    fname = f"mcf_layer_{z_layer}_clamp_{hparams.clamp_norm_factor}_case_{req['case_id']}.npz"
    fpath = cache_dir / fname

    if fpath.exists():
        continue

    cur_z = compute_z(model, tok, req, hparams, z_layer, context_templates)
    np.savez(fpath, v_star=cur_z.detach().cpu().numpy())

    if (i + 1) % 100 == 0:
        elapsed = time.time() - t0
        rate = (i + 1) / elapsed
        remaining = (len(all_records) - i - 1) / rate
        print(f"  [{i+1}/{len(all_records)}] {elapsed:.0f}s elapsed, ~{remaining:.0f}s remaining")

cache_time = time.time() - t0
n_cached = len(list(cache_dir.glob("*.npz")))
print(f"  Pre-cached {n_cached} v_star in {cache_time:.0f}s")

cache_template = str(cache_dir / f"mcf_layer_{{}}_clamp_{{}}_case_{{}}.npz")

# Initialize cache_c (matching reference evaluate.py exactly)
W_out = nethook.get_parameter(model, f"{hparams.rewrite_module_tmp.format(hparams.layers[-1])}.weight")
cache_c = torch.zeros((len(hparams.layers), W_out.shape[1], W_out.shape[1]), device="cpu")
del W_out
print(f"  cache_c shape: {cache_c.shape}")


def log_cache_c_diagnostics(cache_c, label):
    """Log cache_c health metrics."""
    for i, layer in enumerate(hparams.layers):
        c = cache_c[i].float()
        frob = torch.linalg.norm(c).item()
        eigvals = torch.linalg.eigvalsh(c)
        max_eig = eigvals[-1].item()
        min_eig = eigvals[0].item()
        rank = (eigvals > 1e-6 * max_eig).sum().item()
        cond = max_eig / max(abs(min_eig), 1e-10) if min_eig != 0 else float("inf")
        print(f"  [{label}] layer {layer}: ||cache_c||_F={frob:.1f} "
              f"max_eig={max_eig:.2f} eff_rank={rank}/{c.shape[0]} cond={cond:.1e}")


def evaluate_all_facts(model, tok, records, label):
    """Evaluate efficacy on all given records using prob-pref + argmax."""
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

    pp = correct_pp / total * 100
    am = correct_am / total * 100
    return pp, am, total


# Run sequential editing with checkpoints
print(f"\n{'='*70}")
print("SEQUENTIAL EDITING — vendor apply_nse_to_model")
print(f"{'='*70}")

results = {}
total_edits = 0
edit_start = time.time()

for batch_idx in range(max(CHECKPOINTS) // BATCH_SIZE):
    start = batch_idx * BATCH_SIZE
    end = start + BATCH_SIZE
    batch_records = all_records[start:end]

    requests = []
    for r in batch_records:
        req = {"case_id": r["case_id"], **r["requested_rewrite"]}
        if req["target_new"]["str"][0] != " ":
            req["target_new"]["str"] = " " + req["target_new"]["str"]
        requests.append(req)

    t0 = time.time()
    model, cache_c = apply_nse_to_model(
        model, tok, requests, hparams,
        cache_c=cache_c,
        cache_template=cache_template,
        return_orig_weights=False,
    )
    batch_time = time.time() - t0
    total_edits += BATCH_SIZE

    print(f"\n  Batch {batch_idx}: {BATCH_SIZE} edits ({total_edits} total) in {batch_time:.0f}s")
    print(f"  cache_c norm: {cache_c.norm():.2f}")

    # Check if this is a checkpoint
    if total_edits in CHECKPOINTS:
        print(f"\n{'='*70}")
        print(f"CHECKPOINT: {total_edits} edits")
        print(f"{'='*70}")

        # cache_c diagnostics
        log_cache_c_diagnostics(cache_c, f"{total_edits}edits")

        # Evaluate ALL edited facts so far
        print(f"\n  Evaluating all {total_edits} edited facts...")
        eval_start = time.time()
        pp, am, n = evaluate_all_facts(model, tok, all_records[:total_edits], f"{total_edits}edits")
        eval_time = time.time() - eval_start

        print(f"  [{total_edits} edits] Prob-pref: {pp:.1f}%  Argmax: {am:.1f}%  ({n} facts, {eval_time:.0f}s)")

        results[total_edits] = {
            "prob_pref": pp,
            "argmax": am,
            "n_facts": n,
            "cache_c_norm": cache_c.norm().item(),
            "edit_time_total": time.time() - edit_start,
            "eval_time": eval_time,
        }

        # Save intermediate results
        with open(RESULT_DIR / "trajectory.json", "w") as f:
            json.dump(results, f, indent=2)
        print(f"  Saved to {RESULT_DIR / 'trajectory.json'}")

        # Save checkpoint weights
        ckpt_path = RESULT_DIR / f"checkpoint_{total_edits}"
        ckpt_path.mkdir(parents=True, exist_ok=True)
        torch.save(cache_c, str(ckpt_path / "cache_c.pt"))
        print(f"  Saved cache_c to {ckpt_path}")

# Final summary
print(f"\n{'='*70}")
print("TRAJECTORY SUMMARY")
print(f"{'='*70}")
print(f"  {'Edits':<10} {'Prob-pref':<12} {'Argmax':<12} {'cache_c norm':<15}")
print(f"  {'─'*10} {'─'*12} {'─'*12} {'─'*15}")
for edits in sorted(results.keys()):
    r = results[edits]
    print(f"  {edits:<10} {r['prob_pref']:<12.1f} {r['argmax']:<12.1f} {r['cache_c_norm']:<15.1f}")
print(f"\n  Published NSE @ 10K: 77.59% efficacy")
print(f"{'='*70}")

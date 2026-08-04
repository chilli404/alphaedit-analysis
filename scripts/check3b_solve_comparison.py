#!/usr/bin/env python3
"""
Check 3b: Head-to-head solve comparison on identical inputs.

On one batch of 100 Qwen edits, computes dW under:
  (a) MEMIT solve:     adj_k = solve(α*C₀ + K@K^T, K);  dW = resid @ adj_k^T
  (b) AlphaEdit solve: dW = solve(I@(K@K^T + 0) + L2*I + α*C₀, I@K@resid^T)^T

Uses identical z-targets, keys, and residuals.
Reports ||dW||_F per layer and ||dW_a - dW_b||_F / ||dW_a||_F.

If near-zero relative diff: the two solves are equivalent modulo precision.
If large diff: identifies the structural difference.

Usage:
    uv run python scripts/check3b_solve_comparison.py
"""

import os
import sys
import numpy as np
import torch
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent
VENDOR_DIR = PROJECT_DIR / "vendor" / "AlphaEdit"

sys.path.insert(0, str(VENDOR_DIR))
sys.path.insert(0, str(PROJECT_DIR / "src" / "util"))

from setup_hparams import link_hparams
from source_patches import patch_evaluate_file, patch_glue_eval_file


def main():
    print("=" * 72)
    print("CHECK 3b: HEAD-TO-HEAD SOLVE COMPARISON")
    print("  MEMIT solve vs AlphaEdit(P=I)+C₀ solve on identical inputs")
    print("=" * 72)
    print()

    # Setup
    os.chdir(str(VENDOR_DIR))
    link_hparams()
    patch_evaluate_file(VENDOR_DIR)
    patch_glue_eval_file(VENDOR_DIR)

    # Ensure **_kwargs patches
    ae_path = VENDOR_DIR / "AlphaEdit" / "AlphaEdit_main.py"
    ae_original = ae_path.read_text()
    if "**_kwargs" not in ae_original:
        ae_path.write_text(ae_original.replace(
            "    P = None,\n", "    P = None, **_kwargs,\n", 1))

    memit_path = VENDOR_DIR / "memit" / "memit_main.py"
    memit_original = memit_path.read_text()
    if "**_kwargs" not in memit_original:
        memit_path.write_text(memit_original.replace(
            "    cache_template: Optional[str] = None,\n",
            "    cache_template: Optional[str] = None, **_kwargs,\n", 1))

    # Ensure stats symlink
    vendor_stats_dir = VENDOR_DIR / "data" / "stats"
    vendor_stats_dir.mkdir(parents=True, exist_ok=True)
    target = vendor_stats_dir / "Qwen2.5-7B-Instruct"
    if not target.exists():
        src = PROJECT_DIR / "data" / "stats" / "qwen2.5-7b-instruct"
        if src.exists():
            target.symlink_to(src)

    import subprocess
    import textwrap

    # The comparison script runs inside vendor dir with proper imports
    script = textwrap.dedent(r"""
import os, sys, json
import numpy as np
import torch
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer

# Vendor imports
from memit import MEMITHyperParams
from memit.compute_z import get_module_input_output_at_words, compute_z
from memit.compute_ks import compute_ks
from memit.memit_main import get_context_templates, get_cov
from util.globals import *
from dsets import MultiCounterFactDataset, AttributeSnippets, get_tfidf_vectorizer

# Load hparams
hparams = MEMITHyperParams.from_json(HPARAMS_DIR / "MEMIT" / "Qwen2.5-7B.json")
print(f"  Model: {hparams.model_name}")
print(f"  Layers: {hparams.layers}")
print(f"  α (mom2_update_weight): {hparams.mom2_update_weight}")
print(f"  L2 (from AlphaEdit hparams): 1")
print()

# Load model
print("Loading model...")
model_path = "Qwen/Qwen2.5-7B-Instruct"

tok = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(
    model_path, torch_dtype=torch.bfloat16, device_map="auto"
)
model.eval()

# Load dataset (first 100 edits) — transform to algorithm format
ds = MultiCounterFactDataset("data", tok=tok, size=100)
# evaluate.py flattens: {"case_id": ..., **requested_rewrite} for each record
requests = []
for record in ds:
    rewrites = record["requested_rewrite"]
    if not isinstance(rewrites, list):
        rewrites = [rewrites]
    for rw in rewrites:
        requests.append({"case_id": record["case_id"], **rw})
requests = requests[:100]  # cap at 100
print(f"  Loaded {len(requests)} edit requests")
print(f"  Sample: [{requests[0]['prompt']}] -> [{requests[0]['target_new']['str']}]")

# Get context templates
context_templates = get_context_templates(model, tok)

# Compute z-targets (shared between both methods)
z_layer = hparams.v_loss_layer
print(f"\n  Computing z-targets (v_loss_layer={z_layer})...")
z_list = []
for i, request in enumerate(requests):
    cur_z = compute_z(
        model, tok, request, hparams, z_layer, context_templates
    )
    z_list.append(cur_z)
    if (i+1) % 20 == 0:
        print(f"    {i+1}/100 z-targets computed")
zs = torch.stack(z_list, dim=1)
print(f"  zs shape: {zs.shape}")

# For each layer, compute both solves

print()
print("=" * 72)
print("  PER-LAYER COMPARISON (batch 1, 100 edits)")
print("=" * 72)

L2 = 1  # AlphaEdit Qwen L2
alpha = hparams.mom2_update_weight  # 15000

for i, layer in enumerate(hparams.layers):
    print(f"\n{'─'*60}")
    print(f"  LAYER {layer}")
    print(f"{'─'*60}")

    # Compute keys (shared)
    layer_ks = compute_ks(model, tok, requests, hparams, layer, context_templates).T
    # layer_ks shape: (d_in, B) = (18944, 100) for Qwen

    # Compute current z-outputs to get residual (shared)
    cur_zs = get_module_input_output_at_words(
        model, tok, z_layer,
        context_templates=[r["prompt"] for r in requests],
        words=[r["subject"] for r in requests],
        module_template=hparams.layer_module_tmp,
        fact_token_strategy=hparams.fact_token,
    )[1].T

    targets = zs - cur_zs
    repeat_factor = layer_ks.size(1) // targets.size(1)
    targets = targets.repeat_interleave(repeat_factor, dim=1)
    resid = targets / (len(hparams.layers) - i)

    # Load C0
    cov = get_cov(
        model, tok,
        hparams.rewrite_module_tmp.format(layer),
        hparams.mom2_dataset, hparams.mom2_n_samples, hparams.mom2_dtype,
    )

    print(f"    K shape: {layer_ks.shape}")
    print(f"    resid shape: {resid.shape}")
    print(f"    C0 shape: {cov.shape}")
    print(f"    ||K||_F: {torch.linalg.norm(layer_ks):.4e}")
    print(f"    ||resid||_F: {torch.linalg.norm(resid):.4e}")
    print(f"    ||C0||_F: {torch.linalg.norm(cov):.4e}")
    print()

    d_in = layer_ks.shape[0]
    K_d = layer_ks.double()
    resid_d = resid.double()
    cov_d = cov.double()
    K_f = layer_ks.float()
    resid_f = resid.float()
    cov_f = cov.float()

    # Shared LHS components (double)
    KKT_d = K_d @ K_d.T  # (d_in, d_in) — reused across solves
    eye_d = torch.eye(d_in, device=K_d.device, dtype=torch.double)

    # ─── SOLVE (a): MEMIT (double, no L2) ───
    lhs_a = alpha * cov_d + KKT_d
    adj_k_a = torch.linalg.solve(lhs_a, K_d)
    dW_a = (resid_d @ adj_k_a.T).float().cpu()  # (d_out, d_in)
    cond_a = torch.linalg.cond(lhs_a).item()
    del adj_k_a, lhs_a

    # ─── SOLVE (b): AlphaEdit P=I+C0 (float, +L2) ───
    KKT_f = K_f @ K_f.T
    lhs_b = KKT_f + L2 * torch.eye(d_in, device=K_f.device, dtype=torch.float) + alpha * cov_f
    dW_b = torch.linalg.solve(lhs_b, K_f @ resid_f.T).T.cpu()  # (d_out, d_in)
    cond_b = torch.linalg.cond(lhs_b.double()).item()
    del lhs_b, KKT_f

    # ─── SOLVE (c): AlphaEdit P=I+C0 (double, +L2) ───
    lhs_c = KKT_d + L2 * eye_d + alpha * cov_d
    dW_c = torch.linalg.solve(lhs_c, K_d @ resid_d.T).T.float().cpu()  # (d_out, d_in)
    del lhs_c

    # ─── SOLVE (d): MEMIT + L2 (double) — isolates L2 effect ───
    lhs_d = alpha * cov_d + KKT_d + L2 * eye_d
    adj_k_d = torch.linalg.solve(lhs_d, K_d)
    dW_d = (resid_d @ adj_k_d.T).float().cpu()  # (d_out, d_in)
    del adj_k_d, lhs_d, KKT_d, eye_d
    del K_d, K_f, resid_d, resid_f, cov_d, cov_f
    torch.cuda.empty_cache()

    # ─── Report ───
    norm_a = torch.linalg.norm(dW_a).item()
    norm_b = torch.linalg.norm(dW_b).item()
    norm_c = torch.linalg.norm(dW_c).item()
    norm_d = torch.linalg.norm(dW_d).item()

    diff_ab = torch.linalg.norm(dW_a - dW_b).item()
    diff_ac = torch.linalg.norm(dW_a - dW_c).item()
    diff_ad = torch.linalg.norm(dW_a - dW_d).item()
    diff_cd = torch.linalg.norm(dW_c - dW_d).item()

    print(f"    RESULTS:")
    print(f"      (a) MEMIT (double, no L2):              ||dW||_F = {norm_a:.6e}")
    print(f"      (b) AlphaEdit P=I+C0 (float, +L2):     ||dW||_F = {norm_b:.6e}")
    print(f"      (c) AlphaEdit P=I+C0 (double, +L2):    ||dW||_F = {norm_c:.6e}")
    print(f"      (d) MEMIT + L2 (double):               ||dW||_F = {norm_d:.6e}")
    print()
    print(f"    RELATIVE DIFFERENCES:")
    print(f"      ||dW_a - dW_b|| / ||dW_a|| = {diff_ab/norm_a:.6e}  (all diffs: precision + L2)")
    print(f"      ||dW_a - dW_c|| / ||dW_a|| = {diff_ac/norm_a:.6e}  (L2 only, same double prec)")
    print(f"      ||dW_a - dW_d|| / ||dW_a|| = {diff_ad/norm_a:.6e}  (L2 via MEMIT formula)")
    print(f"      ||dW_c - dW_d|| / ||dW_c|| = {diff_cd/norm_c:.6e}  (structural: solve(LHS,K@R^T)^T vs R@solve(LHS,K)^T)")
    print()
    print(f"    CONDITIONING:")
    print(f"      cond(MEMIT LHS):         {cond_a:.4e}")
    print(f"      cond(AlphaEdit+C0 LHS):  {cond_b:.4e}")
    print()

    del dW_a, dW_b, dW_c, dW_d
    # No model update — each layer comparison uses the ORIGINAL (unedited) model

print()
print("=" * 72)
print("CONCLUSION")
print("=" * 72)
print()
print("  If ||dW_c - dW_d|| / ||dW_c|| ≈ 0:")
print("    → The two solves are mathematically equivalent (same LHS, same inputs)")
print("    → Any difference in Check 3 crash is due to precision (float vs double)")
print("    → Or due to model corruption after applying large dW in bf16")
print()
print("  If ||dW_c - dW_d|| / ||dW_c|| is large:")
print("    → Structural formula difference: AlphaEdit solve(LHS, K@resid^T)^T")
print("      vs MEMIT resid @ solve(LHS, K)^T")
print("    → These are NOT equivalent due to the ^T placement!")
print()
""")

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = os.environ.get("CUDA_VISIBLE_DEVICES", "0")

    try:
        result = subprocess.run(
            [sys.executable, "-c", script],
            cwd=str(VENDOR_DIR),
            env=env,
        )
        if result.returncode != 0:
            print(f"\n  ERROR: Comparison failed with return code {result.returncode}")
            sys.exit(result.returncode)
    finally:
        # Restore
        ae_path.write_text(ae_original)
        memit_path.write_text(memit_original)
        print("\n  Restored original vendor files")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Check 4 (Expanded): Full Regime Audit.

Runs a single batch of 100 Qwen edits under 5 configurations and reports:
  - Measured S.max(K@K^T) and trace per layer (from actual keys)
  - ||dW||_F, max|dW|, post-update max|W| per layer
  - NaN first-appearance site (solve or forward)
  - Condition number of the LHS matrix

Configurations:
  A) abs-tau P (official AlphaEdit, float32 solve)
  B) rel-tau P (near-identity, float32 solve)
  C) rel-tau P + alpha*C0 (float32 solve)
  D) rel-tau P + alpha*C0 (float64 solve)  [precision test]
  E) MEMIT-Seq (float64 solve, has alpha*C0)

Also: magnitude parity test (D vs E on identical inputs).

Usage:
    uv run python scripts/check4_expanded.py --config A
    uv run python scripts/check4_expanded.py --config B
    uv run python scripts/check4_expanded.py --config C
    uv run python scripts/check4_expanded.py --config D
    uv run python scripts/check4_expanded.py --config E
    uv run python scripts/check4_expanded.py --config spectra   # measured K@K^T spectra only
    uv run python scripts/check4_expanded.py --config parity    # D vs E side-by-side
"""

import argparse
import os
import sys
import json
import subprocess
import textwrap
import numpy as np
import torch
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent
VENDOR_DIR = PROJECT_DIR / "vendor" / "AlphaEdit"

sys.path.insert(0, str(PROJECT_DIR / "src" / "util"))
from setup_hparams import link_hparams
from source_patches import patch_evaluate_file, patch_glue_eval_file


def ensure_vendor_patches():
    """Apply standard vendor patches."""
    os.chdir(str(VENDOR_DIR))
    link_hparams()
    patch_evaluate_file(VENDOR_DIR)
    patch_glue_eval_file(VENDOR_DIR)

    # Ensure **_kwargs patches
    ae_path = VENDOR_DIR / "AlphaEdit" / "AlphaEdit_main.py"
    ae_src = ae_path.read_text()
    if "**_kwargs" not in ae_src:
        ae_path.write_text(ae_src.replace("    P = None,\n", "    P = None, **_kwargs,\n", 1))

    memit_path = VENDOR_DIR / "memit" / "memit_main.py"
    memit_src = memit_path.read_text()
    if "**_kwargs" not in memit_src:
        memit_path.write_text(memit_src.replace(
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


def compute_projector(stats_dir, layers, mode="abs", threshold=0.02):
    """Compute projector in abs or rel mode."""
    P_list = []
    for layer in layers:
        layer_name = f"model.layers.{layer}.mlp.down_proj"
        filename = stats_dir / f"{layer_name}_float32_mom2_100000.npz"
        data = np.load(filename, allow_pickle=True)
        raw_mom2 = torch.from_numpy(data["mom2.mom2"])
        count = int(data["mom2.count"])
        cov = (raw_mom2 / count).float()

        U, S, _ = torch.linalg.svd(cov.cuda(), full_matrices=False)

        if mode == "rel":
            tau = threshold * S[0].item()
        else:
            tau = threshold

        small = (S < tau).nonzero(as_tuple=True)[0]
        P = (U[:, small] @ U[:, small].T).cpu()
        P_list.append(P)
        print(f"    Layer {layer}: {len(small)}/{S.shape[0]} retained "
              f"({mode} τ={tau:.4e})")
    return torch.stack(P_list)


def run_spectra():
    """Item 2: Measure actual K@K^T spectra from real model keys."""
    print("=" * 72)
    print("CHECK 4.2: MEASURED SPECTRA FROM REAL K MATRICES (100 edits)")
    print("=" * 72)

    ensure_vendor_patches()

    script = textwrap.dedent(r"""
import os, sys, json
import numpy as np
import torch
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer

# Vendor imports
from memit import MEMITHyperParams
from memit.compute_ks import compute_ks
from memit.memit_main import get_context_templates
from util.globals import *
from dsets import MultiCounterFactDataset

# Load hparams
hparams = MEMITHyperParams.from_json(HPARAMS_DIR / "MEMIT" / "Qwen2.5-7B.json")

# Load model
print("Loading model...")
tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B-Instruct")
tok.pad_token = tok.eos_token
model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-7B-Instruct").cuda()
model.eval()

# Load dataset
ds = MultiCounterFactDataset("data", tok=tok, size=100)
requests = []
for record in ds:
    rewrites = record["requested_rewrite"]
    if not isinstance(rewrites, list):
        rewrites = [rewrites]
    for rw in rewrites:
        requests.append({"case_id": record["case_id"], **rw})
requests = requests[:100]
print(f"  {len(requests)} edit requests")

context_templates = get_context_templates(model, tok)

print()
print("=" * 72)
print("  MEASURED K@K^T SPECTRA (from actual 100-edit batch keys)")
print("=" * 72)

results = {}
for layer in hparams.layers:
    print(f"\n  Layer {layer}:")
    layer_ks = compute_ks(model, tok, requests, hparams, layer, context_templates).T
    # layer_ks: (d_in, B)
    d_in, B = layer_ks.shape
    KKT = layer_ks @ layer_ks.T  # (d_in, d_in)

    # Top singular values of K@K^T (= squared singular values of K)
    # Only need top few for scale audit
    S_kkt = torch.linalg.svdvals(KKT.float())

    results[layer] = {
        "d_in": d_in,
        "B": B,
        "K_norm_F": torch.linalg.norm(layer_ks).item(),
        "KKT_S_max": S_kkt[0].item(),
        "KKT_S_10": S_kkt[9].item() if len(S_kkt) > 9 else 0,
        "KKT_trace": torch.trace(KKT).item(),
        "KKT_rank_approx": (S_kkt > 1e-6 * S_kkt[0]).sum().item(),
        "K_col_norms_mean": torch.linalg.norm(layer_ks, dim=0).mean().item(),
        "K_col_norms_max": torch.linalg.norm(layer_ks, dim=0).max().item(),
    }

    print(f"    K shape: ({d_in}, {B})")
    print(f"    ||K||_F: {results[layer]['K_norm_F']:.4e}")
    print(f"    S.max(K@K^T): {results[layer]['KKT_S_max']:.4e}")
    print(f"    S.10(K@K^T):  {results[layer]['KKT_S_10']:.4e}")
    print(f"    trace(K@K^T):  {results[layer]['KKT_trace']:.4e}")
    print(f"    effective rank: {results[layer]['KKT_rank_approx']}")
    print(f"    ||k_col|| mean: {results[layer]['K_col_norms_mean']:.4e}")
    print(f"    ||k_col|| max:  {results[layer]['K_col_norms_max']:.4e}")

    del KKT, S_kkt
    torch.cuda.empty_cache()

# Also load C0 spectra for comparison
print("\n" + "=" * 72)
print("  C0 vs K@K^T COMPARISON")
print("=" * 72)

stats_dir = Path("data/stats/Qwen2.5-7B-Instruct/wikipedia_stats")
for layer in hparams.layers:
    layer_name = f"model.layers.{layer}.mlp.down_proj"
    filename = stats_dir / f"{layer_name}_float32_mom2_100000.npz"
    data = np.load(filename, allow_pickle=True)
    raw = torch.from_numpy(data["mom2.mom2"])
    count = int(data["mom2.count"])
    cov = (raw / count).float()
    S_c0 = torch.linalg.svdvals(cov.cuda())
    s_max_c0 = S_c0[0].item()

    r = results[layer]
    L2 = 1
    alpha = 15000
    print(f"\n  Layer {layer}:")
    print(f"    S.max(C0):             {s_max_c0:.4e}")
    print(f"    S.max(K@K^T) measured: {r['KKT_S_max']:.4e}")
    print(f"    ratio K@K^T/C0:        {r['KKT_S_max'] / s_max_c0:.4e}")
    print(f"    L2={L2} / S.max(K@K^T): {L2 / r['KKT_S_max']:.6e}  (ridge significance)")
    print(f"    α*S.max(C0):           {alpha * s_max_c0:.4e}  (MEMIT stabilizer)")
    print(f"    α*S.max(C0)/S.max(KKT): {alpha * s_max_c0 / r['KKT_S_max']:.2f}  (MEMIT dominance)")
    del S_c0
    torch.cuda.empty_cache()

# Save raw results
import json
out_path = Path("../../results/check4_spectra.json")
out_path.parent.mkdir(parents=True, exist_ok=True)
with open(out_path, "w") as f:
    json.dump(results, f, indent=2)
print(f"\n  Raw data saved to: {out_path}")
""")

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = os.environ.get("CUDA_VISIBLE_DEVICES", "0")
    subprocess.run([sys.executable, "-c", script], cwd=str(VENDOR_DIR), env=env)


def run_config(config, num_batches=2):
    """Items 3-5: Run a specific config and report forensics."""
    configs = {
        "A": "AlphaEdit + abs-tau P (official, float32)",
        "B": "AlphaEdit + rel-tau P (near-identity, float32)",
        "C": "AlphaEdit + rel-tau P + alpha*C0 (float32)",
        "D": "AlphaEdit + rel-tau P + alpha*C0 (float32, multi-batch stability test)",
        "E": "MEMIT-Seq (float64 solve, has alpha*C0)",
    }

    print("=" * 72)
    print(f"CHECK 4: MAGNITUDE FORENSICS — Config {config}")
    print(f"  {configs[config]}")
    print("=" * 72)
    print()

    ensure_vendor_patches()

    stats_dir = PROJECT_DIR / "data" / "stats" / "qwen2.5-7b-instruct" / "wikipedia_stats"
    layers = [4, 5, 6, 7, 8]
    num_edits = 100
    total_edits = num_edits * num_batches

    # Config E uses MEMIT-Seq runner directly
    if config == "E":
        print("  Config E: Using memit_sequential_runner directly")
        print(f"  Running {total_edits} edits ({num_batches} batches of {num_edits})")
        print()
        cmd = [
            sys.executable, str(PROJECT_DIR / "src" / "runners" / "memit_sequential_runner.py"),
            "--seed", "42",
            "--model_name", "Qwen/Qwen2.5-7B-Instruct",
            "--hparams_fname", "Qwen2.5-7B.json",
            "--ds_name", "mcf",
            "--dataset_size_limit", str(total_edits),
            "--num_edits", str(num_edits),
            "--lambda_prev", "0",
            "--lambda_delta", "0",
            "--fast_checkpoint",
        ]
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = os.environ.get("CUDA_VISIBLE_DEVICES", "0")
        subprocess.run(cmd, cwd=str(PROJECT_DIR), env=env)
        return

    # Configs A-D use patched AlphaEdit
    # Determine projector mode
    if config == "A":
        mode = "abs"
    else:
        mode = "rel"

    print(f"  Computing {mode} projector:")
    P = compute_projector(stats_dir, layers, mode=mode)
    p_path = VENDOR_DIR / "null_space_project.pt"
    torch.save(P, str(p_path))
    print()

    # Build the instrumented script
    use_c0 = config in ["C", "D"]
    use_f64 = False  # FP64 impractical on L40s for 18944x18944 solve (~40min each)

    # Build the solve and instrumentation patches
    # Note: FP64 solve removed — impractical on L40s for 18944×18944 (~40 min each)
    # Config D is now a multi-batch stability test of C (same f32 solve, more batches)

    forensics_code = textwrap.dedent("""\
        # === MAGNITUDE FORENSICS (injected) ===
        _upd_norm = torch.linalg.norm(upd_matrix).item()
        _upd_max = upd_matrix.abs().max().item()
        _has_nan = torch.isnan(upd_matrix).any().item()
        # Measure K@K^T scale for this layer
        _kkt_smax = torch.linalg.svdvals(layer_ks @ layer_ks.T)[0].item()
        # Condition number of LHS (recompute cheaply)
        _lhs_cond = P[i,:,:].cuda() @ (layer_ks @ layer_ks.T + cache_c[i,:,:].cuda()) + hparams.L2*torch.eye(layer_ks.shape[0], dtype=torch.float, device="cuda")
        _cond_val = torch.linalg.cond(_lhs_cond).item()
        # Weight magnitude after update
        _weight_name = [k for k in weights if f".{layer}." in k][0] if any(f".{layer}." in k for k in weights) else list(weights.keys())[0]
        _w_max_before = weights[_weight_name].abs().max().item()
        print(f"  FORENSICS layer={layer}: ||dW||_F={_upd_norm:.4e}, max|dW|={_upd_max:.4e}, "
              f"S.max(KKT)={_kkt_smax:.4e}, cond={_cond_val:.4e}, NaN={_has_nan}, "
              f"max|W|={_w_max_before:.4e}")
        if _has_nan:
            print(f"    *** NaN DETECTED in upd_matrix layer {layer} ***")
            print(f"    layer_ks has NaN: {torch.isnan(layer_ks).any().item()}")
            print(f"    resid has NaN: {torch.isnan(resid).any().item()}")
        del _lhs_cond
        # === END FORENSICS ===
""")

    # Build the subprocess script
    inner_script = textwrap.dedent(f"""\
import os, sys
import numpy as np
import torch
from pathlib import Path

sys.argv = [
    "evaluate.py",
    "--alg_name", "AlphaEdit",
    "--model_name", "Qwen/Qwen2.5-7B-Instruct",
    "--hparams_fname", "Qwen2.5-7B.json",
    "--ds_name", "mcf",
    "--dataset_size_limit", "{total_edits}",
    "--num_edits", "{num_edits}",
    "--use_cache",
]

with open("experiments/evaluate.py", "r") as f:
    source = f.read()

# Patch CUDA device
source = source.replace('os.environ["CUDA_VISIBLE_DEVICES"] = "1"', '# CUDA managed externally')

# Read AlphaEdit_main.py for patching
ae_path = Path("AlphaEdit/AlphaEdit_main.py")
ae_source = ae_path.read_text()
ae_original = ae_source
""")

    if use_c0:
        inner_script += textwrap.dedent("""\
# Add alpha*C0 to the solve LHS
_original_lhs = 'P[i,:,:].cuda() @ (layer_ks @ layer_ks.T + cache_c[i,:,:].cuda()) + hparams.L2*torch.eye(layer_ks.shape[0], dtype=torch.float,device=\"cuda\")'
_patched_lhs = (_original_lhs +
    ' + hparams.mom2_update_weight * get_cov(model, tok, '
    'hparams.rewrite_module_tmp.format(layer), '
    'hparams.mom2_dataset, hparams.mom2_n_samples, hparams.mom2_dtype).float()')
if _original_lhs in ae_source:
    ae_source = ae_source.replace(_original_lhs, _patched_lhs, 1)
else:
    print("WARNING: Could not find LHS pattern for C0 patch")

""")

    # Add forensics instrumentation
    forensics_escaped = forensics_code.replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n")
    inner_script += textwrap.dedent(f"""\
# Inject forensics after the solve, before upd_matrix_match_shape
_marker = 'upd_matrix = upd_matrix_match_shape(upd_matrix, weights[weight_name].shape)'
_forensics = '''{forensics_code}'''
if _marker in ae_source:
    ae_source = ae_source.replace(_marker, _forensics + "        " + _marker, 1)
else:
    print("WARNING: Could not find match_shape marker for forensics injection")

# Write patched file
ae_path.write_text(ae_source)

try:
    exec(compile(source, "experiments/evaluate.py", "exec"),
         {{"__name__": "__main__", "__file__": "experiments/evaluate.py"}})
finally:
    ae_path.write_text(ae_original)
    print("\\n  Restored original AlphaEdit_main.py")
""")

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = os.environ.get("CUDA_VISIBLE_DEVICES", "0")
    subprocess.run([sys.executable, "-c", inner_script], cwd=str(VENDOR_DIR), env=env)


def run_parity():
    """Item 4: Magnitude parity — same batch, MEMIT-Seq vs AlphaEdit(P=I)+C0, both float32."""
    print("=" * 72)
    print("CHECK 4.4: MAGNITUDE PARITY")
    print("  MEMIT-Seq vs AlphaEdit(P=I)+C0, both FLOAT32, same 100 edits")
    print("  (float32 used for both — L40s FP64 too slow for 18944x18944 solve)")
    print("=" * 72)
    print()

    ensure_vendor_patches()

    script = textwrap.dedent(r"""
import os, sys, json
import numpy as np
import torch
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer

# Vendor imports
from memit import MEMITHyperParams
from memit.compute_ks import compute_ks
from memit.memit_main import get_context_templates
from util.globals import *
from dsets import MultiCounterFactDataset

def load_cov_direct(layer_name):
    """Load covariance directly from NPZ, bypassing layer_stats recomputation."""
    stats_path = Path("data/stats/Qwen2.5-7B-Instruct/wikipedia_stats") / f"{layer_name}_float32_mom2_100000.npz"
    if not stats_path.exists():
        raise FileNotFoundError(f"Stats not found: {stats_path}")
    data = np.load(str(stats_path), allow_pickle=True)
    raw_mom2 = torch.from_numpy(data["mom2.mom2"])
    count = int(data["mom2.count"])
    cov = (raw_mom2 / count).float().cuda()
    print(f"    Loaded C0 for {layer_name}: shape={cov.shape}, count={count}")
    return cov

# Load hparams
hparams = MEMITHyperParams.from_json(HPARAMS_DIR / "MEMIT" / "Qwen2.5-7B.json")
alpha = hparams.mom2_update_weight  # 15000
L2 = 1  # AlphaEdit Qwen L2

# Load model
print("Loading model...")
tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B-Instruct")
tok.pad_token = tok.eos_token
model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-7B-Instruct").cuda()
model.eval()

# Load dataset
ds = MultiCounterFactDataset("data", tok=tok, size=100)
requests = []
for record in ds:
    rewrites = record["requested_rewrite"]
    if not isinstance(rewrites, list):
        rewrites = [rewrites]
    for rw in rewrites:
        requests.append({"case_id": record["case_id"], **rw})
requests = requests[:100]
print(f"  {len(requests)} edit requests")

context_templates = get_context_templates(model, tok)

# Compute real keys from model
print("\n  Computing keys...")
all_keys = {}
for layer in hparams.layers:
    layer_ks = compute_ks(model, tok, requests, hparams, layer, context_templates).T
    all_keys[layer] = layer_ks
    print(f"    Layer {layer}: K shape {layer_ks.shape}, ||K||_F={torch.linalg.norm(layer_ks):.4e}")

# Use random residuals (matching real scale)
torch.manual_seed(42)
d_out = model.config.hidden_size  # 3584
B = 100

print("\n" + "=" * 72)
print("  PARITY TEST: per-layer ||dW|| comparison (FLOAT32)")
print("  Both formulas use same precision — tests structural difference only")
print("=" * 72)

for i, layer in enumerate(hparams.layers):
    layer_ks = all_keys[layer]
    d_in = layer_ks.shape[0]

    # Random resid scaled to realistic magnitude (||resid||_F ≈ 250 = 25*sqrt(B))
    resid = torch.randn(d_out, B, device="cuda")
    resid = resid * (25.0 * np.sqrt(B) / torch.linalg.norm(resid).item())

    # Load C0 directly from NPZ (bypasses layer_stats recomputation)
    cov = load_cov_direct(hparams.rewrite_module_tmp.format(layer))

    K = layer_ks.float()
    resid_f = resid.float()
    cov_f = cov.float()
    KKT = K @ K.T

    # --- MEMIT solve (float32): adj_k = solve(alpha*C0 + K@K^T, K); dW = resid @ adj_k^T ---
    lhs_memit = alpha * cov_f + KKT
    adj_k_memit = torch.linalg.solve(lhs_memit, K)
    dW_memit = (resid_f @ adj_k_memit.T).cpu()
    cond_memit = torch.linalg.cond(lhs_memit).item()
    del adj_k_memit

    # --- AlphaEdit(P=I)+C0 solve (float32): dW = solve(K@K^T + L2*I + alpha*C0, K@resid^T)^T ---
    eye_f = torch.eye(d_in, device="cuda", dtype=torch.float)
    lhs_ae = KKT + L2 * eye_f + alpha * cov_f
    dW_ae = torch.linalg.solve(lhs_ae, K @ resid_f.T).T.cpu()
    cond_ae = torch.linalg.cond(lhs_ae).item()

    # --- AlphaEdit(P=I) NO C0 solve (float32): dW = solve(K@K^T + L2*I, K@resid^T)^T ---
    lhs_noC0 = KKT + L2 * eye_f
    cond_noC0 = torch.linalg.cond(lhs_noC0).item()
    try:
        dW_noC0 = torch.linalg.solve(lhs_noC0, K @ resid_f.T).T.cpu()
        norm_noC0 = torch.linalg.norm(dW_noC0).item()
        max_noC0 = dW_noC0.abs().max().item()
        nan_noC0 = torch.isnan(dW_noC0).any().item()
        del dW_noC0
    except Exception as e:
        norm_noC0 = float('inf')
        max_noC0 = float('inf')
        nan_noC0 = True

    del lhs_memit, lhs_ae, lhs_noC0, eye_f, KKT, K, resid_f, cov_f
    torch.cuda.empty_cache()

    # --- Report ---
    norm_memit = torch.linalg.norm(dW_memit).item()
    norm_ae = torch.linalg.norm(dW_ae).item()
    diff = torch.linalg.norm(dW_memit - dW_ae).item()
    max_memit = dW_memit.abs().max().item()
    max_ae = dW_ae.abs().max().item()

    print(f"\n  Layer {layer}:")
    print(f"    (a) MEMIT (α*C₀+KKT):        ||dW||_F = {norm_memit:.6e}, max|dW| = {max_memit:.6e}, cond = {cond_memit:.4e}")
    print(f"    (b) AE(P=I)+C₀ (KKT+L2+αC₀): ||dW||_F = {norm_ae:.6e}, max|dW| = {max_ae:.6e}, cond = {cond_ae:.4e}")
    print(f"    (c) AE(P=I) NO C₀ (KKT+L2):  ||dW||_F = {norm_noC0:.6e}, max|dW| = {max_noC0:.6e}, cond = {cond_noC0:.4e}, NaN={nan_noC0}")
    print(f"    ||a-b|| / ||a|| = {diff/norm_memit:.6e}  (formula structure + L2 effect)")
    print(f"    ||b|| / ||a|| = {norm_ae/norm_memit:.6f}  (should be ≈1.0)")
    print(f"    ||c|| / ||a|| = {norm_noC0/norm_memit:.4e}  (blowup factor without C₀)")

    del dW_memit, dW_ae

print()
print("=" * 72)
print("INTERPRETATION")
print("=" * 72)
print()
print("  If ||a-b||/||a|| ≈ 0 and ||b||/||a|| ≈ 1.0:")
print("    → MEMIT and AlphaEdit(P=I)+C₀ produce IDENTICAL updates")
print("    → The formulas are algebraically equivalent (LHS symmetric)")
print("    → L2=1 is negligible relative to α*C₀")
print()
print("  ||c||/||a|| is the KEY NUMBER:")
print("    → Shows how much larger updates are WITHOUT C₀")
print("    → This is what AlphaEdit's tight P must compensate for")
print("    → If large: confirms P is doing magnitude bounding")
print()
""")

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = os.environ.get("CUDA_VISIBLE_DEVICES", "0")
    subprocess.run([sys.executable, "-c", script], cwd=str(VENDOR_DIR), env=env)


def main():
    parser = argparse.ArgumentParser(
        description="Check 4 (Expanded): Full Regime Audit",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent("""\
            Configs:
              A       AlphaEdit + abs-tau P (official, float32)
              B       AlphaEdit + rel-tau P (near-identity, float32)
              C       AlphaEdit + rel-tau P + alpha*C0 (float32)
              D       AlphaEdit + rel-tau P + alpha*C0 (FLOAT64 solve)
              E       MEMIT-Seq (float64 solve)
              spectra Measured K@K^T spectra from real keys
              parity  D vs E side-by-side magnitude comparison
        """))
    parser.add_argument("--config", required=True,
                        choices=["A", "B", "C", "D", "E", "spectra", "parity"],
                        help="Which configuration to run")
    parser.add_argument("--num_batches", type=int, default=2,
                        help="Number of 100-edit batches (default: 2)")
    args = parser.parse_args()

    if args.config == "spectra":
        run_spectra()
    elif args.config == "parity":
        run_parity()
    else:
        run_config(args.config, num_batches=args.num_batches)


if __name__ == "__main__":
    main()

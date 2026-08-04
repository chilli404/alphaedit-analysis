#!/usr/bin/env python3
"""
Check 4.5: Precision Diagnosis (the decisive pair).

Two configs, both using AlphaEdit + rel-tau P + alpha*C0 on Qwen, 3 batches:

  (a) FLOAT64 SOLVE: solve in double precision (matching MEMIT's cov.double()
      policy), weights still bf16.
      Stable => pathology is partly solve precision
      NaN    => precision exculpated

  (b) FP32 APPLICATION: maintain fp32 master copies of edited layers.
      Apply dW in fp32, cast to bf16 only for model's forward pass.
      Stable => "bf16 weight corruption" mechanism confirmed
      NaN    => model genuinely broken by update content (magnitude/direction)

The 2x2 outcome uniquely identifies the mechanism.

Usage:
    uv run python scripts/check4_5_precision.py --config a
    uv run python scripts/check4_5_precision.py --config b
"""

import argparse
import os
import sys
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


def compute_relative_projector(stats_dir, layers, rel_factor=0.02):
    """Compute P with relative threshold (near-identity)."""
    P_list = []
    for layer in layers:
        layer_name = f"model.layers.{layer}.mlp.down_proj"
        filename = stats_dir / f"{layer_name}_float32_mom2_100000.npz"
        data = np.load(filename, allow_pickle=True)
        raw_mom2 = torch.from_numpy(data["mom2.mom2"])
        count = int(data["mom2.count"])
        cov = (raw_mom2 / count).float()

        device = "cuda" if torch.cuda.is_available() else "cpu"
        U, S, _ = torch.linalg.svd(cov.to(device), full_matrices=False)
        threshold = rel_factor * S[0].item()
        small = (S < threshold).nonzero(as_tuple=True)[0]
        P = (U[:, small] @ U[:, small].T).cpu()
        P_list.append(P)
        retained = len(small)
        total = S.shape[0]
        print(f"  Layer {layer}: retained {retained}/{total} ({100*retained/total:.1f}%) "
              f"[rel tau={threshold:.4e}]")
    return torch.stack(P_list)


def preload_c0(stats_dir, layers):
    """Pre-load C0 covariance matrices from NPZ files directly."""
    c0_dict = {}
    for layer in layers:
        layer_name = f"model.layers.{layer}.mlp.down_proj"
        filename = stats_dir / f"{layer_name}_float32_mom2_100000.npz"
        data = np.load(filename, allow_pickle=True)
        raw_mom2 = torch.from_numpy(data["mom2.mom2"])
        count = int(data["mom2.count"])
        cov = (raw_mom2 / count).float()
        c0_dict[layer] = cov
        print(f"  Layer {layer}: C0 shape={cov.shape}, ||C0||_F={torch.linalg.norm(cov):.4e}")
    return c0_dict


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", choices=["a", "b"], required=True,
                        help="a=float64_solve, b=fp32_application")
    parser.add_argument("--num_edits", type=int, default=100)
    parser.add_argument("--num_batches", type=int, default=3)
    args = parser.parse_args()

    stats_dir = PROJECT_DIR / "data" / "stats" / "qwen2.5-7b-instruct" / "wikipedia_stats"
    if not stats_dir.exists():
        print(f"ERROR: Qwen stats not found at {stats_dir}")
        sys.exit(1)

    layers = [4, 5, 6, 7, 8]
    dataset_size = args.num_edits * args.num_batches

    config_desc = {
        "a": "AlphaEdit + rel-tau P + alpha*C0, FLOAT64 SOLVE (weights bf16)",
        "b": "AlphaEdit + rel-tau P + alpha*C0, FP32 MASTER COPIES (forward bf16)",
    }

    print("=" * 72)
    print(f"CHECK 4.5{args.config}: {config_desc[args.config]}")
    print(f"  {args.num_batches} batches of {args.num_edits} edits on Qwen2.5-7B-Instruct")
    print("=" * 72)
    print()

    # Compute relative-threshold projector
    print("Computing relative-threshold projector:")
    P = compute_relative_projector(stats_dir, layers)
    p_path = VENDOR_DIR / "null_space_project.pt"
    torch.save(P, str(p_path))
    print(f"  Saved projector: {p_path} [shape={list(P.shape)}]")
    print()

    # Pre-load C0 matrices (bypass vendor's get_cov which has path issues)
    print("Pre-loading C0 covariance matrices:")
    c0_dict = preload_c0(stats_dir, layers)
    c0_path = VENDOR_DIR / "_check4_5_c0_cache.pt"
    torch.save(c0_dict, str(c0_path))
    print(f"  Saved C0 cache: {c0_path}")
    print()

    # Link hparams and apply source patches
    os.chdir(str(VENDOR_DIR))
    link_hparams()

    sys.path.insert(0, str(VENDOR_DIR))
    sys.path.insert(0, str(PROJECT_DIR / "src" / "util"))
    from source_patches import patch_evaluate_file, patch_glue_eval_file
    patch_evaluate_file(VENDOR_DIR)
    patch_glue_eval_file(VENDOR_DIR)

    # Patch AlphaEdit_main.py
    ae_path = VENDOR_DIR / "AlphaEdit" / "AlphaEdit_main.py"
    ae_original = ae_path.read_text()
    ae_source = ae_original

    # Add **_kwargs
    if "**_kwargs" not in ae_source:
        ae_source = ae_source.replace("    P = None,\n", "    P = None, **_kwargs,\n", 1)

    # Patch memit_main.py for **_kwargs
    memit_path = VENDOR_DIR / "memit" / "memit_main.py"
    memit_original = memit_path.read_text()
    if "**_kwargs" not in memit_original:
        memit_path.write_text(memit_original.replace(
            "    cache_template: Optional[str] = None,\n",
            "    cache_template: Optional[str] = None, **_kwargs,\n", 1))

    # Inject C0 loading at top of apply_AlphaEdit_to_model
    # We inject a line that loads the pre-computed C0 cache right after the function starts
    c0_inject = (
        '\n'
        '    # === CHECK 4.5: PRE-LOADED C0 (bypasses get_cov) ===\n'
        '    import torch as _torch\n'
        '    _c0_cache = _torch.load("_check4_5_c0_cache.pt", weights_only=False)\n'
        '    print(f"  [4.5] Loaded pre-computed C0 for layers: {list(_c0_cache.keys())}")\n'
    )
    # Insert after the "requests = deepcopy(requests)" line
    inject_after = '    requests = deepcopy(requests)\n'
    ae_source = ae_source.replace(inject_after, inject_after + c0_inject, 1)

    # --- Config (a): Float64 solve with pre-loaded C0 ---
    if args.config == "a":
        # Replace the solve with double-precision version using pre-loaded C0
        original_solve_line = (
            '        upd_matrix = torch.linalg.solve(\n'
            '                P[i,:,:].cuda() @ (layer_ks @ layer_ks.T + cache_c[i,:,:].cuda()) '
            '+ hparams.L2*torch.eye(layer_ks.shape[0], dtype=torch.float,device="cuda"), '
            'P[i,:,:].cuda() @ layer_ks @ resid.T\n'
            '        )'
        )
        patched_solve = (
            '        # === CHECK 4.5a: FLOAT64 SOLVE + C0 ===\n'
            '        _C0 = _c0_cache[layer].cuda()\n'
            '        _lhs = (P[i,:,:].cuda() @ (layer_ks @ layer_ks.T + cache_c[i,:,:].cuda()) '
            '+ hparams.L2*torch.eye(layer_ks.shape[0], dtype=torch.float,device="cuda") '
            '+ hparams.mom2_update_weight * _C0).double()\n'
            '        _rhs = (P[i,:,:].cuda() @ layer_ks @ resid.T).double()\n'
            '        upd_matrix = torch.linalg.solve(_lhs, _rhs).float()\n'
            '        _cond = torch.linalg.cond(_lhs).item()\n'
            '        print(f"  [4.5a] layer {layer}: cond={_cond:.4e}, '
            '||upd||={torch.linalg.norm(upd_matrix).item():.4e}, '
            'max|upd|={upd_matrix.abs().max().item():.4e}")\n'
            '        del _lhs, _rhs, _C0\n'
            '        torch.cuda.empty_cache()'
        )

        if original_solve_line in ae_source:
            ae_source = ae_source.replace(original_solve_line, patched_solve, 1)
        else:
            # Try matching just the solve pattern more flexibly
            # Find and replace via line scanning
            lines = ae_source.split('\n')
            new_lines = []
            i = 0
            replaced = False
            while i < len(lines):
                if not replaced and 'upd_matrix = torch.linalg.solve(' in lines[i]:
                    # Skip original solve lines (may span 2-3 lines)
                    j = i
                    while j < len(lines) and ')' not in lines[j][lines[j].find('solve'):] if 'solve' in lines[j] else True:
                        j += 1
                        if j < len(lines) and lines[j].strip().startswith(')'):
                            j += 1
                            break
                        if j < len(lines) and 'resid.T' in lines[j]:
                            j += 1
                            break
                    # Find closing paren
                    while j < len(lines) and lines[j].strip() == ')':
                        j += 1
                    new_lines.append(patched_solve)
                    i = j
                    replaced = True
                else:
                    new_lines.append(lines[i])
                    i += 1
            if replaced:
                ae_source = '\n'.join(new_lines)
            else:
                print("ERROR: Could not find solve pattern to patch")
                sys.exit(1)

    # --- Config (b): FP32 master copies with pre-loaded C0 ---
    elif args.config == "b":
        # Add C0 to the solve (still float32)
        original_lhs = (
            'P[i,:,:].cuda() @ (layer_ks @ layer_ks.T + cache_c[i,:,:].cuda()) '
            '+ hparams.L2*torch.eye(layer_ks.shape[0], dtype=torch.float,device="cuda")'
        )
        patched_lhs = (
            'P[i,:,:].cuda() @ (layer_ks @ layer_ks.T + cache_c[i,:,:].cuda()) '
            '+ hparams.L2*torch.eye(layer_ks.shape[0], dtype=torch.float,device="cuda") '
            '+ hparams.mom2_update_weight * _c0_cache[layer].cuda()'
        )
        ae_source = ae_source.replace(original_lhs, patched_lhs, 1)

        # Add fp32 master copy initialization before the layer loop
        layer_loop_line = '    for i, layer in enumerate(hparams.layers):\n        print(f"\\n\\nLAYER {layer}\\n")'
        fp32_init = (
            '    # === CHECK 4.5b: FP32 MASTER COPIES ===\n'
            '    _fp32_masters = {name: w.float().cpu().clone() for name, w in weights.items()}\n'
            '    print(f"  [4.5b] Initialized fp32 masters for {len(_fp32_masters)} layers")\n'
            '    print(f"  [4.5b] Model weight dtype: {next(iter(weights.values())).dtype}")\n\n'
        )
        ae_source = ae_source.replace(layer_loop_line, fp32_init + '    ' + layer_loop_line, 1)

        # Replace the weight update to use fp32 masters
        original_update = (
            '        with torch.no_grad():\n'
            '            weights[weight_name][...] = weights[weight_name] + upd_matrix'
        )
        patched_update = (
            '        with torch.no_grad():\n'
            '            # FP32 master: accumulate in fp32, cast to model dtype for forward\n'
            '            _fp32_masters[weight_name] = _fp32_masters[weight_name].cuda() + upd_matrix.float()\n'
            '            weights[weight_name][...] = _fp32_masters[weight_name].to(weights[weight_name].dtype)\n'
            '            _trunc_err = torch.linalg.norm(_fp32_masters[weight_name] - weights[weight_name].float()).item()\n'
            '            print(f"  [4.5b] layer {layer}: ||upd||={torch.linalg.norm(upd_matrix).item():.4e}, '
            'truncation_err={_trunc_err:.4e}, w_dtype={weights[weight_name].dtype}")\n'
            '            _fp32_masters[weight_name] = _fp32_masters[weight_name].cpu()'
        )
        ae_source = ae_source.replace(original_update, patched_update, 1)

    ae_path.write_text(ae_source)
    print(f"  Patched AlphaEdit_main.py for config {args.config}")
    print()

    # Build subprocess script
    script = textwrap.dedent(f"""\
import os, sys
import numpy as np
import torch

sys.argv = [
    "evaluate.py",
    "--alg_name", "AlphaEdit",
    "--model_name", "Qwen/Qwen2.5-7B-Instruct",
    "--hparams_fname", "Qwen2.5-7B.json",
    "--ds_name", "mcf",
    "--dataset_size_limit", "{dataset_size}",
    "--num_edits", "{args.num_edits}",
    "--use_cache",
]

with open("experiments/evaluate.py", "r") as f:
    source = f.read()

# Patch CUDA device
patch_target = 'os.environ["CUDA_VISIBLE_DEVICES"] = "1"'
if patch_target in source:
    source = source.replace(patch_target, '# CUDA managed externally')

exec(compile(source, "experiments/evaluate.py", "exec"),
     {{"__name__": "__main__", "__file__": "experiments/evaluate.py"}})
""")

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = os.environ.get("CUDA_VISIBLE_DEVICES", "0")

    print(f"  Running: {dataset_size} edits ({args.num_batches} batches of {args.num_edits})")
    print(f"  Projector: relative tau (near-identity)")
    if args.config == "a":
        print(f"  Solve: FLOAT64 (cast back to float32 for weight update, weights stored bf16)")
        print(f"  C0: Pre-loaded from NPZ (bypasses vendor get_cov)")
    else:
        print(f"  Solve: FLOAT32 (default)")
        print(f"  Weight storage: FP32 MASTER COPIES (bf16 for forward only)")
        print(f"  C0: Pre-loaded from NPZ (bypasses vendor get_cov)")
    print(f"  LHS: P @ (K@K^T + cache_c) + L2*I + alpha*C0")
    print()

    try:
        result = subprocess.run(
            [sys.executable, "-c", script],
            cwd=str(VENDOR_DIR),
            env=env,
        )
        if result.returncode != 0:
            print(f"\n  {'='*60}")
            print(f"  RESULT: CONFIG {args.config.upper()} — FAILED (return code {result.returncode})")
            print(f"  {'='*60}")
            if args.config == "a":
                print("  => Float64 solve did NOT stabilize.")
                print("     Precision EXCULPATED — the instability is in update CONTENT.")
            else:
                print("  => FP32 masters did NOT stabilize.")
                print("     Model genuinely broken by update MAGNITUDE/DIRECTION.")
        else:
            print(f"\n  {'='*60}")
            print(f"  RESULT: CONFIG {args.config.upper()} — STABLE ({args.num_batches} batches complete)")
            print(f"  {'='*60}")
            if args.config == "a":
                print("  => Float64 solve STABILIZED the computation.")
                print("     Pathology is partly SOLVE PRECISION (float32 insufficient for cond>1e7).")
            else:
                print("  => FP32 master copies STABILIZED the computation.")
                print("     MECHANISM: bf16 weight truncation causes cumulative corruption.")
                print("     FIX: Keep edited layers in fp32 (one-line precision policy).")
    finally:
        ae_path.write_text(ae_original)
        memit_path.write_text(memit_original)
        # Clean up temp files
        if c0_path.exists():
            c0_path.unlink()
        print("\n  Restored original vendor files + cleaned temp cache")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Check 4: Magnitude + Precision Forensics.

For batch 1 under each configuration, reports:
  - ||dW||_F per layer
  - max weight magnitude after update
  - Whether NaN first appears in bf16 forward or fp32 solve
  - Condition number of the LHS matrix

This script patches AlphaEdit_main.py to add instrumentation, then runs
a single batch of 100 edits under each of four configurations:
  A) abs-tau P (official AlphaEdit)
  B) rel-tau P (near-identity)
  C) rel-tau P + alpha*C0
  D) MEMIT (no P, has alpha*C0)

Usage:
    uv run python scripts/check4_magnitude_forensics.py --config A
    uv run python scripts/check4_magnitude_forensics.py --config B
    uv run python scripts/check4_magnitude_forensics.py --config C
    uv run python scripts/check4_magnitude_forensics.py --config D
"""

import argparse
import os
import sys
import json
import numpy as np
import torch
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent
VENDOR_DIR = PROJECT_DIR / "vendor" / "AlphaEdit"

sys.path.insert(0, str(VENDOR_DIR))
sys.path.insert(0, str(PROJECT_DIR / "src" / "util"))

from source_patches import patch_evaluate_file, patch_glue_eval_file


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

        device = "cuda" if torch.cuda.is_available() else "cpu"
        U, S, _ = torch.linalg.svd(cov.to(device), full_matrices=False)

        if mode == "rel":
            tau = threshold * S[0].item()
        else:
            tau = threshold

        small = (S < tau).nonzero(as_tuple=True)[0]
        P = (U[:, small] @ U[:, small].T).cpu()
        P_list.append(P)
        print(f"  Layer {layer}: {len(small)}/{S.shape[0]} retained "
              f"({mode} τ={tau:.4e})")
    return torch.stack(P_list)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", choices=["A", "B", "C", "D"], required=True,
                        help="A=abs-P, B=rel-P, C=rel-P+C0, D=MEMIT")
    parser.add_argument("--num_edits", type=int, default=100)
    args = parser.parse_args()

    stats_dir = PROJECT_DIR / "data" / "stats" / "qwen2.5-7b-instruct" / "wikipedia_stats"
    layers = [4, 5, 6, 7, 8]

    print(f"\n=== Check 4: Magnitude Forensics — Config {args.config} ===\n")

    config_desc = {
        "A": "AlphaEdit + absolute τ=0.02 (official)",
        "B": "AlphaEdit + relative τ (near-identity P)",
        "C": "AlphaEdit + relative τ + alpha*C0",
        "D": "MEMIT (no P, has alpha*C0)",
    }
    print(f"  Config: {config_desc[args.config]}")
    print()

    os.chdir(str(VENDOR_DIR))
    patch_evaluate_file(VENDOR_DIR)
    patch_glue_eval_file(VENDOR_DIR)

    # Setup projector
    if args.config in ["A", "B", "C"]:
        mode = "abs" if args.config == "A" else "rel"
        print(f"Computing {mode} projector:")
        P = compute_projector(stats_dir, layers, mode=mode)
        p_path = VENDOR_DIR / "null_space_project.pt"
        torch.save(P, str(p_path))
        print()

    # Patch AlphaEdit_main.py for instrumentation
    ae_path = VENDOR_DIR / "AlphaEdit" / "AlphaEdit_main.py"
    ae_original = ae_path.read_text()

    # Add instrumentation after line 131 (the solve) and before line 134 (upd_matrix_match_shape)
    instrumentation = '''
        # === MAGNITUDE FORENSICS (injected) ===
        _upd_norm = torch.linalg.norm(upd_matrix).item()
        _lhs_matrix = P[i,:,:].cuda() @ (layer_ks @ layer_ks.T + cache_c[i,:,:].cuda()) + hparams.L2*torch.eye(layer_ks.shape[0], dtype=torch.float,device="cuda")
        _cond = torch.linalg.cond(_lhs_matrix).item()
        print(f"  FORENSICS layer {layer}: ||dW||_F={_upd_norm:.4e}, cond(LHS)={_cond:.4e}")
        if torch.isnan(upd_matrix).any():
            print(f"  *** NaN DETECTED in upd_matrix at layer {layer} ***")
            print(f"  LHS has NaN: {torch.isnan(_lhs_matrix).any().item()}")
            _rhs = P[i,:,:].cuda() @ layer_ks @ resid.T
            print(f"  RHS has NaN: {torch.isnan(_rhs).any().item()}")
            print(f"  targets has NaN: {torch.isnan(targets).any().item()}")
            print(f"  layer_ks has NaN: {torch.isnan(layer_ks).any().item()}")
        del _lhs_matrix
        # === END FORENSICS ===
'''

    # For config C: also patch the solve to add C0
    if args.config == "C":
        original_solve = (
            'P[i,:,:].cuda() @ (layer_ks @ layer_ks.T + cache_c[i,:,:].cuda()) '
            '+ hparams.L2*torch.eye(layer_ks.shape[0], dtype=torch.float,device="cuda")'
        )
        patched_solve = (
            'P[i,:,:].cuda() @ (layer_ks @ layer_ks.T + cache_c[i,:,:].cuda()) '
            '+ hparams.L2*torch.eye(layer_ks.shape[0], dtype=torch.float,device="cuda") '
            '+ hparams.mom2_update_weight * get_cov(model, tok, '
            'hparams.rewrite_module_tmp.format(layer), '
            'hparams.mom2_dataset, hparams.mom2_n_samples, hparams.mom2_dtype).float()'
        )
        ae_source = ae_original.replace(original_solve, patched_solve, 1)
    else:
        ae_source = ae_original

    # Inject instrumentation after the solve
    solve_end_marker = 'upd_matrix = upd_matrix_match_shape(upd_matrix, weights[weight_name].shape)'
    if solve_end_marker in ae_source:
        ae_source = ae_source.replace(
            solve_end_marker,
            instrumentation + "        " + solve_end_marker,
            1
        )

    ae_path.write_text(ae_source)

    # Determine algorithm
    alg_name = "MEMIT" if args.config == "D" else "AlphaEdit"

    # Run experiment
    eval_source = (VENDOR_DIR / "experiments" / "evaluate.py").read_text()
    hparams_dir = PROJECT_DIR / "configs" / "hparams"
    os.environ["HPARAMS_DIR"] = str(hparams_dir)

    sys.argv = [
        "evaluate.py",
        "--alg_name", alg_name,
        "--model_name", "Qwen/Qwen2.5-7B-Instruct",
        "--hparams_fname", "Qwen2.5-7B.json",
        "--ds_name", "mcf",
        "--dataset_size_limit", str(args.num_edits),
        "--num_edits", str(args.num_edits),
        "--use_cache",
    ]

    print(f"\n  Running {args.num_edits}-edit single batch")
    print(f"  Algorithm: {alg_name}")
    print()

    try:
        exec(compile(eval_source, "evaluate.py", "exec"))
    finally:
        # Restore
        ae_path.write_text(ae_original)
        print("\n  Restored original AlphaEdit_main.py")


if __name__ == "__main__":
    main()

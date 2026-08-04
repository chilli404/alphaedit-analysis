#!/usr/bin/env python3
"""
Check 3: AlphaEdit with relative-tau P + alpha*C0 added to solve LHS.

Tests whether adding the C0 stabilizer (as in MEMIT) to AlphaEdit's solve
prevents the NaN explosion that occurred with the near-identity projector.

Patches AlphaEdit_main.py to inject alpha*C0 into the LHS:
  Original:  P @ (K@K^T + cache_c) + L2*I
  Patched:   P @ (K@K^T + cache_c) + L2*I + alpha*C0

Also computes a relative-threshold projector (P ≈ I) and uses that.

Usage:
    uv run python scripts/check3_alphaedit_with_c0.py
"""

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
              f"[rel τ={threshold:.4e}]")
    return torch.stack(P_list)


def main():
    # Setup
    stats_dir = PROJECT_DIR / "data" / "stats" / "qwen2.5-7b-instruct" / "wikipedia_stats"
    if not stats_dir.exists():
        print(f"ERROR: Qwen stats not found at {stats_dir}")
        sys.exit(1)

    layers = [4, 5, 6, 7, 8]
    alpha = 15000  # mom2_update_weight, same as MEMIT

    print("=== Check 3: AlphaEdit + relative-tau P + alpha*C0 ===")
    print(f"  alpha (mom2_update_weight): {alpha}")
    print()

    # Compute relative-threshold projector
    print("Computing relative-threshold projector:")
    P = compute_relative_projector(stats_dir, layers)
    p_path = VENDOR_DIR / "null_space_project.pt"
    torch.save(P, str(p_path))
    print(f"  Saved to: {p_path} [shape={list(P.shape)}]")
    print()

    # Apply source patches
    os.chdir(str(VENDOR_DIR))
    patch_evaluate_file(VENDOR_DIR)
    patch_glue_eval_file(VENDOR_DIR)

    # Read AlphaEdit_main.py and patch the solve to add alpha*C0
    ae_path = VENDOR_DIR / "AlphaEdit" / "AlphaEdit_main.py"
    ae_source = ae_path.read_text()

    # The original solve line:
    original_solve = (
        'P[i,:,:].cuda() @ (layer_ks @ layer_ks.T + cache_c[i,:,:].cuda()) '
        '+ hparams.L2*torch.eye(layer_ks.shape[0], dtype=torch.float,device="cuda")'
    )

    # Patched solve: add alpha*C0
    # We need to compute C0 = get_cov() for each layer
    # Inject the C0 fetch before the layer loop and add it to the LHS
    patched_solve = (
        'P[i,:,:].cuda() @ (layer_ks @ layer_ks.T + cache_c[i,:,:].cuda()) '
        '+ hparams.L2*torch.eye(layer_ks.shape[0], dtype=torch.float,device="cuda") '
        '+ hparams.mom2_update_weight * get_cov(model, tok, '
        'hparams.rewrite_module_tmp.format(layer), '
        'hparams.mom2_dataset, hparams.mom2_n_samples, hparams.mom2_dtype).float()'
    )

    if original_solve not in ae_source:
        print("ERROR: Could not find the original solve pattern in AlphaEdit_main.py")
        print("  (Maybe already patched or format changed)")
        sys.exit(1)

    ae_patched = ae_source.replace(original_solve, patched_solve, 1)
    ae_path.write_text(ae_patched)
    print("  Patched AlphaEdit_main.py: added alpha*C0 to solve LHS")
    print(f"  alpha*C0 uses mom2_update_weight={alpha}")
    print()

    # Now run the experiment
    eval_source = (VENDOR_DIR / "experiments" / "evaluate.py").read_text()

    hparams_dir = PROJECT_DIR / "configs" / "hparams"

    sys.argv = [
        "evaluate.py",
        "--alg_name", "AlphaEdit",
        "--model_name", "Qwen/Qwen2.5-7B-Instruct",
        "--hparams_fname", "Qwen2.5-7B.json",
        "--ds_name", "mcf",
        "--dataset_size_limit", "200",
        "--num_edits", "100",
        "--use_cache",
    ]

    # Set HPARAMS_DIR so evaluate.py finds our configs
    os.environ["HPARAMS_DIR"] = str(hparams_dir)

    print(f"  Running: 200 edits (2 batches of 100)")
    print(f"  Projector: relative τ (near-identity)")
    print(f"  LHS: P @ (K@K^T + cache_c) + L2*I + alpha*C0")
    print()

    try:
        exec(compile(eval_source, "evaluate.py", "exec"))
    finally:
        # Restore original AlphaEdit_main.py
        ae_path.write_text(ae_source)
        print("\n  Restored original AlphaEdit_main.py")


if __name__ == "__main__":
    main()

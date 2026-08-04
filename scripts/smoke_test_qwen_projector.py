#!/usr/bin/env python3
"""
Check 7: Functional smoke test — compare AlphaEdit editing with
absolute-threshold projector vs relative-threshold (near-identity) projector.

Usage:
    uv run python scripts/smoke_test_qwen_projector.py --threshold abs
    uv run python scripts/smoke_test_qwen_projector.py --threshold rel

Reports batch efficacy under each projector mode.
"""

import argparse
import json
import sys
import os
from pathlib import Path

import numpy as np
import torch

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent
VENDOR_DIR = PROJECT_DIR / "vendor" / "AlphaEdit"

sys.path.insert(0, str(VENDOR_DIR))
sys.path.insert(0, str(PROJECT_DIR / "src" / "util"))


def compute_projector_absolute(stats_dir, layers, threshold=0.02):
    """Compute projector with absolute threshold (official AlphaEdit)."""
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
        small = (S < threshold).nonzero(as_tuple=True)[0]
        P = (U[:, small] @ U[:, small].T).cpu()
        P_list.append(P)
        print(f"  Layer {layer}: retained {len(small)}/{S.shape[0]} "
              f"({100*len(small)/S.shape[0]:.1f}%) [abs τ={threshold}]")
    return torch.stack(P_list)


def compute_projector_relative(stats_dir, layers, rel_factor=0.02):
    """Compute projector with relative threshold (τ_rel = rel_factor * S.max())."""
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
        print(f"  Layer {layer}: retained {len(small)}/{S.shape[0]} "
              f"({100*len(small)/S.shape[0]:.1f}%) [rel τ={threshold:.4e}]")
    return torch.stack(P_list)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--threshold", choices=["abs", "rel"], required=True,
                        help="'abs' for official τ=0.02, 'rel' for τ=0.02*S.max()")
    parser.add_argument("--num_edits", type=int, default=200,
                        help="Number of edits for smoke test (default: 200)")
    parser.add_argument("--batch_size", type=int, default=100,
                        help="Edit batch size (default: 100)")
    args = parser.parse_args()

    # Find stats
    stats_dir = PROJECT_DIR / "data" / "stats" / "qwen2.5-7b-instruct" / "wikipedia_stats"
    if not stats_dir.exists():
        print(f"ERROR: Qwen stats not found at {stats_dir}")
        sys.exit(1)

    layers = [4, 5, 6, 7, 8]

    print(f"\n=== Computing {args.threshold.upper()} projector ===")
    if args.threshold == "abs":
        P = compute_projector_absolute(stats_dir, layers, threshold=0.02)
    else:
        P = compute_projector_relative(stats_dir, layers, rel_factor=0.02)

    # Save projector for the experiment run
    p_path = VENDOR_DIR / "null_space_project.pt"
    torch.save(P, str(p_path))
    print(f"\nSaved projector to: {p_path} [shape={list(P.shape)}]")

    # Now run a minimal AlphaEdit experiment via seeded_runner (subprocess)
    print(f"\n=== Running {args.num_edits}-edit smoke test ===")
    print(f"  Threshold mode: {args.threshold}")
    print(f"  Batch size: {args.batch_size}")
    print()

    import subprocess

    # seeded_runner.py handles all source patching and proper exec context
    cmd = [
        sys.executable, str(PROJECT_DIR / "src" / "runners" / "seeded_runner.py"),
        "--seed", "42",
        "--alg_name", "AlphaEdit",
        "--model_name", "Qwen/Qwen2.5-7B-Instruct",
        "--hparams_fname", "Qwen2.5-7B.json",
        "--ds_name", "mcf",
        "--dataset_size_limit", str(args.num_edits),
        "--num_edits", str(args.batch_size),
        "--use_cache",
        "--skip_generation_tests",
    ]

    env = os.environ.copy()
    env["HPARAMS_DIR"] = str(PROJECT_DIR / "configs" / "hparams")
    # Ensure the projector we just saved is found
    env["PYTHONPATH"] = str(VENDOR_DIR) + ":" + env.get("PYTHONPATH", "")

    print(f"  Command: {' '.join(cmd[-10:])}")
    print(f"  Projector at: {p_path}")
    print()

    result = subprocess.run(cmd, env=env, cwd=str(VENDOR_DIR))
    if result.returncode != 0:
        print(f"\n  ERROR: Experiment exited with code {result.returncode}")
        sys.exit(1)


if __name__ == "__main__":
    main()

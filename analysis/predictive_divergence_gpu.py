#!/usr/bin/env python3
"""
GPU-accelerated extended cache metrics for predictive divergence analysis.

Loads raw cache_c.pt tensors from checkpoints and computes metrics NOT
available in the standard mechanism_analyzer JSONL:
  - rayleigh_quotient_avg: Tr(C²)/Tr(C) — average key-space concentration
  - key_crowding_proxy: Tr(C²)/Tr(C)² — related to mean pairwise alignment
  - linear_effective_rank: Tr(C)/||C||₂ (spectral norm based)
  - top_eigenvalue_concentration: λ₁/Σλᵢ

These are hypothesized leading indicators of seed-level collapse divergence.

Requires access to checkpoint directories containing cache_c.pt files.

Usage:
    # From cluster with S3 access
    uv run python -m analysis.predictive_divergence_gpu \\
        --seed 42 \\
        --checkpoint_base /s3-data/continual-learning/alphaedit/checkpoints/AlphaEdit/seed42

    # Local with pulled checkpoints
    uv run python -m analysis.predictive_divergence_gpu \\
        --seed 42 \\
        --checkpoint_base ~/.cache/alphaedit_checkpoints/AlphaEdit/seed42
"""

import argparse
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = PROJECT_ROOT / "results" / "mechanism_analysis"


def compute_extended_cache_metrics(cache_c: torch.Tensor, layer_idx: int) -> dict:
    """
    Compute extended spectral metrics from a single layer's cache_c matrix.

    Args:
        cache_c: (hidden_dim, hidden_dim) covariance accumulator
        layer_idx: which layer this cache belongs to

    Returns:
        dict with all extended metrics
    """
    C = cache_c.float().cuda()

    # Eigendecomposition (cache_c is symmetric PSD)
    eigvals = torch.linalg.eigvalsh(C)
    eigvals_pos = eigvals[eigvals > 1e-10]

    # Basic metrics (for cross-validation with mechanism_analyzer)
    svs = eigvals.flip(0).clamp(min=0).sqrt()
    svs_pos = svs[svs > 1e-10]
    numerical_rank = int((svs > 1e-5).sum().item())

    # Effective rank (entropy-based)
    if len(svs_pos) > 1:
        p = svs_pos / svs_pos.sum()
        entropy = -(p * torch.log(p)).sum().item()
        effective_rank = math.exp(entropy)
    else:
        effective_rank = float(len(svs_pos))

    # Extended metrics
    trace_C = eigvals_pos.sum().item()
    trace_C2 = (eigvals_pos ** 2).sum().item()
    spectral_norm = eigvals_pos.max().item() if len(eigvals_pos) > 0 else 0.0
    lambda_sum = eigvals_pos.sum().item()
    lambda_max = eigvals_pos.max().item() if len(eigvals_pos) > 0 else 0.0

    # Rayleigh quotient average: Tr(C²)/Tr(C)
    rayleigh_quotient_avg = trace_C2 / trace_C if trace_C > 1e-10 else 0.0

    # Key crowding proxy: Tr(C²)/Tr(C)²
    key_crowding_proxy = trace_C2 / (trace_C ** 2) if trace_C > 1e-10 else 0.0

    # Linear effective rank: Tr(C)/||C||₂
    linear_effective_rank = trace_C / spectral_norm if spectral_norm > 1e-10 else 0.0

    # Top eigenvalue concentration: λ₁/Σλᵢ
    top_eigenvalue_concentration = lambda_max / lambda_sum if lambda_sum > 1e-10 else 0.0

    # Condition number
    lambda_min = eigvals_pos.min().item() if len(eigvals_pos) > 0 else 0.0
    condition = lambda_max / lambda_min if lambda_min > 1e-10 else float("inf")

    return {
        "layer_idx": layer_idx,
        "numerical_rank": numerical_rank,
        "effective_rank": round(effective_rank, 2),
        "rayleigh_quotient_avg": round(rayleigh_quotient_avg, 6),
        "key_crowding_proxy": round(key_crowding_proxy, 8),
        "linear_effective_rank": round(linear_effective_rank, 2),
        "top_eigenvalue_concentration": round(top_eigenvalue_concentration, 6),
        "condition": round(condition, 2) if not math.isinf(condition) else "inf",
        "trace_C": round(trace_C, 4),
        "trace_C2": round(trace_C2, 4),
        "spectral_norm": round(spectral_norm, 6),
    }


def find_checkpoints(checkpoint_base: Path) -> list[tuple[int, Path]]:
    """
    Find all batch_N checkpoint directories and return sorted by batch index.

    Returns list of (batch_idx, ckpt_dir) tuples.
    """
    checkpoints = []
    for d in sorted(checkpoint_base.iterdir()):
        if d.is_dir() and d.name.startswith("batch_"):
            batch_idx = int(d.name.replace("batch_", ""))
            cache_path = d / "cache_c.pt"
            if cache_path.exists():
                checkpoints.append((batch_idx, d))
    return sorted(checkpoints, key=lambda x: x[0])


def main():
    parser = argparse.ArgumentParser(
        description="GPU-accelerated extended cache metrics for predictive divergence"
    )
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--checkpoint_base", type=str, required=True,
                        help="Directory containing batch_N/ subdirectories with cache_c.pt")
    parser.add_argument("--batch_size", type=int, default=100,
                        help="Edits per batch (for computing total_edits)")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Output directory (default: results/mechanism_analysis/seed{N})")
    args = parser.parse_args()

    checkpoint_base = Path(args.checkpoint_base)
    if not checkpoint_base.exists():
        print(f"ERROR: Checkpoint directory not found: {checkpoint_base}")
        sys.exit(1)

    output_dir = Path(args.output_dir) if args.output_dir else RESULTS_DIR / f"seed{args.seed}"
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    output_path = output_dir / f"extended_cache_seed{args.seed}_{timestamp}.jsonl"

    print("=" * 70)
    print("Extended Cache Metrics (GPU)")
    print(f"  Seed:            {args.seed}")
    print(f"  Checkpoint base: {checkpoint_base}")
    print(f"  Output:          {output_path}")
    print("=" * 70)

    # Find checkpoints
    checkpoints = find_checkpoints(checkpoint_base)
    if not checkpoints:
        print("ERROR: No checkpoint directories with cache_c.pt found.")
        sys.exit(1)
    print(f"\nFound {len(checkpoints)} checkpoints: "
          f"batch {checkpoints[0][0]} to batch {checkpoints[-1][0]}")

    # Process each checkpoint
    records_written = 0
    with open(output_path, "w") as f_out:
        for batch_idx, ckpt_dir in checkpoints:
            total_edits = (batch_idx + 1) * args.batch_size
            cache_path = ckpt_dir / "cache_c.pt"

            print(f"\n  Batch {batch_idx} ({total_edits} edits)...")
            cache_c = torch.load(cache_path, map_location="cpu")

            # cache_c shape: (n_layers, hidden_dim, hidden_dim)
            if cache_c.dim() == 3:
                n_layers = cache_c.shape[0]
                for layer_pos in range(n_layers):
                    metrics = compute_extended_cache_metrics(cache_c[layer_pos], layer_pos)
                    record = {
                        "seed": args.seed,
                        "batch_idx": batch_idx,
                        "total_edits": total_edits,
                        "layer_position": layer_pos,
                        **metrics,
                    }
                    f_out.write(json.dumps(record) + "\n")
                    records_written += 1
                print(f"    {n_layers} layers processed")
            elif cache_c.dim() == 2:
                # Single layer
                metrics = compute_extended_cache_metrics(cache_c, 0)
                record = {
                    "seed": args.seed,
                    "batch_idx": batch_idx,
                    "total_edits": total_edits,
                    "layer_position": 0,
                    **metrics,
                }
                f_out.write(json.dumps(record) + "\n")
                records_written += 1
                print(f"    1 layer processed")

            # Free GPU memory
            del cache_c
            torch.cuda.empty_cache()

    print(f"\n{'=' * 70}")
    print(f"Complete. Wrote {records_written} records to {output_path}")
    print("=" * 70)


if __name__ == "__main__":
    main()

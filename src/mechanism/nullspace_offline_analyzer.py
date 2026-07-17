#!/usr/bin/env python3
"""
Null-Space Offline Analyzer: Computes rank consumption metrics from
existing failure curve checkpoints WITHOUT re-editing the model.

This script produces output identical in format to nullspace_tracker.py
but reads cache_c directly from checkpoints saved by checkpoint_runner.py,
eliminating the need to reload the model and re-run editing.

Requirements:
  - Failure curve checkpoints must already exist (from checkpoint_runner.py)
  - Only AlphaEdit checkpoints contain cache_c (MEMIT has no projection)

Output: JSONL file with one record per checkpoint, same schema as
        nullspace_tracker.py (batch_idx, total_edits_so_far, per-layer metrics).

Usage:
    python src/nullspace_offline_analyzer.py \\
        --seed 42 \\
        --checkpoint_dir ~/.cache/alphaedit_checkpoints/AlphaEdit/seed42

    # Auto-detect checkpoint directory (same resolution as checkpoint_runner.py)
    python src/nullspace_offline_analyzer.py --seed 42
"""

import argparse
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch


# Default layers and threshold matching vendor/AlphaEdit/hparams/AlphaEdit/Llama3-8B.json
DEFAULT_LAYERS = [4, 5, 6, 7, 8]
DEFAULT_THRESHOLD = 2e-2


def get_project_root() -> Path:
    return Path(__file__).resolve().parent.parent.parent


def resolve_checkpoint_dir(explicit_dir: str | None, seed: int) -> Path:
    """Resolve checkpoint directory using same priority as checkpoint_runner.py."""
    if explicit_dir:
        return Path(explicit_dir)

    # Priority 1: S3 mount
    s3_path = Path("/s3-data/continual-learning/alphaedit/checkpoints/AlphaEdit") / f"seed{seed}"
    if s3_path.exists():
        return s3_path

    # Priority 2: Local cache
    return Path.home() / ".cache" / "alphaedit_checkpoints" / "AlphaEdit" / f"seed{seed}"


def find_all_checkpoints(ckpt_dir: Path) -> list[tuple[int, Path]]:
    """Find all valid checkpoints sorted by batch index."""
    if not ckpt_dir.exists():
        return []

    checkpoints = []
    for batch_dir in sorted(ckpt_dir.glob("batch_*")):
        metadata_file = batch_dir / "metadata.json"
        cache_file = batch_dir / "cache_c.pt"

        if not metadata_file.exists():
            continue
        if not cache_file.exists():
            continue

        try:
            batch_idx = int(batch_dir.name.split("_")[1])
            checkpoints.append((batch_idx, batch_dir))
        except (ValueError, IndexError):
            continue

    return checkpoints


def compute_rank_metrics(
    cache_c: torch.Tensor,
    layers: list[int],
    threshold: float,
) -> dict:
    """
    Compute null-space rank consumption metrics from cache_c tensor.

    Matches the exact computation in nullspace_tracker.py:
    - numerical_rank: count of singular values > threshold
    - effective_rank: exp(entropy(normalized_svs))
    - top singular values
    """
    per_layer = {}

    for i, layer in enumerate(layers):
        layer_record = {}

        if i >= cache_c.shape[0]:
            # Checkpoint doesn't have this layer's data
            continue

        cache_layer = cache_c[i].float()

        # Hidden dimension (from the square cache_c matrix)
        hidden_dim = cache_layer.shape[0]
        layer_record["hidden_dim"] = hidden_dim

        if cache_layer.abs().max() > 0:
            svs = torch.linalg.svdvals(cache_layer)

            # Numerical rank (same threshold as AlphaEdit's get_project())
            numerical_rank = int((svs > threshold).sum().item())
            layer_record["cache_c_numerical_rank"] = numerical_rank

            # Effective rank: exp(H(normalized_svs))
            svs_pos = svs[svs > 0]
            if len(svs_pos) > 1:
                p = svs_pos / svs_pos.sum()
                entropy = -(p * torch.log(p)).sum().item()
                effective_rank = math.exp(entropy)
            else:
                effective_rank = float(len(svs_pos))
            layer_record["cache_c_effective_rank"] = round(effective_rank, 2)

            # Top 10 singular values
            layer_record["cache_c_top_svs"] = svs[:10].tolist()

            # Note: nullspace_rank_initial requires P which is not in checkpoints.
            # We report it as None; it can be computed once from covariance stats
            # if needed (it's static and doesn't change across checkpoints).
            layer_record["nullspace_rank_initial"] = None
            layer_record["consumption_ratio"] = None
        else:
            layer_record["cache_c_numerical_rank"] = 0
            layer_record["cache_c_effective_rank"] = 0.0
            layer_record["cache_c_top_svs"] = []
            layer_record["nullspace_rank_initial"] = None
            layer_record["consumption_ratio"] = 0.0

        per_layer[str(layer)] = layer_record

    return per_layer


def analyze_checkpoints(
    ckpt_dir: Path,
    layers: list[int],
    threshold: float,
    output_jsonl: Path,
    device: str = "cpu",
) -> list[dict]:
    """Load each checkpoint's cache_c and compute rank metrics."""
    checkpoints = find_all_checkpoints(ckpt_dir)

    if not checkpoints:
        print(f"ERROR: No valid AlphaEdit checkpoints found in {ckpt_dir}")
        print("  Checkpoints must contain both metadata.json and cache_c.pt")
        sys.exit(1)

    print(f"  Found {len(checkpoints)} checkpoints")

    # Ensure output directory exists
    output_jsonl.parent.mkdir(parents=True, exist_ok=True)

    records = []
    for batch_idx, batch_dir in checkpoints:
        # Load metadata
        with open(batch_dir / "metadata.json", "r") as f:
            metadata = json.load(f)

        num_edits_per_batch = metadata.get("num_edits_per_batch", 100)
        total_edits = metadata.get("total_edits", (batch_idx + 1) * num_edits_per_batch)

        # Load cache_c
        cache_c = torch.load(
            str(batch_dir / "cache_c.pt"),
            map_location=device,
            weights_only=True,
        )

        # Compute metrics
        per_layer = compute_rank_metrics(cache_c, layers, threshold)

        record = {
            "batch_idx": batch_idx,
            "num_requests": num_edits_per_batch,
            "total_edits_so_far": total_edits,
            "total_edits_after": total_edits,
            "layers": per_layer,
            "source": "offline_analyzer",
            "checkpoint_path": str(batch_dir),
        }

        records.append(record)

        # Write incrementally
        with open(output_jsonl, "a") as f:
            f.write(json.dumps(record) + "\n")

        # Summary for this checkpoint
        ranks = [
            v.get("cache_c_numerical_rank", 0)
            for v in per_layer.values()
        ]
        print(f"  Batch {batch_idx:3d} ({total_edits:5d} edits): "
              f"numerical_rank per layer = {ranks}")

        # Free memory
        del cache_c

    return records


def main():
    parser = argparse.ArgumentParser(
        description="Offline null-space rank analysis from failure curve checkpoints"
    )
    parser.add_argument("--seed", type=int, required=True,
                        help="Seed used for the failure curve run")
    parser.add_argument("--checkpoint_dir", type=str, default=None,
                        help="Explicit checkpoint directory (default: auto-resolve)")
    parser.add_argument("--layers", type=int, nargs="+", default=DEFAULT_LAYERS,
                        help=f"Layers to analyze (default: {DEFAULT_LAYERS})")
    parser.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD,
                        help=f"Singular value threshold for rank (default: {DEFAULT_THRESHOLD})")
    parser.add_argument("--device", default="cpu",
                        help="Device for SVD computation (default: cpu, use cuda for speed)")
    parser.add_argument("--output", type=str, default=None,
                        help="Output JSONL path (default: results/nullspace_tracking/offline_seed{seed}.jsonl)")

    args = parser.parse_args()

    project_root = get_project_root()

    # Resolve checkpoint directory
    ckpt_dir = resolve_checkpoint_dir(args.checkpoint_dir, args.seed)

    # Resolve output path
    if args.output:
        output_jsonl = Path(args.output)
    else:
        output_dir = project_root / "results" / "nullspace_tracking"
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        output_jsonl = output_dir / f"offline_rank_trace_seed{args.seed}_{timestamp}.jsonl"

    print(f"{'=' * 70}")
    print("Null-Space Offline Analyzer")
    print(f"  Seed:            {args.seed}")
    print(f"  Checkpoint dir:  {ckpt_dir}")
    print(f"  Layers:          {args.layers}")
    print(f"  Threshold:       {args.threshold}")
    print(f"  Device:          {args.device}")
    print(f"  Output:          {output_jsonl}")
    print(f"  Started:         {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}")
    print(f"{'=' * 70}")

    if not ckpt_dir.exists():
        print(f"\nERROR: Checkpoint directory does not exist: {ckpt_dir}")
        print("  Run the failure curve first:")
        print(f"    EVAL_AT_CHECKPOINTS_ONLY=true bash scripts/run_failure_curve_checkpointed.sh {args.seed} AlphaEdit 10000")
        sys.exit(1)

    records = analyze_checkpoints(
        ckpt_dir=ckpt_dir,
        layers=args.layers,
        threshold=args.threshold,
        output_jsonl=output_jsonl,
        device=args.device,
    )

    print(f"\n{'=' * 70}")
    print("Offline null-space analysis complete.")
    print(f"  Checkpoints analyzed: {len(records)}")
    print(f"  Output: {output_jsonl}")
    if records:
        last = records[-1]
        print(f"  Final state ({last['total_edits_so_far']} edits):")
        for layer_str, data in last["layers"].items():
            rank = data.get("cache_c_numerical_rank", "?")
            eff = data.get("cache_c_effective_rank", "?")
            print(f"    Layer {layer_str}: numerical_rank={rank}, effective_rank={eff}")
    print(f"  Finished: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()

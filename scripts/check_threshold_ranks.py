#!/usr/bin/env python3
"""Check what effective rank (r/d) each nullspace_threshold produces for GPT-J.

Loads covariance stats for each edited layer, computes SVD, and reports the
null-space rank at each threshold. Use this to verify which thresholds give
the target r/d values (e.g., 20.7% and 56.1%) before launching GPU jobs.

IMPORTANT: The threshold is compared ABSOLUTELY against singular values:
    small_singular_indices = (S < threshold)
This matches vendor/AlphaEdit/experiments/evaluate.py:451 and
src/runners/checkpoint_runner.py:388.

Usage:
    uv run python scripts/check_threshold_ranks.py
    uv run python scripts/check_threshold_ranks.py --thresholds 1 5 10 50 100 500
    uv run python scripts/check_threshold_ranks.py --model qwen2.5-7b
"""

import argparse
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent

MODEL_CONFIGS = {
    "gpt-j-6b": {
        "stats_subdir": "gpt-j-6b",
        "layers": [3, 4, 5, 6, 7, 8],
        "layer_template": "transformer.h.{layer}.mlp.fc_out_float32_mom2_100000.npz",
    },
    "qwen2.5-7b": {
        "stats_subdir": "qwen2.5-7b-instruct",
        "layers": [4, 5, 6, 7, 8, 9],
        "layer_template": "model.layers.{layer}.mlp.down_proj_float32_mom2_100000.npz",
    },
    "llama3-8b": {
        "stats_subdir": "llama3-8b-instruct",
        "layers": [4, 5, 6, 7, 8, 9],
        "layer_template": "model.layers.{layer}.mlp.down_proj_float32_mom2_100000.npz",
    },
}


def compute_ranks_at_thresholds(stats_dir: Path, model_config: dict, thresholds: list[float]) -> dict:
    """For each threshold, compute null-space rank per layer and average r/d.

    The threshold is ABSOLUTE: rank(P) = count(S < threshold).
    This matches the actual code in evaluate.py and checkpoint_runner.py.
    """
    results = {}
    layers = model_config["layers"]
    layer_template = model_config["layer_template"]

    layer_eigenvalues = {}
    for layer_num in layers:
        stats_file = stats_dir / layer_template.format(layer=layer_num)
        if not stats_file.exists():
            print(f"  WARNING: Missing stats for layer {layer_num}: {stats_file}")
            continue

        data = np.load(stats_file)
        mom2 = data["mom2.mom2"]
        count = int(data["mom2.count"])
        cov = mom2 / count if count > 1 else mom2

        print(f"  Computing SVD for layer {layer_num} ({cov.shape[0]}x{cov.shape[1]})...")
        S = np.linalg.svdvals(cov)
        layer_eigenvalues[layer_num] = np.sort(S)[::-1]  # descending

    if not layer_eigenvalues:
        print("ERROR: No covariance stats found")
        sys.exit(1)

    d = len(next(iter(layer_eigenvalues.values())))

    # Print singular value distribution to help user choose thresholds
    print(f"\n  Total dimensions per layer: {d}")
    print(f"  Layers: {list(layer_eigenvalues.keys())}")
    print(f"\n  Singular value distribution (across all layers):")
    all_S = np.concatenate(list(layer_eigenvalues.values()))
    print(f"    min:    {all_S.min():.4f}")
    print(f"    p5:     {np.percentile(all_S, 5):.4f}")
    print(f"    p25:    {np.percentile(all_S, 25):.4f}")
    print(f"    median: {np.median(all_S):.4f}")
    print(f"    p75:    {np.percentile(all_S, 75):.4f}")
    print(f"    p95:    {np.percentile(all_S, 95):.4f}")
    print(f"    max:    {all_S.max():.4f}")

    print(f"\n  Per-layer max singular value:")
    for layer_num, S in layer_eigenvalues.items():
        print(f"    Layer {layer_num}: S_max={S[0]:.2f}, S_min={S[-1]:.6f}, S_median={np.median(S):.4f}")

    # Compute ranks at each threshold (ABSOLUTE comparison: S < threshold)
    print(f"\n{'Threshold':>10} | {'Avg r/d':>8} | " +
          " | ".join(f"  L{l}  " for l in layer_eigenvalues.keys()))
    print("-" * (10 + 3 + 8 + 3 + len(layer_eigenvalues) * 9))

    for t in sorted(thresholds):
        layer_ranks = {}
        for layer_num, S in layer_eigenvalues.items():
            # ABSOLUTE threshold: count singular values BELOW threshold
            # This is the null-space rank = rank(P)
            rank_P = int(np.sum(S < t))
            layer_ranks[layer_num] = rank_P

        avg_r_over_d = np.mean([r / d for r in layer_ranks.values()])

        results[t] = {
            "avg_r_over_d": float(avg_r_over_d),
            "layer_ranks": {str(l): int(r) for l, r in layer_ranks.items()},
            "layer_r_over_d": {str(l): float(r / d) for l, r in layer_ranks.items()},
        }

        layer_pcts = " | ".join(f"{r/d*100:5.1f}%" for r in layer_ranks.values())
        print(f"  {t:>8.4f} | {avg_r_over_d*100:6.1f}% | {layer_pcts}")

    # Find thresholds closest to target r/d values
    print("\n" + "=" * 60)
    print("  Target r/d matching:")
    targets = [0.207, 0.561]
    for target in targets:
        best_t = min(thresholds, key=lambda t: abs(results[t]["avg_r_over_d"] - target))
        actual = results[best_t]["avg_r_over_d"]
        print(f"    r/d={target*100:.1f}%: closest threshold={best_t} "
              f"(actual avg r/d={actual*100:.1f}%)")

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Check threshold->rank mapping (ABSOLUTE threshold, matching real code)"
    )
    parser.add_argument("--model", type=str, default="gpt-j-6b",
                        choices=list(MODEL_CONFIGS.keys()),
                        help="Model to check thresholds for")
    parser.add_argument("--thresholds", nargs="+", type=float, default=None,
                        help="Thresholds to evaluate (absolute singular value cutoffs). "
                             "If not specified, auto-generates from data distribution.")
    args = parser.parse_args()

    config = MODEL_CONFIGS[args.model]
    stats_dir = PROJECT_ROOT / "data" / "stats" / config["stats_subdir"] / "wikipedia_stats"

    print("=" * 60)
    print(f"Nullspace Threshold -> Rank Mapping ({args.model})")
    print("=" * 60)
    print(f"\n  Stats dir: {stats_dir}")
    print(f"  NOTE: threshold is ABSOLUTE (S < threshold = null-space dims)")

    if not stats_dir.exists():
        print(f"\n  ERROR: Stats directory not found: {stats_dir}")
        print("  Run: bash scripts/link_stats.sh")
        sys.exit(1)

    # If no thresholds specified, auto-generate from distribution
    if args.thresholds is None:
        # Load one layer to determine scale
        layer_num = config["layers"][0]
        sample_file = stats_dir / config["layer_template"].format(layer=layer_num)
        if sample_file.exists():
            data = np.load(sample_file)
            mom2 = data["mom2.mom2"]
            count = int(data["mom2.count"])
            cov = mom2 / count if count > 1 else mom2
            S_sample = np.linalg.svdvals(cov)
            S_max = S_sample.max()
            S_median = np.median(S_sample)
            # Generate thresholds spanning the range where rank changes
            thresholds = sorted(set([
                S_median * 0.01,
                S_median * 0.1,
                S_median * 0.5,
                S_median * 1.0,
                S_median * 2.0,
                S_median * 5.0,
                S_median * 10.0,
                S_max * 0.01,
                S_max * 0.05,
                S_max * 0.1,
                S_max * 0.2,
                S_max * 0.5,
            ]))
            # Round for readability
            thresholds = [round(t, 4) if t < 1 else round(t, 2) for t in thresholds]
            thresholds = sorted(set(thresholds))
            print(f"  Auto-generated {len(thresholds)} thresholds from data distribution")
        else:
            # Fallback: use a wide range
            thresholds = [0.1, 0.5, 1.0, 5.0, 10.0, 50.0, 100.0, 500.0, 1000.0]
            print("  Using fallback threshold range (couldn't load sample stats)")
    else:
        thresholds = args.thresholds

    compute_ranks_at_thresholds(stats_dir, config, thresholds)

    print("\n" + "=" * 60)
    print("Use the threshold values above in run_capacity_replication_seed2024.sh:")
    print("  THRESHOLD_BINDING=<value>     # for r/d ~ 20.7%")
    print("  THRESHOLD_PERMISSIVE=<value>  # for r/d ~ 56.1%")
    print("=" * 60)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Sweep k-means cluster count to find optimal separation.

Tests k∈{50, 75, 100} (configurable) and reports within-batch cosine
for both key-clustered and key-dispersed orderings at each k value.
Selects the k that maximizes the cosine ratio while maintaining
reasonable batch balance.

No model loading needed — uses precomputed keys.

Usage:
    uv run python analysis/sweep_k_clusters.py \
        --seed 42 \
        --keys_path results/matched_ordering/key_geometry/keys_seed42_layer6.npz \
        --stream_path results/matched_ordering/orderings/clustered_seed42.json \
        --k_values 30 50 75 100
"""

import argparse
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src" / "datasets"))

from generate_orderings import (
    spherical_kmeans,
    create_key_clustered_ordering,
    create_key_dispersed_ordering,
)


def evaluate_ordering(keys, ordering, batch_size, case_id_to_idx):
    """Compute within-batch cosine stats for an ordering."""
    norms = np.linalg.norm(keys, axis=1, keepdims=True)
    normed = keys / np.maximum(norms, 1e-8)

    n_batches = len(ordering) // batch_size
    batch_means = []
    all_pairs = []

    for b in range(n_batches):
        batch_records = ordering[b * batch_size: (b + 1) * batch_size]
        indices = [case_id_to_idx[r["case_id"]] for r in batch_records]
        batch_normed = normed[indices]
        cos_matrix = batch_normed @ batch_normed.T
        n = len(indices)
        mask = np.triu(np.ones((n, n), dtype=bool), k=1)
        pairs = cos_matrix[mask]
        batch_means.append(float(pairs.mean()))
        all_pairs.append(pairs)

    flat = np.concatenate(all_pairs)
    return {
        "mean": float(np.mean(batch_means)),
        "median": float(np.median(batch_means)),
        "p95": float(np.percentile(flat, 95)),
        "p99": float(np.percentile(flat, 99)),
        "frac_above_0.2": float((flat > 0.2).mean()),
        "frac_above_0.3": float((flat > 0.3).mean()),
        "mean_batch_max": float(np.mean([np.max(p) for p in all_pairs])),
    }


def main():
    parser = argparse.ArgumentParser(description="Sweep k-means cluster count")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--keys_path", type=str, required=True)
    parser.add_argument("--stream_path", type=str, required=True)
    parser.add_argument("--k_values", type=int, nargs="+", default=[30, 50, 75, 100])
    parser.add_argument("--batch_size", type=int, default=100)
    args = parser.parse_args()

    keys_path = Path(args.keys_path)
    if not keys_path.is_absolute():
        keys_path = PROJECT_ROOT / args.keys_path
    stream_path = Path(args.stream_path)
    if not stream_path.is_absolute():
        stream_path = PROJECT_ROOT / args.stream_path

    print(f"\n{'='*70}")
    print("K-Cluster Sweep")
    print(f"  Keys: {keys_path}")
    print(f"  k values: {args.k_values}")
    print(f"  Batch size: {args.batch_size}")
    print(f"{'='*70}")

    # Load
    npz = np.load(keys_path)
    keys = npz["keys"]
    saved_case_ids = npz["case_ids"].tolist()

    with open(stream_path) as f:
        all_records = json.load(f)

    # Align records with keys
    record_by_id = {r["case_id"]: r for r in all_records}
    records = [record_by_id[cid] for cid in saved_case_ids]
    case_id_to_idx = {r["case_id"]: i for i, r in enumerate(records)}

    print(f"  Loaded {len(keys)} keys, {len(records)} records")

    # Sweep
    results = []
    print(f"\n  {'k':>5} {'Clusters':>8} {'Min':>5} {'Max':>5} {'Std':>6} | "
          f"{'KC mean':>8} {'KD mean':>8} {'Ratio':>6} | "
          f"{'KC p95':>7} {'KD p95':>7} {'Ratio':>6} | "
          f"{'KC >0.2':>7} {'KD >0.2':>7} {'Ratio':>6}")
    print(f"  {'─'*105}")

    for k in args.k_values:
        # Cluster
        assignments = spherical_kmeans(keys, k, max_iter=100, seed=args.seed)
        cluster_sizes = np.bincount(assignments)

        # Generate orderings
        rng_c = random.Random(args.seed + 3000)
        rng_d = random.Random(args.seed + 4000)
        kc = create_key_clustered_ordering(records, assignments, rng_c, args.batch_size)
        kd = create_key_dispersed_ordering(records, assignments, rng_d, args.batch_size)

        # Evaluate
        kc_stats = evaluate_ordering(keys, kc, args.batch_size, case_id_to_idx)
        kd_stats = evaluate_ordering(keys, kd, args.batch_size, case_id_to_idx)

        mean_ratio = kc_stats['mean'] / kd_stats['mean']
        p95_ratio = kc_stats['p95'] / kd_stats['p95']
        frac02_ratio = kc_stats['frac_above_0.2'] / max(kd_stats['frac_above_0.2'], 1e-10)

        print(f"  {k:>5} {len(cluster_sizes):>8} {cluster_sizes.min():>5} "
              f"{cluster_sizes.max():>5} {cluster_sizes.std():>6.1f} | "
              f"{kc_stats['mean']:>8.4f} {kd_stats['mean']:>8.4f} {mean_ratio:>6.2f}x | "
              f"{kc_stats['p95']:>7.4f} {kd_stats['p95']:>7.4f} {p95_ratio:>6.2f}x | "
              f"{kc_stats['frac_above_0.2']:>7.4f} {kd_stats['frac_above_0.2']:>7.4f} {frac02_ratio:>6.2f}x")

        results.append({
            "k": k,
            "n_clusters": len(cluster_sizes),
            "cluster_sizes": {"min": int(cluster_sizes.min()), "max": int(cluster_sizes.max()),
                            "mean": float(cluster_sizes.mean()), "std": float(cluster_sizes.std())},
            "key_clustered": kc_stats,
            "key_dispersed": kd_stats,
            "ratios": {"mean": mean_ratio, "p95": p95_ratio, "frac_above_0.2": frac02_ratio},
        })

    # Best k
    best = max(results, key=lambda r: r["ratios"]["mean"])
    print(f"\n  Best k={best['k']}: mean ratio={best['ratios']['mean']:.2f}x, "
          f"p95 ratio={best['ratios']['p95']:.2f}x, "
          f"frac>0.2 ratio={best['ratios']['frac_above_0.2']:.2f}x")

    # Save
    out_dir = PROJECT_ROOT / "results" / "matched_ordering" / "diagnostics"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"k_sweep_seed{args.seed}.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"  Saved: {out_path}")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Cross-batch cosine analysis: Why dispersed streams cause more forgetting.

Hypothesis: In dispersed streams, every old edit is guaranteed to encounter a
high-cosine subsequent key (because all clusters are represented in every batch).
In clustered streams, old edits from cluster A rarely encounter subsequent keys
from cluster A (those keys are temporally concentrated), so most old edits have
LOW max_cos_subsequent.

This script computes, for each edit position in each ordering:
  - max_cos_subsequent: max cosine to any key inserted after this position
  - mean_cos_subsequent: mean cosine to all subsequent keys
  - n_high_cos_subsequent: count of subsequent keys with cosine > threshold

Then compares these distributions between key_clustered and key_dispersed.

Requires:
  - Key vectors: results/matched_ordering/key_geometry/keys_seed42_layer6.npz
  - Stream orderings: results/matched_ordering/orderings/key_{clustered,dispersed}_seed{N}.json

Usage:
    uv run python -m analysis.cross_batch_cosine
    uv run python -m analysis.cross_batch_cosine --seed 42
    uv run python -m analysis.cross_batch_cosine --seed 2024 --output-dir results/figures/paper
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

try:
    from analysis.style import RESULTS, PAPER_OUTPUT
except ImportError:
    RESULTS = Path(__file__).resolve().parent.parent / "results"
    PAPER_OUTPUT = RESULTS / "figures" / "paper"

ORDERINGS_DIR = RESULTS / "matched_ordering" / "orderings"
KEY_GEOMETRY_DIR = RESULTS / "matched_ordering" / "key_geometry"
FULL_EVAL_DIR = RESULTS / "matched_ordering"

BATCH_SIZE = 100
HIGH_COS_THRESHOLD = 0.3  # What counts as "high" cosine overlap


# ─── Data Loading ────────────────────────────────────────────────────────────


def load_keys_and_index(seed: int) -> Tuple[np.ndarray, Dict[int, int]]:
    """Load key vectors and build case_id → index mapping.

    Returns:
        keys: (N, D) float32 array, L2-normalized
        case_id_to_idx: dict mapping case_id → row index in keys array
    """
    keys_path = KEY_GEOMETRY_DIR / f"keys_seed{seed}_layer6.npz"
    if not keys_path.exists():
        # Fall back to full_mcf keys
        keys_path = RESULTS / "key_vectors" / "full_mcf" / f"keys_seed{seed}_layer6.npz"
    if not keys_path.exists():
        raise FileNotFoundError(f"No key vectors found for seed {seed}")

    data = np.load(keys_path)
    keys = data["keys"].astype(np.float32)
    case_ids = data["case_ids"]

    # L2 normalize
    norms = np.linalg.norm(keys, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-8)
    keys = keys / norms

    case_id_to_idx = {int(cid): i for i, cid in enumerate(case_ids)}
    return keys, case_id_to_idx


def load_ordering(seed: int, ordering: str) -> List[int]:
    """Load stream ordering and return list of case_ids in order."""
    path = ORDERINGS_DIR / f"{ordering}_seed{seed}.json"
    if not path.exists():
        raise FileNotFoundError(f"No ordering file: {path}")

    with open(path) as f:
        records = json.load(f)

    return [r["case_id"] for r in records]


def load_full_eval_outcomes(seed: int, ordering: str, alg: str = "AlphaEdit") -> Dict[int, bool]:
    """Load retention outcomes at 5K for each case_id.

    Returns dict: case_id → True if retained (efficacy > 0.5 at final checkpoint).
    """
    eval_path = FULL_EVAL_DIR / alg / ordering / f"seed{seed}" / f"full_eval_seed{seed}.json"
    if not eval_path.exists():
        return {}

    with open(eval_path) as f:
        data = json.load(f)

    # Use 5K checkpoint cohort data to determine per-case outcomes
    # We don't have per-case outcomes directly in full_eval, so use cohort-level
    # For this analysis, we'll use the cohort efficacy as a proxy
    return data


# ─── Core Computation ────────────────────────────────────────────────────────


def compute_subsequent_cosines(
    keys: np.ndarray,
    case_id_to_idx: Dict[int, int],
    ordering: List[int],
    max_position: int = 5000,
) -> Dict[str, np.ndarray]:
    """For each edit position, compute cosine statistics to subsequent keys.

    Args:
        keys: (N, D) normalized key vectors
        case_id_to_idx: case_id → row in keys
        ordering: list of case_ids in stream order
        max_position: only analyze first N edits (to match 5K evaluation)

    Returns dict with arrays indexed by position:
        max_cos: max cosine to any subsequent key
        mean_cos: mean cosine to all subsequent keys
        n_high: count of subsequent keys with cos > threshold
        percentile_90: 90th percentile of subsequent cosines
    """
    n = min(len(ordering), max_position)

    # Build key matrix in stream order
    valid_positions = []
    stream_keys = []
    for pos in range(n):
        cid = ordering[pos]
        if cid in case_id_to_idx:
            valid_positions.append(pos)
            stream_keys.append(keys[case_id_to_idx[cid]])

    stream_keys = np.array(stream_keys, dtype=np.float32)
    n_valid = len(valid_positions)

    print(f"  Computing cosines for {n_valid} valid positions...")

    # Compute full pairwise cosine matrix (n_valid × n_valid)
    # Since keys are normalized, cosine = dot product
    cos_matrix = stream_keys @ stream_keys.T  # (n_valid, n_valid)

    # For each position i, subsequent positions are j > i
    max_cos = np.full(n_valid, np.nan)
    mean_cos = np.full(n_valid, np.nan)
    n_high = np.zeros(n_valid, dtype=np.int32)
    percentile_90 = np.full(n_valid, np.nan)

    # Process in chunks to show progress
    for i in range(n_valid - 1):
        subsequent = cos_matrix[i, i+1:]  # all cosines to keys after position i
        max_cos[i] = subsequent.max()
        mean_cos[i] = subsequent.mean()
        n_high[i] = (subsequent > HIGH_COS_THRESHOLD).sum()
        percentile_90[i] = np.percentile(subsequent, 90)

    return {
        "positions": np.array(valid_positions),
        "max_cos": max_cos,
        "mean_cos": mean_cos,
        "n_high": n_high,
        "percentile_90": percentile_90,
    }


def compute_first_1k_exposure(
    keys: np.ndarray,
    case_id_to_idx: Dict[int, int],
    ordering: List[int],
) -> Dict[str, np.ndarray]:
    """For just the first 1K edits, compute their max_cos to ALL subsequent keys.

    This directly measures "how exposed are early edits to future interference?"
    """
    n = min(len(ordering), 5000)

    # Build stream key matrix
    stream_keys = []
    stream_positions = []
    for pos in range(n):
        cid = ordering[pos]
        if cid in case_id_to_idx:
            stream_keys.append(keys[case_id_to_idx[cid]])
            stream_positions.append(pos)

    stream_keys = np.array(stream_keys, dtype=np.float32)
    n_valid = len(stream_positions)

    # Only analyze first 1K positions
    first_1k_end = 0
    for i, pos in enumerate(stream_positions):
        if pos >= 1000:
            break
        first_1k_end = i + 1

    print(f"  First 1K: {first_1k_end} valid keys, subsequent: {n_valid - first_1k_end} keys")

    first_1k_keys = stream_keys[:first_1k_end]
    subsequent_keys = stream_keys[first_1k_end:]

    if len(subsequent_keys) == 0:
        return {"max_cos": np.array([]), "mean_cos": np.array([])}

    # Compute cosines: (first_1k) × (subsequent)
    cos_matrix = first_1k_keys @ subsequent_keys.T

    return {
        "max_cos": cos_matrix.max(axis=1),
        "mean_cos": cos_matrix.mean(axis=1),
        "n_high": (cos_matrix > HIGH_COS_THRESHOLD).sum(axis=1),
        "percentile_90": np.percentile(cos_matrix, 90, axis=1),
        "percentile_99": np.percentile(cos_matrix, 99, axis=1),
    }


# ─── Analysis ────────────────────────────────────────────────────────────────


def compare_orderings(seed: int) -> Dict:
    """Run full comparison between key_clustered and key_dispersed for one seed."""
    print(f"\n{'='*60}")
    print(f"  CROSS-BATCH COSINE ANALYSIS — Seed {seed}")
    print(f"{'='*60}")

    keys, case_id_to_idx = load_keys_and_index(seed)
    print(f"  Loaded {len(case_id_to_idx)} key vectors, dim={keys.shape[1]}")

    results = {}

    for ordering_name in ["key_clustered", "key_dispersed"]:
        print(f"\n  --- {ordering_name} ---")
        ordering = load_ordering(seed, ordering_name)
        print(f"  Stream length: {len(ordering)}")

        # Full position-level analysis
        print(f"  Computing full subsequent cosines...")
        full_stats = compute_subsequent_cosines(keys, case_id_to_idx, ordering)

        # First-1K exposure analysis
        print(f"  Computing first-1K exposure...")
        first_1k_stats = compute_first_1k_exposure(keys, case_id_to_idx, ordering)

        results[ordering_name] = {
            "full": {
                "max_cos_mean": float(np.nanmean(full_stats["max_cos"])),
                "max_cos_median": float(np.nanmedian(full_stats["max_cos"])),
                "max_cos_std": float(np.nanstd(full_stats["max_cos"])),
                "mean_cos_mean": float(np.nanmean(full_stats["mean_cos"])),
                "n_high_mean": float(np.mean(full_stats["n_high"])),
                "n_high_median": float(np.median(full_stats["n_high"])),
                "pct90_mean": float(np.nanmean(full_stats["percentile_90"])),
            },
            "first_1k": {
                "max_cos_mean": float(np.mean(first_1k_stats["max_cos"])),
                "max_cos_median": float(np.median(first_1k_stats["max_cos"])),
                "max_cos_std": float(np.std(first_1k_stats["max_cos"])),
                "max_cos_min": float(np.min(first_1k_stats["max_cos"])),
                "max_cos_max": float(np.max(first_1k_stats["max_cos"])),
                "mean_cos_mean": float(np.mean(first_1k_stats["mean_cos"])),
                "n_high_mean": float(np.mean(first_1k_stats["n_high"])),
                "n_high_median": float(np.median(first_1k_stats["n_high"])),
                "pct90_mean": float(np.mean(first_1k_stats["percentile_90"])),
                "pct99_mean": float(np.mean(first_1k_stats["percentile_99"])),
                "frac_above_0.3": float(np.mean(first_1k_stats["max_cos"] > 0.3)),
                "frac_above_0.4": float(np.mean(first_1k_stats["max_cos"] > 0.4)),
                "frac_above_0.5": float(np.mean(first_1k_stats["max_cos"] > 0.5)),
            },
        }

        # Print summary
        f1k = results[ordering_name]["first_1k"]
        print(f"    First-1K max_cos to subsequent keys:")
        print(f"      mean={f1k['max_cos_mean']:.4f}, median={f1k['max_cos_median']:.4f}")
        print(f"      min={f1k['max_cos_min']:.4f}, max={f1k['max_cos_max']:.4f}")
        print(f"      frac > 0.3: {f1k['frac_above_0.3']:.3f}")
        print(f"      frac > 0.4: {f1k['frac_above_0.4']:.3f}")
        print(f"      frac > 0.5: {f1k['frac_above_0.5']:.3f}")
        print(f"      mean n_subsequent with cos > {HIGH_COS_THRESHOLD}: {f1k['n_high_mean']:.1f}")

    # Compute effect sizes
    clust_max = results["key_clustered"]["first_1k"]["max_cos_mean"]
    disp_max = results["key_dispersed"]["first_1k"]["max_cos_mean"]
    diff = disp_max - clust_max

    results["comparison"] = {
        "first_1k_max_cos_diff": diff,
        "first_1k_max_cos_ratio": disp_max / clust_max if clust_max > 0 else float("inf"),
        "dispersed_minus_clustered": {
            "max_cos_mean": disp_max - clust_max,
            "n_high_mean": (
                results["key_dispersed"]["first_1k"]["n_high_mean"]
                - results["key_clustered"]["first_1k"]["n_high_mean"]
            ),
            "frac_above_0.4": (
                results["key_dispersed"]["first_1k"]["frac_above_0.4"]
                - results["key_clustered"]["first_1k"]["frac_above_0.4"]
            ),
        },
    }

    print(f"\n  {'='*50}")
    print(f"  COMPARISON (dispersed - clustered):")
    print(f"    First-1K max_cos_mean: {diff:+.4f}")
    print(f"    (clustered={clust_max:.4f}, dispersed={disp_max:.4f})")
    print(f"  {'='*50}")

    return results


def compute_by_age_bin(
    keys: np.ndarray,
    case_id_to_idx: Dict[int, int],
    ordering: List[int],
) -> List[Dict]:
    """Compute max_cos_subsequent binned by edit age (1K cohorts)."""
    n = min(len(ordering), 5000)

    stream_keys = []
    for pos in range(n):
        cid = ordering[pos]
        if cid in case_id_to_idx:
            stream_keys.append(keys[case_id_to_idx[cid]])
        else:
            stream_keys.append(np.zeros(keys.shape[1]))

    stream_keys = np.array(stream_keys, dtype=np.float32)

    bins = []
    bin_labels = ["0-1K", "1K-2K", "2K-3K", "3K-4K", "4K-5K"]

    for bin_idx, label in enumerate(bin_labels):
        start = bin_idx * 1000
        end = start + 1000
        if end > n:
            break

        bin_keys = stream_keys[start:end]
        subsequent_keys = stream_keys[end:]

        if len(subsequent_keys) == 0:
            bins.append({"label": label, "max_cos_mean": float("nan"), "n": 0})
            continue

        cos_matrix = bin_keys @ subsequent_keys.T
        max_cos = cos_matrix.max(axis=1)

        bins.append({
            "label": label,
            "max_cos_mean": float(np.mean(max_cos)),
            "max_cos_median": float(np.median(max_cos)),
            "max_cos_std": float(np.std(max_cos)),
            "n_edits": int(end - start),
            "n_subsequent": len(subsequent_keys),
        })

    return bins


# ─── Main ────────────────────────────────────────────────────────────────────


def generate(seeds: List[int], output_dir: Path):
    """Run analysis for all seeds and save results."""
    output_dir.mkdir(parents=True, exist_ok=True)

    all_results = {}
    for seed in seeds:
        try:
            results = compare_orderings(seed)

            # Also compute age-binned breakdown
            keys, case_id_to_idx = load_keys_and_index(seed)
            for ordering_name in ["key_clustered", "key_dispersed"]:
                ordering = load_ordering(seed, ordering_name)
                age_bins = compute_by_age_bin(keys, case_id_to_idx, ordering)
                results[ordering_name]["age_bins"] = age_bins

            all_results[f"seed{seed}"] = results
        except FileNotFoundError as e:
            print(f"  SKIP seed {seed}: {e}")

    # Print final comparison table
    print(f"\n\n{'='*70}")
    print("FINAL SUMMARY: First-1K Exposure to Subsequent Keys")
    print(f"{'='*70}")
    print(f"{'Seed':<8} {'Ordering':<16} {'max_cos':<10} {'n_high(>0.3)':<14} {'frac>0.4':<10}")
    print("-" * 70)
    for seed_key, results in all_results.items():
        for ordering in ["key_clustered", "key_dispersed"]:
            f1k = results[ordering]["first_1k"]
            print(f"{seed_key:<8} {ordering:<16} {f1k['max_cos_mean']:.4f}    "
                  f"{f1k['n_high_mean']:<14.1f} {f1k['frac_above_0.4']:.3f}")

    print(f"\n{'='*70}")
    print("AGE-BINNED max_cos_subsequent (mean)")
    print(f"{'='*70}")
    print(f"{'Seed':<8} {'Ordering':<16} {'0-1K':<8} {'1K-2K':<8} {'2K-3K':<8} {'3K-4K':<8} {'4K-5K':<8}")
    print("-" * 70)
    for seed_key, results in all_results.items():
        for ordering in ["key_clustered", "key_dispersed"]:
            age_bins = results[ordering]["age_bins"]
            vals = [f"{b['max_cos_mean']:.4f}" if not np.isnan(b['max_cos_mean']) else "  —  "
                    for b in age_bins]
            print(f"{seed_key:<8} {ordering:<16} {'  '.join(vals)}")

    # Save
    out_path = output_dir / "cross_batch_cosine_analysis.json"
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\n  Saved: {out_path}")

    return all_results


def main():
    parser = argparse.ArgumentParser(
        description="Cross-batch cosine analysis: why dispersed streams cause more forgetting"
    )
    parser.add_argument("--seed", type=int, nargs="+", default=[42])
    parser.add_argument("--output-dir", type=Path, default=PAPER_OUTPUT)
    args = parser.parse_args()

    generate(args.seed, args.output_dir)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Geometric validation of scheduling orderings (Phase 2 gate).

For each ordering x seed, computes Table 9/10 metric suite:
  - mean max-cos to any subsequent edit
  - fraction with max-cos > 0.3 and > 0.4
  - mean # subsequent keys with cos > 0.3
  - within-batch mean pairwise cosine

Gate criterion: greedy_minmax first-1K "frac max-cos > 0.3" must be LOWER than
key_clustered's value. If it fails, the experiment premise is invalid and no GPU
runs should be launched.

Reuses analysis.cross_batch_cosine functions directly.

Usage:
    uv run python scheduling/validate_ordering.py --seed 42
    uv run python scheduling/validate_ordering.py --seed 42 --seed 2024
    uv run python scheduling/validate_ordering.py --seed 42 --orderings greedy_minmax,random,key_clustered,key_dispersed
"""

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from analysis.cross_batch_cosine import (
    compute_by_age_bin,
    compute_first_1k_exposure,
    compute_subsequent_cosines,
    load_ordering,
)

RESULT_ROOT = Path(os.environ.get("RESULT_ROOT", PROJECT_ROOT / "results"))
ORDERINGS_DIR = RESULT_ROOT / "matched_ordering" / "orderings"
KEYS_PATH = RESULT_ROOT / "key_vectors" / "full_mcf" / "keys_seed42_layer6.npz"
REPORTS_DIR = PROJECT_ROOT / "scheduling" / "reports"

BATCH_SIZE = 100
HIGH_COS_THRESHOLD = 0.3


def load_keys_and_index(seed: int):
    """Load key vectors with fallback to seed42 (keys are model-intrinsic).

    The key vectors are extracted from the base model and are the same
    regardless of editing seed. Only case_id mapping matters.
    """
    # Try seed-specific file first
    from analysis.cross_batch_cosine import KEY_GEOMETRY_DIR
    candidates = [
        KEY_GEOMETRY_DIR / f"keys_seed{seed}_layer6.npz",
        RESULT_ROOT / "key_vectors" / "full_mcf" / f"keys_seed{seed}_layer6.npz",
        # Fallback: keys are model-intrinsic, always use seed42
        KEYS_PATH,
    ]

    for keys_path in candidates:
        if keys_path.exists():
            data = np.load(keys_path)
            keys = data["keys"].astype(np.float32)
            case_ids = data["case_ids"]

            # L2 normalize
            norms = np.linalg.norm(keys, axis=1, keepdims=True)
            norms = np.maximum(norms, 1e-8)
            keys = keys / norms

            case_id_to_idx = {int(cid): i for i, cid in enumerate(case_ids)}
            return keys, case_id_to_idx

    raise FileNotFoundError(f"No key vectors found (tried seed {seed} and fallback seed42)")

DEFAULT_ORDERINGS = [
    "key_clustered", "key_dispersed",
    "greedy_minmax", "cluster_topo", "random",
]


def compute_within_batch_cosine(
    keys: np.ndarray,
    case_id_to_idx: dict,
    ordering_case_ids: list,
    batch_size: int = BATCH_SIZE,
) -> float:
    """Compute mean within-batch pairwise cosine across all batches."""
    n_batches = len(ordering_case_ids) // batch_size
    batch_cosines = []

    for b in range(n_batches):
        batch_cids = ordering_case_ids[b * batch_size: (b + 1) * batch_size]
        indices = [case_id_to_idx[cid] for cid in batch_cids if cid in case_id_to_idx]
        if len(indices) < 2:
            continue
        batch_keys = keys[indices]
        cos_matrix = batch_keys @ batch_keys.T
        n = len(indices)
        # Upper triangle (exclude diagonal)
        mask = np.triu(np.ones((n, n), dtype=bool), k=1)
        batch_cosines.append(float(cos_matrix[mask].mean()))

    return float(np.mean(batch_cosines)) if batch_cosines else 0.0


def compute_first_5k_exposure(
    keys: np.ndarray,
    case_id_to_idx: dict,
    ordering_case_ids: list,
) -> dict:
    """Compute exposure metrics for the first 5K edits to all subsequent keys."""
    n = min(len(ordering_case_ids), 10000)

    stream_keys = []
    for pos in range(n):
        cid = ordering_case_ids[pos]
        if cid in case_id_to_idx:
            stream_keys.append(keys[case_id_to_idx[cid]])
        else:
            stream_keys.append(np.zeros(keys.shape[1], dtype=np.float32))

    stream_keys = np.array(stream_keys, dtype=np.float32)

    first_5k_end = min(5000, n)
    first_5k_keys = stream_keys[:first_5k_end]
    subsequent_keys = stream_keys[first_5k_end:]

    if len(subsequent_keys) == 0:
        return {
            "max_cos_mean": float("nan"),
            "frac_above_0.3": float("nan"),
            "frac_above_0.4": float("nan"),
            "n_high_mean": float("nan"),
        }

    cos_matrix = first_5k_keys @ subsequent_keys.T

    max_cos = cos_matrix.max(axis=1)
    n_high = (cos_matrix > HIGH_COS_THRESHOLD).sum(axis=1)

    return {
        "max_cos_mean": float(np.mean(max_cos)),
        "max_cos_median": float(np.median(max_cos)),
        "max_cos_std": float(np.std(max_cos)),
        "frac_above_0.3": float(np.mean(max_cos > 0.3)),
        "frac_above_0.4": float(np.mean(max_cos > 0.4)),
        "n_high_mean": float(np.mean(n_high)),
    }


def compute_ordering_geometry(
    keys: np.ndarray,
    case_id_to_idx: dict,
    ordering_case_ids: list,
) -> dict:
    """Compute full geometry metrics for one ordering.

    Returns dict with first_1k, first_5k, within_batch, and age_bins metrics.
    """
    # First-1K exposure (reuse existing function)
    first_1k = compute_first_1k_exposure(keys, case_id_to_idx, ordering_case_ids)

    # First-5K exposure
    first_5k = compute_first_5k_exposure(keys, case_id_to_idx, ordering_case_ids)

    # Within-batch cosine
    within_batch = compute_within_batch_cosine(keys, case_id_to_idx, ordering_case_ids)

    # Age-binned breakdown
    age_bins = compute_by_age_bin(keys, case_id_to_idx, ordering_case_ids)

    # Format first_1k into standard metrics
    first_1k_metrics = {
        "max_cos_mean": float(np.mean(first_1k["max_cos"])) if len(first_1k["max_cos"]) > 0 else float("nan"),
        "max_cos_median": float(np.median(first_1k["max_cos"])) if len(first_1k["max_cos"]) > 0 else float("nan"),
        "frac_above_0.3": float(np.mean(first_1k["max_cos"] > 0.3)) if len(first_1k["max_cos"]) > 0 else float("nan"),
        "frac_above_0.4": float(np.mean(first_1k["max_cos"] > 0.4)) if len(first_1k["max_cos"]) > 0 else float("nan"),
        "n_high_mean": float(np.mean(first_1k["n_high"])) if len(first_1k["n_high"]) > 0 else float("nan"),
    }

    return {
        "first_1k": first_1k_metrics,
        "first_5k": first_5k,
        "within_batch_cosine": within_batch,
        "age_bins": age_bins,
    }


def check_gate_criterion(results: dict) -> tuple:
    """Check if greedy_minmax passes the gate criterion.

    Gate: greedy_minmax first-1K frac_above_0.3 < key_clustered first-1K frac_above_0.3

    Returns (passed: bool, message: str, details: dict)
    """
    if "greedy_minmax" not in results:
        return False, "greedy_minmax ordering not found in results", {}
    if "key_clustered" not in results:
        return False, "key_clustered ordering not found (needed as reference)", {}

    greedy_frac = results["greedy_minmax"]["first_1k"]["frac_above_0.3"]
    clustered_frac = results["key_clustered"]["first_1k"]["frac_above_0.3"]

    details = {
        "greedy_minmax_frac_0.3": greedy_frac,
        "key_clustered_frac_0.3": clustered_frac,
        "improvement": clustered_frac - greedy_frac,
    }

    if greedy_frac < clustered_frac:
        msg = (f"GATE PASSED: greedy_minmax frac>0.3 = {greedy_frac:.4f} < "
               f"key_clustered = {clustered_frac:.4f} "
               f"(improvement: {clustered_frac - greedy_frac:.4f})")
        return True, msg, details
    else:
        msg = (f"GATE FAILED: greedy_minmax frac>0.3 = {greedy_frac:.4f} >= "
               f"key_clustered = {clustered_frac:.4f}. "
               f"Experiment premise invalid — do not launch GPU runs.")
        return False, msg, details


def check_cluster_topo_discrepancy(results: dict) -> str:
    """Flag if cluster_topo passes but greedy_minmax doesn't."""
    if "cluster_topo" not in results or "key_clustered" not in results:
        return ""

    topo_frac = results["cluster_topo"]["first_1k"]["frac_above_0.3"]
    clustered_frac = results["key_clustered"]["first_1k"]["frac_above_0.3"]
    greedy_frac = results.get("greedy_minmax", {}).get("first_1k", {}).get("frac_above_0.3", 1.0)

    if topo_frac < clustered_frac and greedy_frac >= clustered_frac:
        return (f"DISCREPANCY: cluster_topo passes gate ({topo_frac:.4f} < {clustered_frac:.4f}) "
                f"but greedy_minmax does not ({greedy_frac:.4f}). "
                f"Likely bug in greedy lazy-refresh update.")
    return ""


def generate_report(all_results: dict, seed: int, gate_result: tuple) -> str:
    """Generate markdown report content."""
    passed, gate_msg, gate_details = gate_result

    lines = [
        f"# Geometric Validation Report — Seed {seed}",
        f"",
        f"Generated: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}",
        f"",
        f"## Gate Criterion",
        f"",
        f"**{'PASSED' if passed else 'FAILED'}**: {gate_msg}",
        f"",
        f"## First-1K Cohort Metrics",
        f"",
        f"| Ordering | max_cos_mean | frac > 0.3 | frac > 0.4 | n_high_mean | within_batch_cos |",
        f"|----------|:------------:|:----------:|:----------:|:-----------:|:----------------:|",
    ]

    for name, metrics in sorted(all_results.items()):
        f1k = metrics["first_1k"]
        wb = metrics["within_batch_cosine"]
        lines.append(
            f"| {name:<16} | {f1k['max_cos_mean']:.4f} | "
            f"{f1k['frac_above_0.3']:.4f} | {f1k['frac_above_0.4']:.4f} | "
            f"{f1k['n_high_mean']:.1f} | {wb:.4f} |"
        )

    lines.extend([
        f"",
        f"## First-5K Cohort Metrics",
        f"",
        f"| Ordering | max_cos_mean | frac > 0.3 | frac > 0.4 | n_high_mean |",
        f"|----------|:------------:|:----------:|:----------:|:-----------:|",
    ])

    for name, metrics in sorted(all_results.items()):
        f5k = metrics["first_5k"]
        lines.append(
            f"| {name:<16} | {f5k['max_cos_mean']:.4f} | "
            f"{f5k['frac_above_0.3']:.4f} | {f5k['frac_above_0.4']:.4f} | "
            f"{f5k['n_high_mean']:.1f} |"
        )

    lines.extend([
        f"",
        f"## Age-Binned Max-Cos (first 5 cohorts)",
        f"",
        f"| Ordering | 0-1K | 1K-2K | 2K-3K | 3K-4K | 4K-5K |",
        f"|----------|:----:|:-----:|:-----:|:-----:|:-----:|",
    ])

    for name, metrics in sorted(all_results.items()):
        bins = metrics.get("age_bins", [])
        bin_vals = [f"{b['max_cos_mean']:.3f}" if not np.isnan(b.get("max_cos_mean", float("nan"))) else "—"
                    for b in bins[:5]]
        while len(bin_vals) < 5:
            bin_vals.append("—")
        lines.append(f"| {name:<16} | {' | '.join(bin_vals)} |")

    lines.extend([
        f"",
        f"## Interpretation",
        f"",
    ])

    if passed:
        lines.extend([
            f"The greedy_minmax scheduler successfully reduces first-1K exposure below",
            f"the key_clustered baseline. GPU runs are recommended.",
            f"",
            f"Key observation: within-batch cosine should be SIMILAR across orderings",
            f"(scheduling manipulates cross-batch exposure, not within-batch similarity).",
        ])
    else:
        lines.extend([
            f"The greedy_minmax scheduler FAILED to reduce first-1K exposure below",
            f"key_clustered. This means the experiment's premise fails cheaply.",
            f"Do NOT launch GPU runs until this is resolved.",
        ])

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(
        description="Geometric validation of scheduling orderings"
    )
    parser.add_argument("--seed", type=int, nargs="+", default=[42],
                        help="Seed(s) to validate (default: 42)")
    parser.add_argument("--orderings", type=str, default=None,
                        help="Comma-separated ordering names (default: all available)")
    args = parser.parse_args()

    orderings = args.orderings.split(",") if args.orderings else DEFAULT_ORDERINGS
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    for seed in args.seed:
        print(f"\n{'='*70}")
        print(f"  GEOMETRIC VALIDATION — Seed {seed}")
        print(f"{'='*70}")

        # Load keys
        try:
            keys, case_id_to_idx = load_keys_and_index(seed)
            print(f"  Loaded {len(case_id_to_idx)} keys, dim={keys.shape[1]}")
        except FileNotFoundError as e:
            print(f"  ERROR: {e}")
            continue

        # Compute metrics for each ordering
        all_results = {}
        for ordering_name in orderings:
            ordering_path = ORDERINGS_DIR / f"{ordering_name}_seed{seed}.json"
            if not ordering_path.exists():
                print(f"  SKIP {ordering_name}: file not found at {ordering_path}")
                continue

            print(f"\n  --- {ordering_name} ---")
            ordering_case_ids = load_ordering(seed, ordering_name)
            print(f"    Stream length: {len(ordering_case_ids)}")

            metrics = compute_ordering_geometry(keys, case_id_to_idx, ordering_case_ids)
            all_results[ordering_name] = metrics

            # Print summary
            f1k = metrics["first_1k"]
            print(f"    First-1K: max_cos={f1k['max_cos_mean']:.4f}, "
                  f"frac>0.3={f1k['frac_above_0.3']:.3f}, "
                  f"frac>0.4={f1k['frac_above_0.4']:.3f}, "
                  f"n_high={f1k['n_high_mean']:.1f}")
            print(f"    Within-batch cosine: {metrics['within_batch_cosine']:.4f}")

        # Gate criterion check
        print(f"\n{'─'*70}")
        gate_result = check_gate_criterion(all_results)
        passed, gate_msg, _ = gate_result
        status = "PASSED" if passed else "FAILED"
        print(f"  GATE {status}: {gate_msg}")

        discrepancy = check_cluster_topo_discrepancy(all_results)
        if discrepancy:
            print(f"  WARNING: {discrepancy}")

        # Generate and save report
        report = generate_report(all_results, seed, gate_result)
        report_path = REPORTS_DIR / f"geometry_validation_seed{seed}.md"
        with open(report_path, "w") as f:
            f.write(report)
        print(f"\n  Report saved: {report_path}")

        # Save raw metrics as JSON (for analysis script)
        metrics_path = REPORTS_DIR / f"geometry_metrics_seed{seed}.json"
        with open(metrics_path, "w") as f:
            json.dump(all_results, f, indent=2)
        print(f"  Metrics saved: {metrics_path}")

    print(f"\n{'='*70}")
    print("Validation complete.")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()

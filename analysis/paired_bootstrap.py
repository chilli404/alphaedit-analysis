#!/usr/bin/env python3
"""
Paired bootstrap test for comparing AlphaEdit vs MEMIT.

The key statistical comparison for the reproducibility paper:
- Align case-level results by case_id across both methods
- Average across seeds per (algorithm, case_id) to use all multi-seed data
- Compute per-case differences
- Bootstrap the mean difference to get BCa CI and p-value
- Apply Holm-Bonferroni correction for multiple comparisons
- Report standardized effect sizes (Cohen's d, Cliff's delta)

Usage:
    python analysis/paired_bootstrap.py \
        --results_dir results \
        --method_a AlphaEdit \
        --method_b MEMIT
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats


def _bca_ci(
    data: np.ndarray,
    boot_stats: np.ndarray,
    observed: float,
    confidence: float = 0.95,
) -> tuple[float, float]:
    """
    Bias-corrected and accelerated (BCa) bootstrap confidence interval.

    Provides better coverage than percentile method for skewed distributions,
    which is common with metrics near boundaries (efficacy near 1.0).
    """
    n = len(data)
    alpha = 1 - confidence

    # Bias correction factor (z0)
    prop_below = np.mean(boot_stats < observed)
    if prop_below == 0:
        prop_below = 1 / (2 * len(boot_stats))
    elif prop_below == 1:
        prop_below = 1 - 1 / (2 * len(boot_stats))
    z0 = stats.norm.ppf(prop_below)

    # Acceleration factor (a) via jackknife
    jackknife_stats = np.array([
        np.mean(np.delete(data, i)) for i in range(n)
    ])
    jack_mean = jackknife_stats.mean()
    numerator = np.sum((jack_mean - jackknife_stats) ** 3)
    denominator = 6.0 * (np.sum((jack_mean - jackknife_stats) ** 2) ** 1.5)
    a = numerator / denominator if denominator != 0 else 0.0

    # Adjusted percentiles
    z_alpha_lower = stats.norm.ppf(alpha / 2)
    z_alpha_upper = stats.norm.ppf(1 - alpha / 2)

    # BCa formula for adjusted quantiles
    denom_lower = 1 - a * (z0 + z_alpha_lower)
    denom_upper = 1 - a * (z0 + z_alpha_upper)

    if denom_lower == 0 or denom_upper == 0:
        # Fall back to percentile method if BCa breaks
        ci_lower = np.percentile(boot_stats, 100 * alpha / 2)
        ci_upper = np.percentile(boot_stats, 100 * (1 - alpha / 2))
        return ci_lower, ci_upper

    alpha_lower = stats.norm.cdf(z0 + (z0 + z_alpha_lower) / denom_lower)
    alpha_upper = stats.norm.cdf(z0 + (z0 + z_alpha_upper) / denom_upper)

    # Clamp to valid percentile range
    alpha_lower = np.clip(alpha_lower, 0.001, 0.999)
    alpha_upper = np.clip(alpha_upper, 0.001, 0.999)

    ci_lower = np.percentile(boot_stats, 100 * alpha_lower)
    ci_upper = np.percentile(boot_stats, 100 * alpha_upper)

    return ci_lower, ci_upper


def cohens_d(scores_a: np.ndarray, scores_b: np.ndarray) -> float:
    """
    Cohen's d for paired observations (using pooled SD).

    Provides a standardized effect size measure interpretable across
    metrics with different scales.
    """
    diffs = scores_a - scores_b
    mean_diff = diffs.mean()

    # Pooled standard deviation
    sd_a = np.std(scores_a, ddof=1)
    sd_b = np.std(scores_b, ddof=1)
    pooled_sd = np.sqrt((sd_a**2 + sd_b**2) / 2)

    if pooled_sd == 0:
        return 0.0
    return mean_diff / pooled_sd


def cliffs_delta(scores_a: np.ndarray, scores_b: np.ndarray) -> float:
    """
    Cliff's delta: non-parametric effect size for ordinal data.

    Returns a value in [-1, 1]:
      +1 = all values in A are greater than all values in B
       0 = distributions are identical
      -1 = all values in B are greater than all values in A
    """
    n_a, n_b = len(scores_a), len(scores_b)
    if n_a == 0 or n_b == 0:
        return 0.0

    # Count dominance
    count = 0
    for a in scores_a:
        for b in scores_b:
            if a > b:
                count += 1
            elif a < b:
                count -= 1

    return count / (n_a * n_b)


def paired_bootstrap_test(
    scores_a: np.ndarray,
    scores_b: np.ndarray,
    n_bootstrap: int = 10000,
    confidence: float = 0.95,
    seed: int = 42,
) -> dict:
    """
    Paired bootstrap test with BCa CI and effect sizes.

    Args:
        scores_a: Scores for method A (e.g., AlphaEdit)
        scores_b: Scores for method B (e.g., MEMIT)
        n_bootstrap: Number of bootstrap resamples
        confidence: Confidence level for CI
        seed: Random seed

    Returns:
        Dict with: mean_diff, ci_lower, ci_upper, p_value, n_pairs,
                   cohens_d, cliffs_delta
    """
    scores_a = np.asarray(scores_a, dtype=float)
    scores_b = np.asarray(scores_b, dtype=float)

    assert len(scores_a) == len(scores_b), "Arrays must be same length (paired)"

    # Remove pairs where either is NaN
    mask = ~(np.isnan(scores_a) | np.isnan(scores_b))
    scores_a = scores_a[mask]
    scores_b = scores_b[mask]
    n = len(scores_a)

    if n == 0:
        return {
            "mean_diff": np.nan,
            "ci_lower": np.nan,
            "ci_upper": np.nan,
            "p_value": np.nan,
            "n_pairs": 0,
            "cohens_d": np.nan,
            "cliffs_delta": np.nan,
        }

    diffs = scores_a - scores_b
    mean_diff = diffs.mean()

    # Bootstrap the mean difference
    rng = np.random.default_rng(seed)
    boot_means = np.array([
        rng.choice(diffs, size=n, replace=True).mean()
        for _ in range(n_bootstrap)
    ])

    # BCa confidence interval
    ci_lower, ci_upper = _bca_ci(diffs, boot_means, mean_diff, confidence)

    # Two-sided p-value: proportion of bootstrap samples on the other side of 0
    if mean_diff > 0:
        p_value = 2 * np.mean(boot_means <= 0)
    elif mean_diff < 0:
        p_value = 2 * np.mean(boot_means >= 0)
    else:
        p_value = 1.0

    p_value = min(p_value, 1.0)

    # Effect sizes
    d = cohens_d(scores_a, scores_b)
    delta = cliffs_delta(scores_a, scores_b)

    return {
        "mean_diff": mean_diff,
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
        "p_value": p_value,
        "n_pairs": n,
        "cohens_d": d,
        "cliffs_delta": delta,
    }


def holm_bonferroni(p_values: list[float]) -> list[dict]:
    """
    Holm-Bonferroni step-down correction for multiple comparisons.

    More powerful than Bonferroni while still controlling family-wise error rate.

    Returns a list of dicts with original and adjusted p-values + significance.
    """
    k = len(p_values)
    indexed = sorted(enumerate(p_values), key=lambda x: x[1])

    adjusted = [None] * k
    max_so_far = 0.0

    for rank, (orig_idx, p) in enumerate(indexed):
        correction = k - rank  # Holm step-down multiplier
        adj_p = min(p * correction, 1.0)
        # Ensure monotonicity (adjusted p-values must be non-decreasing)
        max_so_far = max(max_so_far, adj_p)
        adjusted[orig_idx] = max_so_far

    results = []
    for i, (p_raw, p_adj) in enumerate(zip(p_values, adjusted)):
        results.append({
            "p_raw": p_raw,
            "p_adjusted": p_adj,
            "significant_raw": p_raw < 0.05,
            "significant_adjusted": p_adj < 0.05,
        })

    return results


def compare_methods(
    per_case_csv: Path,
    method_a: str = "AlphaEdit",
    method_b: str = "MEMIT",
    metrics: list | None = None,
    n_bootstrap: int = 10000,
) -> pd.DataFrame:
    """
    Run paired bootstrap comparison across all metrics.

    Correctly handles multi-seed data by averaging across seeds per case_id
    before pairing, rather than discarding duplicate seeds.

    Args:
        per_case_csv: Path to per_case_results.csv from aggregate.py
        method_a: First method name
        method_b: Second method name
        metrics: List of metric columns to compare (default: all available)
        n_bootstrap: Number of bootstrap resamples

    Returns:
        DataFrame with one row per metric, columns for diff/CI/p-value/effect sizes
    """
    df = pd.read_csv(per_case_csv)

    if metrics is None:
        metrics = ["efficacy", "generalization", "specificity", "fluency", "consistency"]

    df_a = df[df["algorithm"] == method_a]
    df_b = df[df["algorithm"] == method_b]

    if df_a.empty or df_b.empty:
        print(f"WARNING: Missing data for {method_a} or {method_b}")
        return pd.DataFrame()

    # Average across seeds per (algorithm, case_id) to use ALL multi-seed data.
    # This is the fix for the bug where drop_duplicates discarded multi-seed info.
    available_metrics = [m for m in metrics if m in df.columns]
    agg_cols = {m: "mean" for m in available_metrics}

    df_a_agg = (
        df_a.groupby("case_id")[available_metrics]
        .mean()
        .reset_index()
    )
    df_b_agg = (
        df_b.groupby("case_id")[available_metrics]
        .mean()
        .reset_index()
    )

    # Merge on case_id
    merged = df_a_agg.merge(df_b_agg, on="case_id", suffixes=("_a", "_b"))

    if merged.empty:
        print("WARNING: No overlapping case_ids between methods")
        return pd.DataFrame()

    results = []
    for metric in available_metrics:
        col_a = f"{metric}_a"
        col_b = f"{metric}_b"
        if col_a not in merged.columns or col_b not in merged.columns:
            continue

        test_result = paired_bootstrap_test(
            merged[col_a].values,
            merged[col_b].values,
            n_bootstrap=n_bootstrap,
        )
        test_result["metric"] = metric
        test_result["method_a"] = method_a
        test_result["method_b"] = method_b
        results.append(test_result)

    if not results:
        return pd.DataFrame()

    results_df = pd.DataFrame(results)

    # Apply Holm-Bonferroni correction for multiple comparisons
    p_values = results_df["p_value"].tolist()
    corrections = holm_bonferroni(p_values)
    results_df["p_adjusted"] = [c["p_adjusted"] for c in corrections]
    results_df["significant_raw"] = [c["significant_raw"] for c in corrections]
    results_df["significant_adjusted"] = [c["significant_adjusted"] for c in corrections]

    return results_df


def main():
    parser = argparse.ArgumentParser(
        description="Paired bootstrap comparison of AlphaEdit vs MEMIT"
    )
    parser.add_argument(
        "--results_dir",
        type=Path,
        default=Path("results"),
        help="Directory containing per_case_results.csv",
    )
    parser.add_argument("--method_a", default="AlphaEdit")
    parser.add_argument("--method_b", default="MEMIT")
    parser.add_argument("--n_bootstrap", type=int, default=10000)
    args = parser.parse_args()

    per_case_csv = args.results_dir / "per_case_results.csv"
    if not per_case_csv.exists():
        print(f"ERROR: {per_case_csv} not found. Run aggregate.py first.")
        return

    print(f"=== Paired Bootstrap: {args.method_a} vs {args.method_b} ===")
    print(f"  Bootstrap resamples: {args.n_bootstrap}")
    print(f"  CI method: BCa (bias-corrected & accelerated)")
    print(f"  Multiple comparison correction: Holm-Bonferroni")
    print()

    results = compare_methods(
        per_case_csv, args.method_a, args.method_b,
        n_bootstrap=args.n_bootstrap,
    )

    if results.empty:
        print("No comparison results generated.")
        return

    # Display results
    header = (
        f"{'Metric':<15} {'Δ (A-B)':<9} {'95% BCa CI':<25} "
        f"{'p-val':<8} {'p-adj':<8} {'Cohen d':<9} {'Cliff δ':<9} {'Sig?':<5} {'n':<6}"
    )
    print(header)
    print("-" * len(header))
    for _, row in results.iterrows():
        ci_str = f"[{row['ci_lower']:.4f}, {row['ci_upper']:.4f}]"
        sig = "*" if row["significant_adjusted"] else ""
        print(
            f"{row['metric']:<15} "
            f"{row['mean_diff']:>+.4f}  "
            f"{ci_str:<25} "
            f"{row['p_value']:.4f}  "
            f"{row['p_adjusted']:.4f}  "
            f"{row['cohens_d']:>+.3f}   "
            f"{row['cliffs_delta']:>+.3f}   "
            f"{sig:<5} "
            f"{row['n_pairs']}"
        )

    # Save
    output_path = args.results_dir / "paired_bootstrap_results.csv"
    results.to_csv(output_path, index=False)
    print(f"\nSaved to: {output_path}")

    # Summary
    n_sig = results["significant_adjusted"].sum()
    print(f"\n{n_sig}/{len(results)} metrics significant after Holm-Bonferroni correction (α=0.05)")


if __name__ == "__main__":
    main()

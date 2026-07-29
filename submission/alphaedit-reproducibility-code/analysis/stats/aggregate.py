#!/usr/bin/env python3
"""
Aggregate case-level JSON results into CSV summary tables.

Reads the per-case JSON outputs from AlphaEdit/MEMIT runs and produces:
1. A per-case CSV with all metrics extracted
2. A summary CSV with means and standard deviations across seeds
3. Optionally: confidence intervals (Wilson for binary, BCa bootstrap for continuous)

Averaging specification:
  - Within-case (micro): efficacy_case_i = mean(rewrite_prompts_correct[i])
  - Across-cases (macro): efficacy_overall = mean(efficacy_case_i for all i)
  This is macro-averaging: each case contributes equally regardless of
  how many prompts it has.

Usage:
    python analysis/stats/aggregate.py --results_dir vendor/AlphaEdit/results
    python analysis/stats/aggregate.py --results_dir vendor/AlphaEdit/results --alg AlphaEdit --with_ci
    python analysis/stats/aggregate.py --report_averaging
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


# Metrics to extract from the "post" field of each case JSON
METRICS = {
    "rewrite_prompts_correct": "efficacy",
    "paraphrase_prompts_correct": "generalization",
    "neighborhood_prompts_correct": "specificity",
    "reference_score": "fluency",
    "essence_score": "consistency",
    "rewrite_prompts_probs": "efficacy_prob",
    "paraphrase_prompts_probs": "generalization_prob",
    "neighborhood_prompts_probs": "specificity_prob",
}

# Averaging specification — documents how each metric is aggregated
AVERAGING_SPEC = {
    "efficacy": {
        "within_case": "micro (per-prompt mean)",
        "across_cases": "macro (mean of case means)",
        "type": "binary",
    },
    "generalization": {
        "within_case": "micro (per-prompt mean)",
        "across_cases": "macro (mean of case means)",
        "type": "binary",
    },
    "specificity": {
        "within_case": "micro (per-prompt mean)",
        "across_cases": "macro (mean of case means)",
        "type": "binary",
    },
    "fluency": {
        "within_case": "n/a (scalar per case)",
        "across_cases": "macro (mean of case values)",
        "type": "continuous",
    },
    "consistency": {
        "within_case": "n/a (scalar per case)",
        "across_cases": "macro (mean of case values)",
        "type": "continuous",
    },
    "efficacy_prob": {
        "within_case": "micro (per-prompt mean of P(target_new))",
        "across_cases": "macro (mean of case means)",
        "type": "continuous",
    },
    "generalization_prob": {
        "within_case": "micro (per-prompt mean of P(target_new))",
        "across_cases": "macro (mean of case means)",
        "type": "continuous",
    },
    "specificity_prob": {
        "within_case": "micro (per-prompt mean of P(target_new))",
        "across_cases": "macro (mean of case means)",
        "type": "continuous",
    },
}


def extract_metrics_from_case(case_json: dict) -> dict:
    """Extract standardized metrics from a single case JSON file."""
    post = case_json.get("post", {})

    row = {
        "case_id": case_json["case_id"],
        "num_edits": case_json.get("num_edits"),
        "time": case_json.get("time"),
    }

    # Extract requested_rewrite info
    rw = case_json.get("requested_rewrite", {})
    if isinstance(rw, list):
        rw = rw[0] if rw else {}
    row["subject"] = rw.get("subject", "")
    row["target_new"] = rw.get("target_new", {}).get("str", "")

    # Extract post-edit metrics
    for json_key, metric_name in METRICS.items():
        value = post.get(json_key)
        if isinstance(value, list) and value:
            if isinstance(value[0], dict):
                # Prob fields: list of {"target_new": float, "target_true": float}
                row[metric_name] = sum(d["target_new"] for d in value) / len(value)
            elif isinstance(value[0], bool):
                # Correct fields: list of booleans
                row[metric_name] = sum(value) / len(value)
            else:
                row[metric_name] = sum(value) / len(value)
        elif isinstance(value, list):
            row[metric_name] = None
        else:
            row[metric_name] = value

    return row


def collect_run_results(run_dir: Path) -> pd.DataFrame:
    """Collect all case JSONs from a single run directory."""
    rows = []
    for case_file in sorted(run_dir.glob("*_edits-case_*.json")):
        with open(case_file, "r") as f:
            case_data = json.load(f)
        rows.append(extract_metrics_from_case(case_data))

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    df["run_id"] = run_dir.name
    return df


def _compute_ci_row(values: np.ndarray, metric_type: str) -> dict:
    """Compute CI for a single metric column."""
    from confidence_intervals import compute_metric_ci
    return compute_metric_ci(values, metric_type=metric_type)


def aggregate_results(
    results_dir: Path,
    alg_name: str | None = None,
    output_dir: Path | None = None,
    with_ci: bool = False,
) -> pd.DataFrame:
    """
    Aggregate results across all runs for a given algorithm.

    Returns a DataFrame with one row per (case_id, run_id).
    """
    if output_dir is None:
        output_dir = Path("results")
    output_dir.mkdir(parents=True, exist_ok=True)

    all_dfs = []

    # Find all algorithm directories
    alg_dirs = [results_dir / alg_name] if alg_name else list(results_dir.iterdir())

    for alg_dir in alg_dirs:
        if not alg_dir.is_dir():
            continue
        alg = alg_dir.name

        for run_dir in sorted(alg_dir.iterdir()):
            if not run_dir.is_dir():
                continue

            df = collect_run_results(run_dir)
            if df.empty:
                continue

            df["algorithm"] = alg
            all_dfs.append(df)

    if not all_dfs:
        print("No results found.")
        return pd.DataFrame()

    combined = pd.concat(all_dfs, ignore_index=True)

    # Save per-case CSV
    per_case_path = output_dir / "per_case_results.csv"
    combined.to_csv(per_case_path, index=False)
    print(f"Per-case results: {per_case_path} ({len(combined)} rows)")

    # Create summary statistics
    metric_cols = list(METRICS.values())
    available_metrics = [m for m in metric_cols if m in combined.columns]

    summary = (
        combined.groupby(["algorithm", "run_id"])[available_metrics]
        .agg(["mean", "std", "count"])
    )
    summary_path = output_dir / "summary_results.csv"
    summary.to_csv(summary_path)
    print(f"Summary results: {summary_path}")

    # Grand summary (across all runs)
    grand_summary = (
        combined.groupby("algorithm")[available_metrics]
        .agg(["mean", "std", "count"])
    )
    grand_path = output_dir / "grand_summary.csv"
    grand_summary.to_csv(grand_path)
    print(f"Grand summary: {grand_path}")

    # Confidence intervals
    if with_ci:
        ci_rows = []
        for alg in combined["algorithm"].unique():
            alg_data = combined[combined["algorithm"] == alg]
            for metric in available_metrics:
                values = alg_data[metric].dropna().values
                metric_type = AVERAGING_SPEC.get(metric, {}).get("type", "continuous")
                ci_result = _compute_ci_row(np.array(values), metric_type)
                ci_method = "wilson" if metric_type == "binary" else "bca_bootstrap"
                ci_rows.append({
                    "algorithm": alg,
                    "metric": metric,
                    "metric_mean": ci_result["mean"],
                    "metric_ci_lower": ci_result["ci_lower"],
                    "metric_ci_upper": ci_result["ci_upper"],
                    "metric_ci_method": ci_method,
                    "metric_std": ci_result["std"],
                    "n": ci_result["n"],
                    "averaging_mode": AVERAGING_SPEC.get(metric, {}).get("across_cases", "macro"),
                })

        ci_df = pd.DataFrame(ci_rows)
        ci_path = output_dir / "summary_with_ci.csv"
        ci_df.to_csv(ci_path, index=False)
        print(f"Summary with CIs: {ci_path}")

    return combined


def report_averaging() -> None:
    """Print the averaging specification for all metrics."""
    print("=" * 70)
    print("Metric Averaging Specification")
    print("=" * 70)
    print(f"\n{'Metric':<20} {'Within-Case':<35} {'Across-Cases':<30} {'Type':<12}")
    print("-" * 97)
    for metric, spec in AVERAGING_SPEC.items():
        print(
            f"{metric:<20} {spec['within_case']:<35} "
            f"{spec['across_cases']:<30} {spec['type']:<12}"
        )
    print("\nKey distinction:")
    print("  - Micro (within case): efficacy_case_i = mean(rewrite_prompts_correct[i])")
    print("  - Macro (across cases): efficacy_overall = mean(efficacy_case_i for all i)")
    print("  Each case contributes equally regardless of prompt count.")
    print("=" * 70)


def main():
    parser = argparse.ArgumentParser(description="Aggregate AlphaEdit results")
    parser.add_argument(
        "--results_dir",
        type=Path,
        default=Path("vendor/AlphaEdit/results"),
        help="Root results directory",
    )
    parser.add_argument(
        "--alg",
        type=str,
        default=None,
        help="Filter to specific algorithm (AlphaEdit, MEMIT)",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path("results"),
        help="Output directory for CSV files",
    )
    parser.add_argument(
        "--with_ci",
        action="store_true",
        help="Compute confidence intervals (Wilson for binary, BCa bootstrap for continuous)",
    )
    parser.add_argument(
        "--report_averaging",
        action="store_true",
        help="Print the averaging specification and exit",
    )
    args = parser.parse_args()

    if args.report_averaging:
        report_averaging()
        return

    aggregate_results(args.results_dir, args.alg, args.output_dir, with_ci=args.with_ci)


if __name__ == "__main__":
    main()

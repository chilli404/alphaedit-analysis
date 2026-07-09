#!/usr/bin/env python3
"""
Aggregate case-level JSON results into CSV summary tables.

Reads the per-case JSON outputs from AlphaEdit/MEMIT runs and produces:
1. A per-case CSV with all metrics extracted
2. A summary CSV with means and standard deviations across seeds

Usage:
    python analysis/aggregate.py --results_dir vendor/AlphaEdit/results
    python analysis/aggregate.py --results_dir vendor/AlphaEdit/results --alg AlphaEdit --ds mcf
"""

import argparse
import json
from pathlib import Path

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
        if isinstance(value, list):
            # Average across prompts if it's a list
            row[metric_name] = sum(value) / len(value) if value else None
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


def aggregate_results(
    results_dir: Path,
    alg_name: str | None = None,
    output_dir: Path | None = None,
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

    return combined


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
    args = parser.parse_args()

    aggregate_results(args.results_dir, args.alg, args.output_dir)


if __name__ == "__main__":
    main()

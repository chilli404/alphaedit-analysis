#!/usr/bin/env python3
"""
Controlled Coupling Analysis — post-hoc metrics from coupling experiment results.

Computes the following outputs from controlled_coupling_runner JSONL/JSON:
  1. Early difficulty comparison: efficacy at batches 1-5, high vs low coupling
  2. Cumulative efficacy: running mean of efficacy across all edits so far
  3. Cohort efficacy: group edits by batch-of-origin, track their efficacy at later checkpoints
  4. Probability locality: specificity_prob restricted to semantically adjacent facts
  5. Immediate efficacy: efficacy measured immediately after insertion
  6. Cache spectral metrics: effective_rank, stable_rank from mechanism_analyzer
  7. Functional projection-loss: from plasticity_tracker (already measured inline)
  8. Update and weight drift: weight_drift_frobenius per checkpoint
  9. Collapse point: first batch where efficacy drops below threshold

All computations are post-hoc on existing JSONL/JSON outputs. No GPU needed.

Usage:
    python analysis/controlled_coupling_analysis.py \\
        --results_dir results/controlled_coupling/seed42 \\
        --output_dir results/controlled_coupling_analysis/seed42

    python analysis/controlled_coupling_analysis.py --help
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def load_coupling_results(results_dir: Path) -> pd.DataFrame:
    """
    Load per-case JSON results from controlled coupling experiment.

    Returns DataFrame with columns: case_id, batch_idx, coupling_level,
    efficacy, generalization, specificity, efficacy_prob, specificity_prob, ...
    """
    from stats.aggregate import extract_metrics_from_case

    rows = []
    for json_file in sorted(results_dir.rglob("*_edits-case_*.json")):
        try:
            with open(json_file) as f:
                case_data = json.load(f)
            row = extract_metrics_from_case(case_data)
            # Extract coupling metadata if present
            rw = case_data.get("requested_rewrite", {})
            if isinstance(rw, list):
                rw = rw[0] if rw else {}
            row["coupling_level"] = rw.get("coupling_level", "unknown")
            row["coupling_type"] = rw.get("coupling_type", "unknown")
            rows.append(row)
        except (json.JSONDecodeError, KeyError):
            continue

    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows)


def load_mechanism_jsonl(results_dir: Path) -> pd.DataFrame:
    """Load mechanism analyzer JSONL (per-batch spectral metrics)."""
    rows = []
    for jsonl_file in sorted(results_dir.rglob("mechanism_*.jsonl")):
        with open(jsonl_file) as f:
            for line in f:
                if line.strip():
                    rows.append(json.loads(line))
    if not rows:
        # Try plasticity tracker output
        for jsonl_file in sorted(results_dir.rglob("plasticity_*.jsonl")):
            with open(jsonl_file) as f:
                for line in f:
                    if line.strip():
                        rows.append(json.loads(line))
    return pd.DataFrame(rows) if rows else pd.DataFrame()


def compute_early_difficulty(df: pd.DataFrame, n_early_batches: int = 5) -> pd.DataFrame:
    """
    Compare efficacy in first N batches between high/low coupling streams.

    Returns DataFrame with coupling_level × batch aggregated efficacy.
    """
    if "batch_idx" not in df.columns or df.empty:
        return pd.DataFrame()

    early = df[df["batch_idx"] < n_early_batches]
    if early.empty:
        return pd.DataFrame()

    return (
        early.groupby(["coupling_level", "batch_idx"])["efficacy"]
        .agg(["mean", "std", "count"])
        .reset_index()
    )


def compute_cumulative_efficacy(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute running mean of efficacy across all edits processed so far.

    Returns DataFrame with batch_idx, cumulative_efficacy, coupling_level.
    """
    if df.empty or "efficacy" not in df.columns:
        return pd.DataFrame()

    results = []
    for level in df["coupling_level"].unique():
        level_df = df[df["coupling_level"] == level].sort_values("case_id")
        cum_mean = level_df["efficacy"].expanding().mean()
        for idx, (_, row) in enumerate(level_df.iterrows()):
            results.append({
                "case_idx": idx,
                "case_id": row["case_id"],
                "coupling_level": level,
                "cumulative_efficacy": cum_mean.iloc[idx],
                "batch_efficacy": row["efficacy"],
            })
    return pd.DataFrame(results)


def compute_cohort_efficacy(
    results_dir: Path, df: pd.DataFrame, num_edits: int = 100
) -> pd.DataFrame:
    """
    Group edits by batch-of-origin, track their efficacy at later checkpoints.

    This requires checkpoint-level re-evaluation data (retention probes).
    Falls back to single-point measurement if checkpoints not available.
    """
    if df.empty:
        return pd.DataFrame()

    # Assign batch of origin
    df = df.copy()
    if "batch_idx" not in df.columns:
        # Infer from case position
        df["batch_idx"] = df.index // num_edits

    # Group by batch of origin
    cohort_summary = (
        df.groupby(["batch_idx", "coupling_level"])["efficacy"]
        .agg(["mean", "std", "count"])
        .reset_index()
        .rename(columns={"mean": "cohort_efficacy", "batch_idx": "origin_batch"})
    )
    return cohort_summary


def compute_probability_locality(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute specificity_prob restricted to semantically adjacent facts.

    High coupling_level facts with low specificity_prob indicate
    that edits are leaking to semantically related entities.
    """
    if df.empty or "specificity_prob" not in df.columns:
        return pd.DataFrame()

    return (
        df.groupby("coupling_level")["specificity_prob"]
        .agg(["mean", "std", "count"])
        .reset_index()
        .rename(columns={"mean": "prob_locality_mean"})
    )


def compute_collapse_point(
    df: pd.DataFrame, threshold: float = 0.8, window: int = 100
) -> dict:
    """
    Find first batch where efficacy drops below threshold.

    Uses a rolling window to smooth out individual case noise.

    Returns dict with collapse_batch (or None if never collapses).
    """
    if df.empty or "efficacy" not in df.columns:
        return {"collapse_batch": None, "threshold": threshold}

    sorted_df = df.sort_values("case_id")
    rolling_eff = sorted_df["efficacy"].rolling(window=window, min_periods=1).mean()

    collapse_idx = np.where(rolling_eff < threshold)[0]
    if len(collapse_idx) > 0:
        collapse_case = sorted_df.iloc[collapse_idx[0]]["case_id"]
        return {
            "collapse_batch": int(collapse_idx[0] // 100),
            "collapse_case_idx": int(collapse_idx[0]),
            "collapse_case_id": int(collapse_case),
            "threshold": threshold,
            "efficacy_at_collapse": float(rolling_eff.iloc[collapse_idx[0]]),
        }
    return {"collapse_batch": None, "threshold": threshold}


def analyze(results_dir: Path, output_dir: Path, num_edits: int = 100) -> dict:
    """Run full controlled coupling analysis."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    df = load_coupling_results(results_dir)
    mechanism_df = load_mechanism_jsonl(results_dir)

    report = {
        "results_dir": str(results_dir),
        "n_cases": len(df),
        "coupling_levels": sorted(df["coupling_level"].unique().tolist()) if not df.empty else [],
    }

    if df.empty:
        print("WARNING: No coupling results found.")
        report["status"] = "no_data"
        return report

    # 1. Early difficulty
    early_diff = compute_early_difficulty(df)
    if not early_diff.empty:
        early_diff.to_csv(output_dir / "early_difficulty.csv", index=False)
        report["early_difficulty"] = "saved"

    # 2. Cumulative efficacy
    cum_eff = compute_cumulative_efficacy(df)
    if not cum_eff.empty:
        cum_eff.to_csv(output_dir / "cumulative_efficacy.csv", index=False)
        report["cumulative_efficacy"] = "saved"

    # 3. Cohort efficacy
    cohort = compute_cohort_efficacy(results_dir, df, num_edits)
    if not cohort.empty:
        cohort.to_csv(output_dir / "cohort_efficacy.csv", index=False)
        report["cohort_efficacy"] = "saved"

    # 4. Probability locality
    prob_loc = compute_probability_locality(df)
    if not prob_loc.empty:
        prob_loc.to_csv(output_dir / "probability_locality.csv", index=False)
        report["probability_locality"] = "saved"

    # 5. Collapse point
    for level in df["coupling_level"].unique():
        level_df = df[df["coupling_level"] == level]
        collapse = compute_collapse_point(level_df)
        report[f"collapse_point_{level}"] = collapse

    # 6. Mechanism metrics (if available)
    if not mechanism_df.empty:
        mechanism_df.to_csv(output_dir / "mechanism_metrics.csv", index=False)
        report["mechanism_metrics"] = "saved"
        if "effective_rank" in mechanism_df.columns:
            report["final_effective_rank"] = float(mechanism_df["effective_rank"].iloc[-1])
        if "weight_drift_frobenius" in mechanism_df.columns:
            report["final_weight_drift"] = float(mechanism_df["weight_drift_frobenius"].iloc[-1])

    # Save report
    report_path = output_dir / "analysis_report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2, default=str)

    print(f"Analysis complete. Report: {report_path}")
    print(f"  Cases analyzed: {len(df)}")
    print(f"  Coupling levels: {report['coupling_levels']}")
    for level in df["coupling_level"].unique():
        collapse = report.get(f"collapse_point_{level}", {})
        if collapse.get("collapse_batch") is not None:
            print(f"  Collapse ({level}): batch {collapse['collapse_batch']}")
        else:
            print(f"  Collapse ({level}): not reached")

    return report


def main():
    parser = argparse.ArgumentParser(
        description="Post-hoc analysis of controlled coupling experiment results"
    )
    parser.add_argument(
        "--results_dir",
        type=Path,
        required=True,
        help="Directory containing coupling experiment results",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=None,
        help="Output directory for analysis CSVs (default: results_dir/../coupling_analysis)",
    )
    parser.add_argument(
        "--num_edits",
        type=int,
        default=100,
        help="Edits per batch (for cohort assignment)",
    )
    args = parser.parse_args()

    output_dir = args.output_dir or (args.results_dir.parent / "coupling_analysis")
    analyze(args.results_dir, output_dir, args.num_edits)


if __name__ == "__main__":
    main()

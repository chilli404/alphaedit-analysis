#!/usr/bin/env python3
"""
Failure curve analysis for AlphaEdit reproducibility study.

Analyzes checkpointed failure curve results to show how AlphaEdit and MEMIT
degrade as total sequential edits increase from 3000 to 10000.

Key outputs:
1. failure_curve.pdf — Main figure: metrics vs edit count (AlphaEdit vs MEMIT)
2. cohort_retention.pdf — How early-edited facts degrade at later checkpoints
3. glue_degradation.pdf — GLUE benchmark preservation vs edit count
4. failure_curve_summary.csv — Tabular summary of all metrics

Usage:
    python analysis/failure_curve.py
    python analysis/failure_curve.py --results_dir results/failure_curve_checkpointed
    python analysis/failure_curve.py --output_dir results/figures/failure_curve
"""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from analysis.stats.aggregate import extract_metrics_from_case
from analysis.stats.confidence_intervals import bootstrap_ci, wilson_interval

# Paper-quality matplotlib settings
plt.rcParams.update({
    "font.size": 10,
    "axes.labelsize": 11,
    "axes.titlesize": 12,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "font.family": "serif",
})

COLORS = {"AlphaEdit": "#2196F3", "MEMIT": "#FF9800"}
SEEDS = [42, 2024]
BATCH_SIZE = 100  # edits per batch


def find_result_dirs(base_dir: Path) -> list[dict]:
    """
    Scan the checkpoint results directory and find all available data points.

    Returns list of dicts with keys:
        seed, algorithm, total_edits, case_dir, glue_dir
    """
    entries = []

    for seed_dir in sorted(base_dir.iterdir()):
        if not seed_dir.is_dir() or not seed_dir.name.startswith("seed"):
            continue
        seed = int(seed_dir.name.replace("seed", ""))

        for edits_dir in sorted(seed_dir.iterdir()):
            if not edits_dir.is_dir() or not edits_dir.name.endswith("edits"):
                continue
            total_edits = int(edits_dir.name.replace("edits", ""))

            # Find algorithm result directories
            # Pattern: alphaedit_results/AlphaEdit/run_000/ or alphaedit_results_MEMIT/MEMIT/run_000/
            for subdir in edits_dir.iterdir():
                if not subdir.is_dir():
                    continue

                # Look for algorithm directories within
                for alg_dir in subdir.iterdir():
                    if not alg_dir.is_dir():
                        continue
                    if alg_dir.name not in ("AlphaEdit", "MEMIT"):
                        continue

                    # Find run directory
                    for run_dir in sorted(alg_dir.iterdir()):
                        if not run_dir.is_dir() or not run_dir.name.startswith("run_"):
                            continue

                        # Check if it has case files
                        case_files = list(run_dir.glob("*_edits-case_*.json"))
                        if not case_files:
                            continue

                        glue_dir = run_dir / "glue_eval"

                        entries.append({
                            "seed": seed,
                            "algorithm": alg_dir.name,
                            "total_edits": total_edits,
                            "case_dir": run_dir,
                            "glue_dir": glue_dir if glue_dir.exists() else None,
                            "n_cases": len(case_files),
                        })

    return entries


def load_cases(case_dir: Path) -> pd.DataFrame:
    """Load all per-case JSONs from a run directory into a DataFrame."""
    rows = []
    for case_file in case_dir.glob("*_edits-case_*.json"):
        with open(case_file) as f:
            case_data = json.load(f)
        rows.append(extract_metrics_from_case(case_data))

    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows)


def load_glue(glue_dir: Path) -> dict | None:
    """Load GLUE evaluation results. Returns dict with task scores."""
    if glue_dir is None or not glue_dir.exists():
        return None

    # Look for the aggregate glue file (not the per-task gen files)
    # Try edit_glue.json first (post-edit), then base_glue.json
    for name in ["edit_glue.json", "base_glue.json"]:
        glue_file = glue_dir / name
        if glue_file.exists():
            with open(glue_file) as f:
                data = json.load(f)
            return data

    # Try to find any numbered GLUE file (e.g., 50_glue.json for batch 50)
    glue_files = sorted(glue_dir.glob("*_glue.json"))
    # Prefer the highest-numbered one (most edits applied)
    for gf in reversed(glue_files):
        if gf.name.startswith("base_"):
            continue
        with open(gf) as f:
            return json.load(f)

    # Fallback: base_glue.json
    base = glue_dir / "base_glue.json"
    if base.exists():
        with open(base) as f:
            return json.load(f)

    return None


def aggregate_failure_curve(entries: list[dict]) -> pd.DataFrame:
    """
    Aggregate per-case metrics for each (seed, algorithm, total_edits) combination.

    Returns DataFrame with columns:
        seed, algorithm, total_edits, efficacy, generalization, specificity,
        efficacy_ci_lo, efficacy_ci_hi, ...
    """
    rows = []

    for entry in entries:
        print(f"  Loading seed={entry['seed']} alg={entry['algorithm']} "
              f"edits={entry['total_edits']} ({entry['n_cases']} cases)...")

        df = load_cases(entry["case_dir"])
        if df.empty:
            continue

        row = {
            "seed": entry["seed"],
            "algorithm": entry["algorithm"],
            "total_edits": entry["total_edits"],
            "n_cases": len(df),
        }

        # Compute aggregate metrics
        for metric in ["efficacy", "generalization", "specificity"]:
            if metric not in df.columns:
                continue
            values = df[metric].dropna().values
            if len(values) == 0:
                continue

            n = len(values)
            successes = int(np.sum(values >= 0.5))
            row[metric] = successes / n
            ci_lo, ci_hi = wilson_interval(successes, n)
            row[f"{metric}_ci_lo"] = ci_lo
            row[f"{metric}_ci_hi"] = ci_hi
            row[f"{metric}_n"] = n

        # Load GLUE if available
        glue = load_glue(entry["glue_dir"])
        if glue:
            for task in ["sst", "mmmlu", "mrpc", "cola", "nli"]:
                task_data = glue.get(task, {})
                if isinstance(task_data, dict) and "f1" in task_data:
                    row[f"glue_{task}_f1"] = task_data["f1"]

        rows.append(row)

    return pd.DataFrame(rows)


def compute_cohort_retention(entries: list[dict]) -> pd.DataFrame:
    """
    For each checkpoint, compute per-cohort metrics.

    A cohort is a group of facts edited in the same batch.
    Cohort K = facts with case_id in [K*100, (K+1)*100).

    This shows how early-edited facts retain their edits as more
    subsequent edits are applied.
    """
    rows = []

    for entry in entries:
        df = load_cases(entry["case_dir"])
        if df.empty or "case_id" not in df.columns:
            continue

        # Assign cohort (batch index)
        df["cohort"] = df["case_id"] // BATCH_SIZE

        # Compute per-cohort metrics
        for cohort, cohort_df in df.groupby("cohort"):
            row = {
                "seed": entry["seed"],
                "algorithm": entry["algorithm"],
                "total_edits": entry["total_edits"],
                "cohort": int(cohort),
                "cohort_edit_count": int(cohort) * BATCH_SIZE + BATCH_SIZE,
                "n_facts": len(cohort_df),
            }

            for metric in ["efficacy", "generalization", "specificity"]:
                if metric in cohort_df.columns:
                    values = cohort_df[metric].dropna().values
                    if len(values) > 0:
                        row[metric] = np.mean(values >= 0.5)

            rows.append(row)

    return pd.DataFrame(rows)


def plot_failure_curve(summary_df: pd.DataFrame, output_dir: Path) -> None:
    """
    Main failure curve figure: metrics vs total edit count.

    Shows individual seed lines (thin) and mean (thick) for each algorithm.
    """
    metrics = ["efficacy", "generalization", "specificity"]
    available = [m for m in metrics if m in summary_df.columns]

    fig, axes = plt.subplots(1, len(available), figsize=(4.5 * len(available), 4))
    if len(available) == 1:
        axes = [axes]

    for ax, metric in zip(axes, available):
        for alg in ["AlphaEdit", "MEMIT"]:
            alg_data = summary_df[summary_df["algorithm"] == alg]
            if alg_data.empty:
                continue

            color = COLORS[alg]

            # Plot per-seed lines (thin, transparent)
            for seed in alg_data["seed"].unique():
                seed_data = alg_data[alg_data["seed"] == seed].sort_values("total_edits")
                ax.plot(
                    seed_data["total_edits"], seed_data[metric],
                    "-", color=color, alpha=0.3, linewidth=1,
                )

            # Plot mean across seeds (thick)
            mean_data = (
                alg_data.groupby("total_edits")[metric]
                .agg(["mean", "std", "count"])
                .reset_index()
            )
            mean_data = mean_data.sort_values("total_edits")

            ax.plot(
                mean_data["total_edits"], mean_data["mean"],
                "-o", color=color, linewidth=2, markersize=5, label=alg,
            )

            # Shade CI (use Wilson CI from individual measurements if available)
            ci_lo_col = f"{metric}_ci_lo"
            ci_hi_col = f"{metric}_ci_hi"
            if ci_lo_col in alg_data.columns:
                ci_data = (
                    alg_data.groupby("total_edits")[[ci_lo_col, ci_hi_col]]
                    .mean()
                    .reset_index()
                    .sort_values("total_edits")
                )
                ax.fill_between(
                    ci_data["total_edits"],
                    ci_data[ci_lo_col],
                    ci_data[ci_hi_col],
                    alpha=0.1, color=color,
                )

        ax.set_xlabel("Total Sequential Edits")
        ax.set_ylabel(metric.capitalize())
        ax.set_title(metric.capitalize())
        ax.legend(loc="lower left")
        ax.set_ylim(0, 1.05)
        ax.axhline(y=0.5, color="gray", linestyle=":", alpha=0.3)
        ax.set_xticks(sorted(summary_df["total_edits"].unique()))
        ax.tick_params(axis="x", rotation=45)

    fig.suptitle(
        "Failure Curve: Metric Degradation vs Edit Count\n"
        "(Llama-3-8B-Instruct, MultiCounterFact, 100-edit batches)",
        y=1.04, fontsize=12,
    )
    fig.tight_layout()

    output_path = output_dir / "failure_curve.pdf"
    fig.savefig(output_path)
    plt.close(fig)
    print(f"Saved: {output_path}")

    # Also save PNG for quick viewing
    fig2, axes2 = plt.subplots(1, len(available), figsize=(4.5 * len(available), 4))
    if len(available) == 1:
        axes2 = [axes2]
    for ax, metric in zip(axes2, available):
        for alg in ["AlphaEdit", "MEMIT"]:
            alg_data = summary_df[summary_df["algorithm"] == alg]
            if alg_data.empty:
                continue
            color = COLORS[alg]
            for seed in alg_data["seed"].unique():
                seed_data = alg_data[alg_data["seed"] == seed].sort_values("total_edits")
                ax.plot(seed_data["total_edits"], seed_data[metric],
                        "-", color=color, alpha=0.3, linewidth=1)
            mean_data = (
                alg_data.groupby("total_edits")[metric]
                .agg(["mean"]).reset_index().sort_values("total_edits")
            )
            ax.plot(mean_data["total_edits"], mean_data["mean"],
                    "-o", color=color, linewidth=2, markersize=5, label=alg)
        ax.set_xlabel("Total Sequential Edits")
        ax.set_ylabel(metric.capitalize())
        ax.set_title(metric.capitalize())
        ax.legend(loc="lower left")
        ax.set_ylim(0, 1.05)
        ax.axhline(y=0.5, color="gray", linestyle=":", alpha=0.3)
        ax.set_xticks(sorted(summary_df["total_edits"].unique()))
        ax.tick_params(axis="x", rotation=45)
    fig2.suptitle(
        "Failure Curve: Metric Degradation vs Edit Count\n"
        "(Llama-3-8B-Instruct, MultiCounterFact, 100-edit batches)",
        y=1.04, fontsize=12,
    )
    fig2.tight_layout()
    fig2.savefig(output_dir / "failure_curve.png")
    plt.close(fig2)


def plot_cohort_retention(cohort_df: pd.DataFrame, output_dir: Path) -> None:
    """
    Cohort retention figure: how early-batch facts degrade at later checkpoints.

    Groups cohorts into bands (e.g., first 1000, 1001-2000, etc.) and shows
    their efficacy at each checkpoint.
    """
    if cohort_df.empty or "efficacy" not in cohort_df.columns:
        print("WARNING: No cohort data for retention plot")
        return

    # Define cohort bands (in terms of edit count)
    bands = [
        (0, 10, "Edits 1–1000"),
        (10, 20, "Edits 1001–2000"),
        (20, 30, "Edits 2001–3000"),
        (30, 50, "Edits 3001–5000"),
    ]
    band_colors = ["#1a237e", "#1565c0", "#42a5f5", "#90caf9"]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    for ax_idx, alg in enumerate(["AlphaEdit", "MEMIT"]):
        ax = axes[ax_idx]
        alg_data = cohort_df[cohort_df["algorithm"] == alg]

        if alg_data.empty:
            ax.set_title(f"{alg} (no data)")
            continue

        for (band_lo, band_hi, label), color in zip(bands, band_colors):
            band_data = alg_data[
                (alg_data["cohort"] >= band_lo) & (alg_data["cohort"] < band_hi)
            ]
            if band_data.empty:
                continue

            # Average efficacy across cohorts in this band and seeds
            grouped = (
                band_data.groupby("total_edits")["efficacy"]
                .mean()
                .reset_index()
                .sort_values("total_edits")
            )

            # Only plot at checkpoints where these cohorts have been edited
            min_edit_for_band = band_hi * BATCH_SIZE
            grouped = grouped[grouped["total_edits"] >= min_edit_for_band]

            if not grouped.empty:
                ax.plot(
                    grouped["total_edits"], grouped["efficacy"],
                    "-o", color=color, linewidth=2, markersize=4, label=label,
                )

        ax.set_xlabel("Total Sequential Edits at Evaluation")
        ax.set_ylabel("Efficacy (retention of edited fact)")
        ax.set_title(f"{alg}: Cohort Retention")
        ax.legend(loc="lower left", fontsize=8)
        ax.set_ylim(0, 1.05)
        ax.axhline(y=0.5, color="gray", linestyle=":", alpha=0.3)

    fig.suptitle(
        "Cohort Retention: Do Earlier Edits Survive Later Ones?",
        y=1.02, fontsize=12,
    )
    fig.tight_layout()

    output_path = output_dir / "cohort_retention.pdf"
    fig.savefig(output_path)
    plt.close(fig)
    print(f"Saved: {output_path}")

    # PNG version
    fig.savefig(output_dir / "cohort_retention.png")


def plot_glue_degradation(summary_df: pd.DataFrame, output_dir: Path) -> None:
    """Plot GLUE task scores vs edit count."""
    glue_cols = [c for c in summary_df.columns if c.startswith("glue_")]
    if not glue_cols:
        print("NOTE: No GLUE data available for degradation plot")
        return

    tasks = sorted(set(c.replace("glue_", "").replace("_f1", "") for c in glue_cols))

    fig, axes = plt.subplots(1, len(tasks), figsize=(3.5 * len(tasks), 4))
    if len(tasks) == 1:
        axes = [axes]

    for ax, task in zip(axes, tasks):
        col = f"glue_{task}_f1"
        if col not in summary_df.columns:
            continue

        for alg in ["AlphaEdit", "MEMIT"]:
            alg_data = summary_df[summary_df["algorithm"] == alg]
            if alg_data.empty:
                continue

            task_data = alg_data[["total_edits", col, "seed"]].dropna()
            if task_data.empty:
                continue

            color = COLORS[alg]
            mean_data = (
                task_data.groupby("total_edits")[col]
                .mean()
                .reset_index()
                .sort_values("total_edits")
            )
            ax.plot(
                mean_data["total_edits"], mean_data[col],
                "-o", color=color, linewidth=2, markersize=4, label=alg,
            )

        ax.set_xlabel("Total Edits")
        ax.set_ylabel("F1")
        ax.set_title(task.upper())
        ax.legend(fontsize=8)
        ax.set_ylim(0, 1.0)

    fig.suptitle("GLUE Preservation vs Edit Count", y=1.02, fontsize=12)
    fig.tight_layout()

    output_path = output_dir / "glue_degradation.pdf"
    fig.savefig(output_path)
    plt.close(fig)
    print(f"Saved: {output_path}")


def plot_gap_analysis(summary_df: pd.DataFrame, output_dir: Path) -> None:
    """
    Plot the AlphaEdit - MEMIT gap vs edit count.

    Key question: does the gap shrink to zero as edits accumulate?
    """
    metrics = ["efficacy", "generalization", "specificity"]
    available = [m for m in metrics if m in summary_df.columns]

    # Compute per-checkpoint means for each algorithm
    ae_means = (
        summary_df[summary_df["algorithm"] == "AlphaEdit"]
        .groupby("total_edits")[available].mean()
    )
    memit_means = (
        summary_df[summary_df["algorithm"] == "MEMIT"]
        .groupby("total_edits")[available].mean()
    )

    # Find common checkpoints
    common_edits = sorted(set(ae_means.index) & set(memit_means.index))
    if len(common_edits) < 2:
        print("NOTE: Need overlapping checkpoints for gap analysis")
        return

    gap_df = pd.DataFrame(index=common_edits)
    for metric in available:
        gap_df[metric] = ae_means.loc[common_edits, metric].values - memit_means.loc[common_edits, metric].values

    fig, ax = plt.subplots(figsize=(8, 4.5))

    metric_colors = {"efficacy": "#4caf50", "generalization": "#9c27b0", "specificity": "#ff5722"}

    for metric in available:
        color = metric_colors.get(metric, "gray")
        ax.plot(
            gap_df.index, gap_df[metric],
            "-o", color=color, linewidth=2, markersize=5, label=metric.capitalize(),
        )

    ax.axhline(y=0, color="black", linestyle="-", alpha=0.5, linewidth=0.8)
    ax.set_xlabel("Total Sequential Edits")
    ax.set_ylabel("AlphaEdit − MEMIT (Δ)")
    ax.set_title("AlphaEdit Advantage Over MEMIT vs Edit Count")
    ax.legend()
    ax.set_xticks(common_edits)
    ax.tick_params(axis="x", rotation=45)

    # Annotate: positive = AlphaEdit better, negative = MEMIT better
    ax.fill_between(gap_df.index, 0, gap_df[available[0]].clip(lower=0),
                    alpha=0.05, color="green", label="_nolegend_")
    ax.fill_between(gap_df.index, 0, gap_df[available[0]].clip(upper=0),
                    alpha=0.05, color="red", label="_nolegend_")

    fig.tight_layout()
    output_path = output_dir / "alphaedit_memit_gap.pdf"
    fig.savefig(output_path)
    plt.close(fig)
    print(f"Saved: {output_path}")


def print_summary_table(summary_df: pd.DataFrame) -> None:
    """Print a formatted summary table to stdout."""
    metrics = ["efficacy", "generalization", "specificity"]
    available = [m for m in metrics if m in summary_df.columns]

    print("\n" + "=" * 80)
    print("FAILURE CURVE SUMMARY")
    print("=" * 80)

    for alg in ["AlphaEdit", "MEMIT"]:
        alg_data = summary_df[summary_df["algorithm"] == alg].sort_values("total_edits")
        if alg_data.empty:
            continue

        print(f"\n{'─' * 40}")
        print(f"  {alg}")
        print(f"{'─' * 40}")

        # Group by total_edits, average across seeds
        grouped = alg_data.groupby("total_edits")[available + ["n_cases"]].mean()
        print(f"{'Edits':>8} | {'N':>6} | " + " | ".join(f"{m[:4]:>7}" for m in available))
        print(f"{'-' * 8}-+-{'-' * 6}-+-" + "-+-".join("-" * 7 for _ in available))

        for edits, row in grouped.iterrows():
            vals = " | ".join(f"{row[m]:7.4f}" for m in available)
            print(f"{edits:>8} | {int(row['n_cases']):>6} | {vals}")

    # Gap summary
    ae = summary_df[summary_df["algorithm"] == "AlphaEdit"].groupby("total_edits")[available].mean()
    memit = summary_df[summary_df["algorithm"] == "MEMIT"].groupby("total_edits")[available].mean()
    common = sorted(set(ae.index) & set(memit.index))

    if common:
        print(f"\n{'─' * 40}")
        print("  Δ (AlphaEdit − MEMIT)")
        print(f"{'─' * 40}")
        print(f"{'Edits':>8} | " + " | ".join(f"{m[:4]:>7}" for m in available))
        print(f"{'-' * 8}-+-" + "-+-".join("-" * 7 for _ in available))
        for edits in common:
            vals = " | ".join(f"{ae.loc[edits, m] - memit.loc[edits, m]:+7.4f}" for m in available)
            print(f"{edits:>8} | {vals}")

    print("\n" + "=" * 80)


def main():
    parser = argparse.ArgumentParser(description="Failure curve analysis")
    parser.add_argument(
        "--results_dir", type=Path,
        default=Path("results/failure_curve_checkpointed"),
        help="Root directory with seed*/Nedits/ structure",
    )
    parser.add_argument(
        "--output_dir", type=Path,
        default=Path("results/figures/failure_curve"),
        help="Output directory for figures and CSV",
    )
    parser.add_argument(
        "--skip_cohort", action="store_true",
        help="Skip cohort retention analysis (slower due to per-case loading)",
    )
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    print("=== Failure Curve Analysis ===\n")
    print(f"Results dir: {args.results_dir}")
    print(f"Output dir:  {args.output_dir}\n")

    # 1. Discover available data
    print("Scanning for results...")
    entries = find_result_dirs(args.results_dir)
    print(f"Found {len(entries)} data points:\n")
    for e in entries:
        print(f"  seed={e['seed']:>4}  alg={e['algorithm']:<10}  "
              f"edits={e['total_edits']:>5}  cases={e['n_cases']:>5}  "
              f"glue={'yes' if e['glue_dir'] else 'no'}")
    print()

    # 2. Aggregate metrics
    print("Aggregating per-case metrics...")
    summary_df = aggregate_failure_curve(entries)

    if summary_df.empty:
        print("ERROR: No data could be aggregated.")
        return

    # Save CSV
    csv_path = args.output_dir / "failure_curve_summary.csv"
    summary_df.to_csv(csv_path, index=False)
    print(f"\nSaved: {csv_path}")

    # 3. Print summary table
    print_summary_table(summary_df)

    # 4. Generate plots
    print("\nGenerating plots...")
    plot_failure_curve(summary_df, args.output_dir)
    plot_gap_analysis(summary_df, args.output_dir)
    plot_glue_degradation(summary_df, args.output_dir)

    # 5. Cohort retention (optional, slower)
    if not args.skip_cohort:
        print("\nComputing cohort retention (this may take a minute)...")
        cohort_df = compute_cohort_retention(entries)
        if not cohort_df.empty:
            cohort_csv = args.output_dir / "cohort_retention.csv"
            cohort_df.to_csv(cohort_csv, index=False)
            print(f"Saved: {cohort_csv}")
            plot_cohort_retention(cohort_df, args.output_dir)

    print("\n=== Done ===")


if __name__ == "__main__":
    main()

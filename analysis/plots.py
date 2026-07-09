#!/usr/bin/env python3
"""
Paper figure generation for AlphaEdit reproducibility study.

Generates publication-quality figures for TMLR submission:
1. Reproduction comparison table + delta heatmap
2. Metric degradation over sequential edit rounds
3. Conflict/paraphrase stress figure
4. Failure characterization curve (preservation vs total edit count)
5. Second model comparison (Llama vs Mistral)

Usage:
    python analysis/plots.py --results_dir results --output_dir results/figures
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

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

# Consistent bootstrap parameters across all plotting functions
N_BOOTSTRAP = 10000


def plot_reproduction_comparison(
    per_case_csv: Path, output_dir: Path
) -> None:
    """
    Bar chart: Efficacy/Generalization/Specificity for AlphaEdit vs MEMIT.
    With error bars from bootstrap CIs.
    """
    df = pd.read_csv(per_case_csv)

    metrics = ["efficacy", "generalization", "specificity"]
    available_metrics = [m for m in metrics if m in df.columns]

    if not available_metrics:
        print("WARNING: No metrics available for comparison plot")
        return

    # Compute means and CIs per algorithm
    algorithms = df["algorithm"].unique()
    data = []
    for alg in algorithms:
        alg_data = df[df["algorithm"] == alg]
        for metric in available_metrics:
            values = alg_data[metric].dropna()
            if len(values) > 0:
                mean = values.mean()
                # Bootstrap CI
                rng = np.random.default_rng(42)
                boots = [rng.choice(values, size=len(values), replace=True).mean()
                         for _ in range(N_BOOTSTRAP)]
                ci_low = np.percentile(boots, 2.5)
                ci_high = np.percentile(boots, 97.5)
                data.append({
                    "Algorithm": alg,
                    "Metric": metric.capitalize(),
                    "Mean": mean,
                    "CI_low": ci_low,
                    "CI_high": ci_high,
                })

    plot_df = pd.DataFrame(data)

    fig, ax = plt.subplots(figsize=(8, 4))
    x = np.arange(len(available_metrics))
    width = 0.35

    for i, alg in enumerate(algorithms):
        alg_df = plot_df[plot_df["Algorithm"] == alg]
        means = alg_df["Mean"].values
        errors = np.array([
            means - alg_df["CI_low"].values,
            alg_df["CI_high"].values - means,
        ])
        ax.bar(
            x + i * width - width / 2,
            means,
            width,
            yerr=errors,
            label=alg,
            capsize=3,
            alpha=0.85,
        )

    ax.set_xlabel("Metric")
    ax.set_ylabel("Score")
    ax.set_title("AlphaEdit vs MEMIT: Sequential Editing on MultiCounterFact")
    ax.set_xticks(x)
    ax.set_xticklabels([m.capitalize() for m in available_metrics])
    ax.legend()
    ax.set_ylim(0, 1.05)
    ax.axhline(y=1.0, color="gray", linestyle="--", alpha=0.3)

    output_path = output_dir / "reproduction_comparison.pdf"
    fig.savefig(output_path)
    plt.close(fig)
    print(f"Saved: {output_path}")


def plot_degradation_over_rounds(
    per_case_csv: Path, output_dir: Path
) -> None:
    """
    Line plot: How metrics change as more edits are applied sequentially.
    X-axis = edit round (case_id // num_edits), Y-axis = metric value.
    """
    df = pd.read_csv(per_case_csv)

    if "case_id" not in df.columns or "efficacy" not in df.columns:
        print("WARNING: Insufficient data for degradation plot")
        return

    # Infer edit round from case_id and num_edits
    num_edits = df["num_edits"].iloc[0] if "num_edits" in df.columns else 100
    df["edit_round"] = df["case_id"] // num_edits

    metrics = ["efficacy", "generalization", "specificity"]
    available_metrics = [m for m in metrics if m in df.columns]

    fig, axes = plt.subplots(1, len(available_metrics), figsize=(4 * len(available_metrics), 3.5))
    if len(available_metrics) == 1:
        axes = [axes]

    for ax, metric in zip(axes, available_metrics):
        for alg in df["algorithm"].unique():
            alg_data = df[df["algorithm"] == alg]
            grouped = alg_data.groupby("edit_round")[metric].mean()
            ax.plot(grouped.index, grouped.values, label=alg, marker="o", markersize=3)

        ax.set_xlabel("Edit Round")
        ax.set_ylabel(metric.capitalize())
        ax.set_title(metric.capitalize())
        ax.legend()
        ax.set_ylim(0, 1.05)

    fig.suptitle("Metric Degradation Over Sequential Edit Rounds", y=1.02)
    fig.tight_layout()

    output_path = output_dir / "degradation_over_rounds.pdf"
    fig.savefig(output_path)
    plt.close(fig)
    print(f"Saved: {output_path}")


def plot_delta_heatmap(
    paired_csv: Path, output_dir: Path
) -> None:
    """
    Heatmap showing AlphaEdit - MEMIT differences across metrics and datasets.
    """
    if not paired_csv.exists():
        print("WARNING: paired_bootstrap_results.csv not found, skipping heatmap")
        return

    df = pd.read_csv(paired_csv)

    if df.empty:
        return

    # Pivot for heatmap
    pivot = df.pivot_table(values="mean_diff", index="metric", aggfunc="mean")

    fig, ax = plt.subplots(figsize=(4, 3))
    sns.heatmap(
        pivot,
        annot=True,
        fmt=".3f",
        center=0,
        cmap="RdYlGn",
        ax=ax,
        cbar_kws={"label": "Δ (AlphaEdit − MEMIT)"},
    )
    ax.set_title("AlphaEdit vs MEMIT: Per-Metric Differences")
    ax.set_ylabel("")

    output_path = output_dir / "delta_heatmap.pdf"
    fig.savefig(output_path)
    plt.close(fig)
    print(f"Saved: {output_path}")


def plot_failure_characterization_curve(
    per_case_csv: Path, output_dir: Path
) -> None:
    """
    KEY FIGURE: Preservation metrics vs total edit count.

    Shows how AlphaEdit and MEMIT degrade as the total number of sequential
    edits increases. Identifies the crossover point (if any) where AlphaEdit's
    null-space advantage disappears.

    X-axis: Total edit count (500, 1000, 1500, 2000, 3000, 5000)
    Y-axis: Metric score (averaged over all cases at that edit count)
    Lines: AlphaEdit vs MEMIT, with shaded 95% CI
    """
    df = pd.read_csv(per_case_csv)

    if "case_id" not in df.columns:
        print("WARNING: No case_id column for failure curve")
        return

    metrics = ["efficacy", "generalization", "specificity"]
    available_metrics = [m for m in metrics if m in df.columns]

    if not available_metrics:
        print("WARNING: No metrics for failure curve")
        return

    # Group by algorithm and infer total edit count from max case_id per run
    # Each run at a different dataset_size_limit will have a different max case_id
    runs = df.groupby(["algorithm", "run_id"]).agg(
        total_edits=("case_id", "max"),
    ).reset_index()

    # Merge back to get total_edits per row
    df = df.merge(runs[["algorithm", "run_id", "total_edits"]], on=["algorithm", "run_id"])

    # Get unique edit count levels
    edit_levels = sorted(df["total_edits"].unique())
    if len(edit_levels) < 2:
        print("WARNING: Need multiple edit count levels for failure curve")
        return

    fig, axes = plt.subplots(1, len(available_metrics), figsize=(4.5 * len(available_metrics), 4))
    if len(available_metrics) == 1:
        axes = [axes]

    colors = {"AlphaEdit": "#2196F3", "MEMIT": "#FF9800"}

    for ax, metric in zip(axes, available_metrics):
        for alg in ["AlphaEdit", "MEMIT"]:
            alg_data = df[df["algorithm"] == alg]
            if alg_data.empty:
                continue

            means = []
            ci_lows = []
            ci_highs = []
            levels = []

            for level in edit_levels:
                level_data = alg_data[alg_data["total_edits"] == level][metric].dropna()
                if len(level_data) == 0:
                    continue
                levels.append(level)
                mean = level_data.mean()
                means.append(mean)

                # Bootstrap CI
                rng = np.random.default_rng(42)
                if len(level_data) > 1:
                    boots = [rng.choice(level_data.values, size=len(level_data), replace=True).mean()
                             for _ in range(N_BOOTSTRAP)]
                    ci_lows.append(np.percentile(boots, 2.5))
                    ci_highs.append(np.percentile(boots, 97.5))
                else:
                    ci_lows.append(mean)
                    ci_highs.append(mean)

            if levels:
                color = colors.get(alg, "gray")
                ax.plot(levels, means, "-o", label=alg, color=color, markersize=5)
                ax.fill_between(levels, ci_lows, ci_highs, alpha=0.15, color=color)

        ax.set_xlabel("Total Sequential Edits")
        ax.set_ylabel(metric.capitalize())
        ax.set_title(metric.capitalize())
        ax.legend()
        ax.set_ylim(0, 1.05)
        ax.axhline(y=0.5, color="gray", linestyle=":", alpha=0.3)

    fig.suptitle("Failure Characterization: Preservation vs Edit Count", y=1.02, fontsize=13)
    fig.tight_layout()

    output_path = output_dir / "failure_characterization_curve.pdf"
    fig.savefig(output_path)
    plt.close(fig)
    print(f"Saved: {output_path}")


def plot_second_model_comparison(
    per_case_csv: Path, output_dir: Path
) -> None:
    """
    Cross-model comparison: Llama-3-8B vs Mistral-7B.

    Grouped bar chart showing metrics for each (model, algorithm) pair.
    Demonstrates generalizability of AlphaEdit's advantage.
    """
    df = pd.read_csv(per_case_csv)

    # Infer model from run metadata or directory structure
    # For now, we look for model information in the data
    # If not available, skip
    if "model" not in df.columns:
        # Try to infer from results directory structure
        print("NOTE: No 'model' column in CSV. Cross-model plot requires model tagging.")
        print("  Add model information during aggregation for this plot.")
        return

    metrics = ["efficacy", "generalization", "specificity"]
    available_metrics = [m for m in metrics if m in df.columns]
    models = df["model"].unique()

    if len(models) < 2:
        print("WARNING: Need results from 2+ models for cross-model comparison")
        return

    fig, axes = plt.subplots(1, len(available_metrics), figsize=(4.5 * len(available_metrics), 4))
    if len(available_metrics) == 1:
        axes = [axes]

    for ax, metric in zip(axes, available_metrics):
        x = np.arange(len(models))
        width = 0.35

        for i, alg in enumerate(["AlphaEdit", "MEMIT"]):
            means = []
            errors_low = []
            errors_high = []
            for model in models:
                model_alg_data = df[(df["model"] == model) & (df["algorithm"] == alg)]
                values = model_alg_data[metric].dropna().values
                if len(values) > 0:
                    mean = np.mean(values)
                    means.append(mean)
                    rng = np.random.default_rng(42)
                    boots = [rng.choice(values, size=len(values), replace=True).mean()
                             for _ in range(N_BOOTSTRAP)]
                    errors_low.append(mean - np.percentile(boots, 2.5))
                    errors_high.append(np.percentile(boots, 97.5) - mean)
                else:
                    means.append(0)
                    errors_low.append(0)
                    errors_high.append(0)

            ax.bar(
                x + i * width - width / 2,
                means,
                width,
                yerr=[errors_low, errors_high],
                label=alg,
                capsize=3,
                alpha=0.85,
            )

        ax.set_xlabel("Model")
        ax.set_ylabel(metric.capitalize())
        ax.set_title(metric.capitalize())
        ax.set_xticks(x)
        ax.set_xticklabels([m.split("/")[-1] for m in models], fontsize=8)
        ax.legend()
        ax.set_ylim(0, 1.05)

    fig.suptitle("Cross-Model Comparison: AlphaEdit vs MEMIT", y=1.02, fontsize=13)
    fig.tight_layout()

    output_path = output_dir / "second_model_comparison.pdf"
    fig.savefig(output_path)
    plt.close(fig)
    print(f"Saved: {output_path}")


def plot_capability_degradation(
    probe_dir: Path, output_dir: Path
) -> None:
    """
    General capability degradation plot: perplexity over edit count.

    Shows whether AlphaEdit and MEMIT preserve general language modeling
    capabilities as edits accumulate. A rising perplexity indicates the
    model is losing general capabilities.

    X-axis: Total edits applied
    Y-axis: Perplexity on WikiText-103 (log scale)
    """
    import json

    if not probe_dir.exists():
        print("NOTE: No capability_probe/ directory found. Skipping capability plot.")
        return

    jsonl_files = sorted(probe_dir.glob("probe_*.jsonl"))
    if not jsonl_files:
        print("NOTE: No probe JSONL files found. Skipping capability plot.")
        return

    # Load all probe records, grouping by algorithm
    algo_data = {}
    for f in jsonl_files:
        # Parse algorithm from filename: probe_seed42_AlphaEdit_2000edits_*.jsonl
        parts = f.stem.split("_")
        alg = None
        for i, p in enumerate(parts):
            if p in ("AlphaEdit", "MEMIT", "ROME"):
                alg = p
                break
        if alg is None:
            continue

        records = []
        with open(f) as fh:
            for line in fh:
                line = line.strip()
                if line:
                    records.append(json.loads(line))

        if alg not in algo_data:
            algo_data[alg] = []
        algo_data[alg].extend(records)

    if not algo_data:
        print("NOTE: No parseable probe data. Skipping capability plot.")
        return

    colors = {"AlphaEdit": "#2196F3", "MEMIT": "#FF9800", "ROME": "#9E9E9E"}

    fig, ax = plt.subplots(figsize=(8, 4.5))

    for alg, records in algo_data.items():
        df = pd.DataFrame(records)
        if "edit_count" not in df.columns or "mean_perplexity" not in df.columns:
            continue

        df = df.sort_values("edit_count")

        # If multiple seeds, average
        grouped = df.groupby("edit_count")["mean_perplexity"].agg(["mean", "std"]).reset_index()

        color = colors.get(alg, "gray")
        ax.plot(
            grouped["edit_count"], grouped["mean"],
            "-o", color=color, markersize=4, label=alg, linewidth=1.5,
        )

        # Shade std if available
        if grouped["std"].notna().any() and (grouped["std"] > 0).any():
            ax.fill_between(
                grouped["edit_count"],
                grouped["mean"] - grouped["std"],
                grouped["mean"] + grouped["std"],
                alpha=0.15, color=color,
            )

    ax.set_xlabel("Total Sequential Edits Applied")
    ax.set_ylabel("Perplexity (WikiText-103)")
    ax.set_title("General Capability Preservation: Perplexity vs Edit Count")
    ax.legend()
    ax.set_yscale("log")

    # Reference line at baseline (first measurement)
    for alg, records in algo_data.items():
        baseline = [r for r in records if r.get("edit_count", -1) == 0]
        if baseline:
            baseline_ppl = baseline[0].get("mean_perplexity")
            if baseline_ppl:
                ax.axhline(y=baseline_ppl, color=colors.get(alg, "gray"),
                          linestyle=":", alpha=0.4, linewidth=0.8)

    output_path = output_dir / "capability_degradation.pdf"
    fig.savefig(output_path)
    plt.close(fig)
    print(f"Saved: {output_path}")


def plot_mitigation_comparison(
    mitigation_dir: Path, output_dir: Path
) -> None:
    """
    Grouped bar chart: metric scores for each mitigation variant vs control.

    Reads per-case CSVs from mitigation results and compares against
    unmodified AlphaEdit (control). Highlights variants that improve
    specificity without sacrificing efficacy.
    """
    import json

    if not mitigation_dir.exists():
        print("NOTE: No mitigation/ directory found. Skipping mitigation plot.")
        return

    # Look for metadata JSONL files to identify variants
    metadata_files = sorted(mitigation_dir.glob("mitigation_*.jsonl"))
    if not metadata_files:
        print("NOTE: No mitigation metadata files found. Skipping mitigation plot.")
        return

    # Parse variant results
    variants = []
    for mf in metadata_files:
        with open(mf) as f:
            for line in f:
                if line.strip():
                    variants.append(json.loads(line))

    if not variants:
        return

    # Group by strategy and parameters
    strategy_labels = []
    for v in variants:
        strategy = v.get("strategy", "unknown")
        if strategy == "svd_truncation":
            label = f"SVD(K={v.get('truncation_interval')},r={v.get('retain_ratio')})"
        elif strategy == "exponential_decay":
            label = f"Decay({v.get('decay_factor')})"
        elif strategy == "periodic_reset":
            label = f"Reset(K={v.get('reset_interval')})"
        else:
            label = strategy
        strategy_labels.append(label)

    # For now, create a placeholder figure showing variant labels
    # Full implementation requires per-case metrics from result JSONs
    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(strategy_labels))
    ax.bar(x, [v.get("total_mitigations_applied", 0) for v in variants], alpha=0.7)
    ax.set_xticks(x)
    ax.set_xticklabels(strategy_labels, rotation=45, ha="right", fontsize=8)
    ax.set_xlabel("Mitigation Variant")
    ax.set_ylabel("Mitigations Applied")
    ax.set_title("Cache Mitigation Sweep: Interventions per Variant")

    output_path = output_dir / "mitigation_comparison.pdf"
    fig.savefig(output_path)
    plt.close(fig)
    print(f"Saved: {output_path}")


def plot_order_sensitivity(
    order_dir: Path, output_dir: Path
) -> None:
    """
    Box/violin plot: metric distribution across orderings for AlphaEdit vs MEMIT.

    Shows whether AlphaEdit is more sensitive to edit ordering than MEMIT.
    Annotates with coefficient of variation and Levene's test p-value.
    """
    import json

    if not order_dir.exists():
        print("NOTE: No order_sensitivity/ directory found. Skipping order plot.")
        return

    metadata_files = sorted(order_dir.glob("order_*.jsonl"))
    if not metadata_files:
        print("NOTE: No order sensitivity metadata files found. Skipping order plot.")
        return

    # Parse metadata to understand which runs exist
    runs = []
    for mf in metadata_files:
        with open(mf) as f:
            for line in f:
                if line.strip():
                    runs.append(json.loads(line))

    if not runs:
        return

    # Group by algorithm
    alg_order_seeds = {}
    for r in runs:
        alg = r.get("alg_name", "unknown")
        if alg not in alg_order_seeds:
            alg_order_seeds[alg] = []
        alg_order_seeds[alg].append(r.get("order_seed"))

    # Create summary figure
    fig, ax = plt.subplots(figsize=(8, 5))

    algorithms = sorted(alg_order_seeds.keys())
    x = np.arange(len(algorithms))
    counts = [len(alg_order_seeds[alg]) for alg in algorithms]

    colors = {"AlphaEdit": "#2196F3", "MEMIT": "#FF9800"}
    bar_colors = [colors.get(alg, "gray") for alg in algorithms]

    ax.bar(x, counts, color=bar_colors, alpha=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(algorithms)
    ax.set_xlabel("Algorithm")
    ax.set_ylabel("Number of Ordering Runs Completed")
    ax.set_title("Edit Order Sensitivity: Runs per Algorithm")

    # Add annotation about what the full analysis will show
    ax.text(
        0.5, 0.85,
        "Full analysis requires per-case result aggregation\n"
        "(box plots of metric variance across orderings)",
        transform=ax.transAxes, ha="center", fontsize=9,
        style="italic", color="gray",
    )

    output_path = output_dir / "order_sensitivity.pdf"
    fig.savefig(output_path)
    plt.close(fig)
    print(f"Saved: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Generate paper figures")
    parser.add_argument(
        "--results_dir", type=Path, default=Path("results")
    )
    parser.add_argument(
        "--output_dir", type=Path, default=Path("results/figures")
    )
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    per_case_csv = args.results_dir / "per_case_results.csv"
    paired_csv = args.results_dir / "paired_bootstrap_results.csv"

    if not per_case_csv.exists():
        print(f"ERROR: {per_case_csv} not found. Run aggregate.py first.")
        return

    print("=== Generating Paper Figures ===\n")
    plot_reproduction_comparison(per_case_csv, args.output_dir)
    plot_degradation_over_rounds(per_case_csv, args.output_dir)
    plot_failure_characterization_curve(per_case_csv, args.output_dir)
    plot_second_model_comparison(per_case_csv, args.output_dir)
    plot_delta_heatmap(paired_csv, args.output_dir)
    plot_capability_degradation(args.results_dir / "capability_probe", args.output_dir)
    plot_mitigation_comparison(args.results_dir / "mitigation", args.output_dir)
    plot_order_sensitivity(args.results_dir / "order_sensitivity", args.output_dir)
    print("\n=== Done ===")


if __name__ == "__main__":
    main()

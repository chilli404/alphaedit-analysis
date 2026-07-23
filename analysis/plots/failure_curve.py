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
import random
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
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
SAMPLE_SIZE = 2000  # Per-checkpoint sample for 4-panel speed


def find_result_dirs(base_dir: Path) -> list[dict]:
    """
    Scan the checkpoint results directory and find all available data points.

    Supports two layouts:
      - Nested:  seed{N}/{M}edits/{wrapper}/AlphaEdit/run_000/*.json
      - Flat:    seed{N}/{M}edits/AlphaEdit/run_000/*.json

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

            # Collect candidate algorithm dirs from both flat and nested layouts
            alg_dirs = []
            for subdir in edits_dir.iterdir():
                if not subdir.is_dir():
                    continue
                if subdir.name in ("AlphaEdit", "MEMIT"):
                    # Flat layout: {M}edits/AlphaEdit/run_000/
                    alg_dirs.append(subdir)
                else:
                    # Nested layout: {M}edits/{wrapper}/AlphaEdit/run_000/
                    for alg_dir in subdir.iterdir():
                        if alg_dir.is_dir() and alg_dir.name in ("AlphaEdit", "MEMIT"):
                            alg_dirs.append(alg_dir)

            for alg_dir in alg_dirs:
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

    glue_file = glue_dir / "edit_glue.json"
    if glue_file.exists():
        with open(glue_file) as f:
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

    output_path = output_dir / "failure_curve.png"
    fig.savefig(output_path)
    plt.close(fig)
    print(f"Saved: {output_path}")


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

    output_path = output_dir / "cohort_retention.png"
    fig.savefig(output_path)
    plt.close(fig)
    print(f"Saved: {output_path}")


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

    output_path = output_dir / "glue_degradation.png"
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
    output_path = output_dir / "alphaedit_memit_gap.png"
    fig.savefig(output_path)
    plt.close(fig)
    print(f"Saved: {output_path}")



# --- 4-Panel Figure Functions ---


def compute_4panel_metrics(entry: dict) -> dict:
    """
    Load case files for one (seed, alg, edits) entry with sampling.
    Returns dict with efficacy, generalization, spec_prob (probability-based).
    """
    case_dir = entry["case_dir"]
    case_files = sorted(case_dir.glob("*_edits-case_*.json"))

    rng = random.Random(42)
    sample = rng.sample(case_files, min(SAMPLE_SIZE, len(case_files)))

    eff_vals = []
    gen_vals = []
    spec_prob_vals = []

    for cf in sample:
        with open(cf) as f:
            d = json.load(f)
        post = d["post"]

        rc = post.get("rewrite_prompts_correct", [])
        if rc:
            eff_vals.append(sum(rc) / len(rc))

        pc = post.get("paraphrase_prompts_correct", [])
        if pc:
            gen_vals.append(sum(pc) / len(pc))

        np_probs = post.get("neighborhood_prompts_probs", [])
        if np_probs:
            prob_correct = sum(1 for p in np_probs if p["target_true"] < p["target_new"])
            spec_prob_vals.append(prob_correct / len(np_probs))

    return {
        "seed": entry["seed"],
        "algorithm": entry["algorithm"],
        "total_edits": entry["total_edits"],
        "efficacy": np.mean(eff_vals) if eff_vals else np.nan,
        "generalization": np.mean(gen_vals) if gen_vals else np.nan,
        "spec_prob": np.mean(spec_prob_vals) if spec_prob_vals else np.nan,
        "n_sampled": len(sample),
    }


def compute_cohort_heatmap(entries: list[dict], algorithm: str = "AlphaEdit") -> tuple:
    """
    Build a cohort x checkpoint heatmap matrix.

    Rows = cohort bands (1000-edit groups)
    Columns = checkpoint edit counts
    Values = mean efficacy for that cohort at that checkpoint
    """
    alg_entries = [e for e in entries if e["algorithm"] == algorithm]
    checkpoints = sorted(set(e["total_edits"] for e in alg_entries))

    max_edits = max(checkpoints)
    n_bands = max_edits // 1000
    band_labels = [f"{i}K\u2013{i+1}K" for i in range(n_bands)]

    heatmap = np.full((n_bands, len(checkpoints)), np.nan)

    for col_idx, edits in enumerate(checkpoints):
        ckpt_entries = [e for e in alg_entries if e["total_edits"] == edits]
        cohort_efficacies = {}

        for entry in ckpt_entries:
            case_dir = entry["case_dir"]
            case_files = sorted(case_dir.glob("*_edits-case_*.json"))

            rng = random.Random(42)
            sample = rng.sample(case_files, min(SAMPLE_SIZE, len(case_files)))

            for cf in sample:
                with open(cf) as f:
                    d = json.load(f)
                case_id = d["case_id"]
                band_idx = (case_id // BATCH_SIZE) // 10

                if band_idx >= n_bands:
                    continue
                if (band_idx + 1) * 1000 > edits:
                    continue

                rc = d["post"].get("rewrite_prompts_correct", [])
                if rc:
                    eff = sum(rc) / len(rc)
                    if band_idx not in cohort_efficacies:
                        cohort_efficacies[band_idx] = []
                    cohort_efficacies[band_idx].append(eff)

        for band_idx, vals in cohort_efficacies.items():
            heatmap[band_idx, col_idx] = np.mean(vals)

    return heatmap, band_labels, checkpoints


def plot_4panel(
    metrics_df: pd.DataFrame,
    heatmap: np.ndarray,
    band_labels: list[str],
    checkpoints: list[int],
    output_dir: Path,
) -> None:
    """Generate the 4-panel failure characterization figure."""

    fig = plt.figure(figsize=(12, 9))
    gs = GridSpec(2, 2, figure=fig, hspace=0.35, wspace=0.30)

    # Panel A: Cumulative Efficacy
    ax_a = fig.add_subplot(gs[0, 0])

    for alg in ["AlphaEdit", "MEMIT"]:
        alg_data = metrics_df[metrics_df["algorithm"] == alg]
        if alg_data.empty:
            continue
        color = COLORS[alg]

        for seed in alg_data["seed"].unique():
            seed_data = alg_data[alg_data["seed"] == seed].sort_values("total_edits")
            ax_a.plot(seed_data["total_edits"], seed_data["efficacy"],
                      "-", color=color, alpha=0.25, linewidth=1)

        mean_data = (
            alg_data.groupby("total_edits")["efficacy"]
            .mean().reset_index().sort_values("total_edits")
        )
        ax_a.plot(mean_data["total_edits"], mean_data["efficacy"],
                  "-o", color=color, linewidth=2.2, markersize=5, label=alg)

    ax_a.set_xlabel("Total Sequential Edits")
    ax_a.set_ylabel("Efficacy (argmax)")
    ax_a.set_title("A. Cumulative Edit Efficacy")
    ax_a.legend(loc="upper right")
    ax_a.set_ylim(-0.02, 1.02)
    ax_a.axhline(y=0.5, color="gray", linestyle=":", alpha=0.3, linewidth=0.8)
    ax_a.set_xticks(sorted(metrics_df["total_edits"].unique()))
    ax_a.tick_params(axis="x", rotation=45)

    # Highlight transition zone
    ax_a.axvspan(5000, 7000, alpha=0.06, color="red", label="_nolegend_")
    ax_a.text(5800, 0.95, "transition\nzone", ha="center", fontsize=7,
              color="red", alpha=0.7, style="italic")

    # Panel B: Cohort Retention Heatmap
    ax_b = fig.add_subplot(gs[0, 1])

    masked_heatmap = np.ma.masked_invalid(heatmap)
    cmap = plt.cm.RdYlGn.copy()
    cmap.set_bad(color="white")

    im = ax_b.imshow(
        masked_heatmap, aspect="auto", cmap=cmap, vmin=0, vmax=1,
        origin="lower", interpolation="nearest",
    )

    ax_b.set_xticks(range(len(checkpoints)))
    ax_b.set_xticklabels([f"{c//1000}K" for c in checkpoints], fontsize=7)
    ax_b.set_xlabel("Evaluation Checkpoint (total edits)")

    n_bands = len(band_labels)
    y_tick_indices = list(range(0, n_bands, 2)) if n_bands > 6 else list(range(n_bands))
    ax_b.set_yticks(y_tick_indices)
    ax_b.set_yticklabels([band_labels[i] for i in y_tick_indices], fontsize=7)
    ax_b.set_ylabel("Edit Cohort (when edited)")
    ax_b.set_title("B. Cohort Retention (AlphaEdit)")

    cbar = fig.colorbar(im, ax=ax_b, fraction=0.046, pad=0.04)
    cbar.set_label("Efficacy", fontsize=8)
    cbar.ax.tick_params(labelsize=7)

    # Panel C: Efficacy vs Probability-Specificity
    ax_c = fig.add_subplot(gs[1, 0])

    ae_data = metrics_df[metrics_df["algorithm"] == "AlphaEdit"]
    ae_mean = ae_data.groupby("total_edits")[["efficacy", "spec_prob"]].mean().reset_index()
    ae_mean = ae_mean.sort_values("total_edits")

    ax_c.plot(ae_mean["total_edits"], ae_mean["efficacy"],
              "-o", color=COLORS["AlphaEdit"], linewidth=2.2, markersize=5,
              label="Efficacy (target retention)")
    ax_c.plot(ae_mean["total_edits"], ae_mean["spec_prob"],
              "-s", color="#4caf50", linewidth=2.2, markersize=5,
              label="Prob-specificity (locality)")

    for seed in ae_data["seed"].unique():
        sd = ae_data[ae_data["seed"] == seed].sort_values("total_edits")
        ax_c.plot(sd["total_edits"], sd["efficacy"],
                  "-", color=COLORS["AlphaEdit"], alpha=0.2, linewidth=1)
        ax_c.plot(sd["total_edits"], sd["spec_prob"],
                  "-", color="#4caf50", alpha=0.2, linewidth=1)

    # Mark crossover point
    eff_vals = ae_mean["efficacy"].values
    spec_vals = ae_mean["spec_prob"].values
    edits_vals = ae_mean["total_edits"].values
    for i in range(len(eff_vals) - 1):
        if eff_vals[i] >= spec_vals[i] and eff_vals[i + 1] < spec_vals[i + 1]:
            t = (eff_vals[i] - spec_vals[i]) / (
                (eff_vals[i] - spec_vals[i]) - (eff_vals[i + 1] - spec_vals[i + 1])
            )
            cross_x = edits_vals[i] + t * (edits_vals[i + 1] - edits_vals[i])
            cross_y = eff_vals[i] + t * (eff_vals[i + 1] - eff_vals[i])
            ax_c.axvline(x=cross_x, color="gray", linestyle="--", alpha=0.5, linewidth=0.8)
            ax_c.plot(cross_x, cross_y, "x", color="black", markersize=10, markeredgewidth=2)
            ax_c.annotate(
                f"crossover\n\u2248{int(cross_x)} edits",
                xy=(cross_x, cross_y), xytext=(cross_x + 400, cross_y + 0.08),
                fontsize=7, ha="left", color="black", style="italic",
                arrowprops=dict(arrowstyle="->", color="black", lw=0.8),
            )
            break

    ax_c.axhline(y=0.5, color="gray", linestyle=":", alpha=0.4, linewidth=0.8)
    ax_c.set_xlabel("Total Sequential Edits")
    ax_c.set_ylabel("Score")
    ax_c.set_title("C. Locality vs Retention (AlphaEdit)")
    ax_c.legend(loc="upper right", fontsize=8)
    ax_c.set_ylim(-0.02, 1.02)
    ax_c.set_xticks(sorted(ae_data["total_edits"].unique()))
    ax_c.tick_params(axis="x", rotation=45)

    # Panel D: AlphaEdit-MEMIT Gap
    ax_d = fig.add_subplot(gs[1, 1])

    ae_means = (
        metrics_df[metrics_df["algorithm"] == "AlphaEdit"]
        .groupby("total_edits")[["efficacy", "spec_prob"]].mean()
    )
    memit_means = (
        metrics_df[metrics_df["algorithm"] == "MEMIT"]
        .groupby("total_edits")[["efficacy", "spec_prob"]].mean()
    )
    common = sorted(set(ae_means.index) & set(memit_means.index))

    if common:
        eff_gap = [ae_means.loc[e, "efficacy"] - memit_means.loc[e, "efficacy"] for e in common]
        spec_gap = [ae_means.loc[e, "spec_prob"] - memit_means.loc[e, "spec_prob"] for e in common]

        ax_d.plot(common, eff_gap, "-o", color=COLORS["AlphaEdit"], linewidth=2.2,
                  markersize=5, label="\u0394 Efficacy")
        ax_d.plot(common, spec_gap, "-s", color="#4caf50", linewidth=2.2,
                  markersize=5, label="\u0394 Prob-specificity")

        ax_d.axhline(y=0, color="black", linestyle="-", alpha=0.5, linewidth=0.8)
        ax_d.fill_between(common, 0, eff_gap, alpha=0.08, color=COLORS["AlphaEdit"])
        ax_d.fill_between(common, 0, spec_gap, alpha=0.08, color="#4caf50")

        if len(spec_gap) > 1 and spec_gap[-1] < 0.03:
            ax_d.annotate(
                "converged",
                xy=(common[-1], spec_gap[-1]),
                xytext=(common[-1] - 800, spec_gap[-1] + 0.08),
                fontsize=7, color="#4caf50", style="italic",
                arrowprops=dict(arrowstyle="->", color="#4caf50", lw=0.8),
            )

    ax_d.set_xlabel("Total Sequential Edits")
    ax_d.set_ylabel("AlphaEdit \u2212 MEMIT (\u0394)")
    ax_d.set_title("D. Advantage Gap (\u2192 0 = no advantage)")
    ax_d.legend(loc="upper right", fontsize=8)
    if common:
        ax_d.set_xticks(common)
    ax_d.tick_params(axis="x", rotation=45)

    fig.suptitle(
        "AlphaEdit Failure Characterization: 10,000 Sequential Edits\n"
        "(Llama-3-8B-Instruct, MultiCounterFact, 100-edit batches)",
        fontsize=12, y=0.98,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "failure_curve_4panel.png"
    fig.savefig(output_path)
    print(f"Saved: {output_path}")
    plt.close(fig)


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
    parser.add_argument(
        "--skip_4panel", action="store_true",
        help="Skip 4-panel figure (slower due to heatmap computation)",
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

    # 6. 4-panel figure (optional, slower due to heatmap + sampling)
    if not args.skip_4panel:
        print("\nGenerating 4-panel figure (sampling up to 2000 cases per checkpoint)...")
        rows_4p = []
        for entry in entries:
            rows_4p.append(compute_4panel_metrics(entry))
        metrics_4p_df = pd.DataFrame(rows_4p)

        print("Computing cohort retention heatmap for 4-panel...")
        heatmap, band_labels, ckpts = compute_cohort_heatmap(entries, algorithm="AlphaEdit")
        print(f"  Heatmap shape: {heatmap.shape} (bands x checkpoints)")

        plot_4panel(metrics_4p_df, heatmap, band_labels, ckpts, args.output_dir)

        csv_4p = args.output_dir / "failure_curve_4panel_data.csv"
        metrics_4p_df.to_csv(csv_4p, index=False)
        print(f"Saved: {csv_4p}")

    print("\n=== Done ===")


if __name__ == "__main__":
    main()

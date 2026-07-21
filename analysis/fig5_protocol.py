"""Figure 5: Sequential-Memory Evaluation Protocol.

Panel A — Cohort retention curves (age-binned, per method)
Panel B — Cross-checkpoint retention trajectories (first-1K over time)
Panel C — Protocol radar chart (all 8 metrics normalized, per method)
Panel D — Retention AUC trajectory (AUC vs total edits)

Usage:
    uv run python -m analysis.fig5_protocol
    uv run python -m analysis.fig5_protocol --output-dir results/figures/paper
    uv run python -m analysis.fig5_protocol --edits 5000 --seed 42
"""

import argparse
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np

from analysis.style import (
    ALGO_COLORS,
    PAPER_OUTPUT,
    RESULTS,
    setup_style,
    save_figure,
)
from src.protocol.sequential_memory_eval import (
    MethodReport,
    compute_cross_checkpoint_auc,
    compute_retention_auc,
    evaluate_method,
    _load_cohort_efficacies,
)

# ─── Configuration ───────────────────────────────────────────────────────────

EDIT_POINTS = [2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000, 10000]
METHODS = ["AlphaEdit", "MEMIT"]
BATCH_SIZE = 100

ALGO_COLORS_EXTENDED = {
    **ALGO_COLORS,
    "MEMIT+SeqReg": "#4CAF50",
}


# ─── Panel A: Age-Binned Retention Curves ────────────────────────────────────


def panel_a(ax, reports: List[MethodReport]):
    """Cohort retention curves: efficacy vs edit age, per method."""
    for report in reports:
        if report.retention_curve is None:
            continue
        color = ALGO_COLORS_EXTENDED.get(report.method, "#666666")
        ax.plot(
            report.retention_curve.ages,
            report.retention_curve.efficacies,
            marker="o",
            markersize=4,
            color=color,
            label=report.method,
            linewidth=1.5,
        )

    ax.set_xlabel("Edit age (edits since insertion)")
    ax.set_ylabel("Efficacy")
    ax.set_ylim(-0.05, 1.05)
    ax.legend(loc="lower left", framealpha=0.9)
    ax.set_title("A. Retention by edit age", loc="left", fontweight="bold")


# ─── Panel B: First-1K Retention Trajectory ──────────────────────────────────


def panel_b(ax, seed: int):
    """First-1K retention across increasing total edit counts."""
    for alg in METHODS:
        color = ALGO_COLORS_EXTENDED.get(alg, "#666666")
        xs = []
        ys = []
        for edits in EDIT_POINTS:
            cohorts = _load_cohort_efficacies(seed, edits, alg, BATCH_SIZE)
            if cohorts is None:
                continue
            first_1k_vals = [cohorts[i] for i in range(10) if i in cohorts]
            if first_1k_vals:
                xs.append(edits)
                ys.append(float(np.mean(first_1k_vals)))
        if xs:
            ax.plot(xs, ys, marker="s", markersize=4, color=color,
                    label=alg, linewidth=1.5)

    # Add SeqReg if available
    seqreg_path = RESULTS / "memit_seqreg" / f"full_eval_seed{seed}_lp1.0_ld1.0.json"
    if seqreg_path.exists():
        import json
        with open(seqreg_path) as f:
            data = json.load(f)
        xs, ys = [], []
        for key, checkpoint in sorted(data.items()):
            if not key.endswith("_edits"):
                continue
            edits_val = int(key.replace("_edits", ""))
            f1k = checkpoint.get("first_1k", {}).get("efficacy")
            if f1k is not None:
                xs.append(edits_val)
                ys.append(f1k)
        if xs:
            ax.plot(xs, ys, marker="^", markersize=4,
                    color=ALGO_COLORS_EXTENDED["MEMIT+SeqReg"],
                    label="MEMIT+SeqReg", linewidth=1.5)

    ax.set_xlabel("Total edits applied")
    ax.set_ylabel("First-1K efficacy")
    ax.set_ylim(-0.05, 1.05)
    ax.legend(loc="lower left", framealpha=0.9)
    ax.set_title("B. First-1K retention trajectory", loc="left", fontweight="bold")


# ─── Panel C: Protocol Summary Bar Chart ─────────────────────────────────────


def panel_c(ax, reports: List[MethodReport]):
    """Grouped bar chart of key protocol metrics per method."""
    metrics = ["current_batch_efficacy", "latest_1k_efficacy",
               "first_1k_retention", "retention_auc"]
    labels = ["Current\nbatch", "Latest\n1K", "First-1K\nretention", "Retention\nAUC"]

    # Filter to methods with data
    valid_reports = [r for r in reports if r.retention_auc is not None]
    if not valid_reports:
        ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
        ax.set_title("C. Protocol metrics comparison", loc="left", fontweight="bold")
        return

    n_methods = len(valid_reports)
    n_metrics = len(metrics)
    x = np.arange(n_metrics)
    width = 0.8 / n_methods

    for i, report in enumerate(valid_reports):
        color = ALGO_COLORS_EXTENDED.get(report.method, "#666666")
        vals = [getattr(report, m) or 0 for m in metrics]
        offset = (i - n_methods / 2 + 0.5) * width
        ax.bar(x + offset, vals, width, color=color, label=report.method,
               edgecolor="white", linewidth=0.5)

    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim(0, 1.1)
    ax.set_ylabel("Score")
    ax.legend(loc="upper right", framealpha=0.9)
    ax.set_title("C. Protocol metrics comparison", loc="left", fontweight="bold")


# ─── Panel D: Retention AUC Trajectory ───────────────────────────────────────


def panel_d(ax, seed: int):
    """Retention AUC vs total edits — shows how quickly each method degrades."""
    for alg in METHODS:
        color = ALGO_COLORS_EXTENDED.get(alg, "#666666")
        xs = []
        ys = []
        for edits in EDIT_POINTS:
            cohorts = _load_cohort_efficacies(seed, edits, alg, BATCH_SIZE)
            if cohorts is None:
                continue
            auc = compute_retention_auc(cohorts, edits, BATCH_SIZE)
            if auc is not None:
                xs.append(edits)
                ys.append(auc)
        if xs:
            ax.plot(xs, ys, marker="o", markersize=4, color=color,
                    label=alg, linewidth=1.5)

    # Add SeqReg
    seqreg_path = RESULTS / "memit_seqreg" / f"full_eval_seed{seed}_lp1.0_ld1.0.json"
    if seqreg_path.exists():
        import json
        with open(seqreg_path) as f:
            data = json.load(f)
        xs, ys = [], []
        for key, checkpoint in sorted(data.items()):
            if not key.endswith("_edits"):
                continue
            auc = checkpoint.get("retention_auc")
            if auc is not None:
                edits_val = int(key.replace("_edits", ""))
                xs.append(edits_val)
                ys.append(auc)
        if xs:
            ax.plot(xs, ys, marker="^", markersize=4,
                    color=ALGO_COLORS_EXTENDED["MEMIT+SeqReg"],
                    label="MEMIT+SeqReg", linewidth=1.5)

    ax.set_xlabel("Total edits applied")
    ax.set_ylabel("Retention AUC")
    ax.set_ylim(-0.05, 1.05)
    ax.axhline(y=1.0, color="gray", linestyle="--", alpha=0.3, linewidth=0.8)
    ax.legend(loc="lower left", framealpha=0.9)
    ax.set_title("D. Retention AUC trajectory", loc="left", fontweight="bold")


# ─── Main Figure ─────────────────────────────────────────────────────────────


def generate(seed: int = 42, edits: int = 5000, output_dir: Path = PAPER_OUTPUT):
    """Generate the 4-panel protocol figure."""
    setup_style()
    output_dir.mkdir(parents=True, exist_ok=True)

    # Compute protocol reports
    reports = []
    for alg in ["AlphaEdit", "MEMIT", "MEMIT+SeqReg"]:
        report = evaluate_method(alg, seed, edits, BATCH_SIZE)
        reports.append(report)

    # Create figure
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    fig.suptitle(
        f"Sequential-Memory Evaluation Protocol (seed {seed}, {edits} edits)",
        fontsize=13, fontweight="bold", y=0.98,
    )

    panel_a(axes[0, 0], reports)
    panel_b(axes[0, 1], seed)
    panel_c(axes[1, 0], reports)
    panel_d(axes[1, 1], seed)

    fig.tight_layout(rect=[0, 0, 1, 0.95])
    save_figure(fig, "fig5_protocol", output_dir)


# ─── CLI ─────────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(description="Generate Figure 5: Protocol")
    parser.add_argument("--output-dir", type=Path, default=PAPER_OUTPUT)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--edits", type=int, default=5000)
    args = parser.parse_args()
    generate(seed=args.seed, edits=args.edits, output_dir=args.output_dir)


if __name__ == "__main__":
    main()

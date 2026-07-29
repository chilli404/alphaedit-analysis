"""Figure 3 — Local plasticity vs global retention (THE central result).

Question answered: Does AlphaEdit's null-space projection truly separate
editability from preservation, or does retention degrade selectively for
early edits while recent edits remain intact?

Panels:
  A. Age-binned retention at 10K (connected points by seed)
  B. First-cohort retention over time (line plot, per-seed)
  C. Latest vs first cohort at 3K and 5K (matched ordering data)
  D. Order sensitivity grows with horizon (dot plots)

Usage:
    uv run python -m analysis.fig3_plasticity
    uv run python -m analysis.fig3_plasticity --output-dir results/figures/paper
"""

import argparse
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from analysis.style import (
    ALGO_COLORS, SEED_COLORS, setup_style, save_figure, PAPER_OUTPUT,
)
from analysis.loaders import (
    load_checkpoint_cohorts,
    load_matched_ordering_full_eval,
)

# ─── Configuration ────────────────────────────────────────────────────────────

SEEDS = [42, 2024]
EDIT_POINTS = [2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000, 10000]
BATCH_SIZE = 100

# Age bins for panel A: cohort indices → label
AGE_BINS = [
    ("First 1K", range(0, 10)),
    ("1K\u20132K", range(10, 20)),
    ("3K\u20135K", range(20, 50)),
    ("5K\u20137K", range(50, 70)),
    ("7K\u20139K", range(70, 90)),
    ("Latest 1K", range(90, 100)),
]


# ─── Panel Functions ──────────────────────────────────────────────────────────


def panel_a_age_binned_retention(ax):
    """Panel A: Age-binned retention at 10K edits."""
    alg = "AlphaEdit"
    target_edits = 10000

    for seed in SEEDS:
        cohorts = load_checkpoint_cohorts(seed, target_edits, alg, batch_size=BATCH_SIZE)
        if cohorts is None:
            continue

        bin_means = []
        for label, idx_range in AGE_BINS:
            effs = [cohorts[i]["efficacy"] for i in idx_range if i in cohorts]
            bin_means.append(np.mean(effs) if effs else np.nan)

        x = np.arange(len(AGE_BINS))
        ax.plot(x, bin_means, color=SEED_COLORS[seed], linewidth=2,
                marker="o", markersize=6, label=f"Seed {seed}")

    ax.set_xticks(range(len(AGE_BINS)))
    ax.set_xticklabels([label for label, _ in AGE_BINS], fontsize=8)
    ax.set_xlabel("Edit-Age Cohort")
    ax.set_ylabel("Efficacy at 10K")
    ax.set_title("(a) Age-Binned Retention at 10K Edits")
    ax.legend(fontsize=8)
    ax.set_ylim(-0.05, 1.05)
    ax.axhline(0.5, color="gray", linestyle=":", alpha=0.4)


def panel_b_first_cohort_trajectory(ax):
    """Panel B: First-1K retention vs total edits, per seed."""
    alg = "AlphaEdit"

    for seed in SEEDS:
        trajectory = []
        for edits in EDIT_POINTS:
            cohorts = load_checkpoint_cohorts(seed, edits, alg, batch_size=BATCH_SIZE)
            if cohorts is None:
                continue
            # First 1K = cohort indices 0-9
            first_1k = [cohorts[i]["efficacy"] for i in range(10) if i in cohorts]
            if first_1k:
                trajectory.append((edits, np.mean(first_1k)))

        if trajectory:
            xs, ys = zip(*trajectory)
            ax.plot(xs, ys, color=SEED_COLORS[seed], linewidth=2,
                    marker="o", markersize=4, label=f"Seed {seed}")

    ax.set_xlabel("Cumulative Edits")
    ax.set_ylabel("First-1K Efficacy")
    ax.set_title("(b) First-Cohort Retention Over Time")
    ax.legend(fontsize=8)
    ax.set_ylim(-0.05, 1.05)
    ax.axhline(0.5, color="gray", linestyle=":", alpha=0.4)


def panel_c_latest_vs_first(ax):
    """Panel C: First-1K vs latest-1K vs latest-100 at 3K and 5K."""
    checkpoints = ["3000_edits", "5000_edits"]
    checkpoint_labels = ["3K", "5K"]
    algs = ["AlphaEdit", "MEMIT-Seq-lp1.0-ld0.0-cache0"]
    alg_labels = ["AlphaEdit", "MEMIT-Seq"]
    ordering = "key_clustered"
    seed = 42

    metrics = ["first_1k", "latest_1k", "latest_100"]
    metric_labels = ["First 1K", "Latest 1K", "Latest 100"]
    metric_colors = ["#E91E63", "#4CAF50", "#FF9800"]

    x_positions = []
    x_labels = []
    group_idx = 0

    for ckpt, ckpt_label in zip(checkpoints, checkpoint_labels):
        for alg, alg_label in zip(algs, alg_labels):
            data = load_matched_ordering_full_eval(seed, ordering, alg)
            if data is None or ckpt not in data:
                group_idx += 1
                continue

            entry = data[ckpt]
            for mi, (metric, mlabel, mcolor) in enumerate(
                zip(metrics, metric_labels, metric_colors)
            ):
                val = entry.get(metric, {}).get("efficacy", np.nan)
                x_pos = group_idx + mi * 0.25
                marker = "o" if "AlphaEdit" in alg else "^"
                ax.scatter(x_pos, val, color=mcolor, marker=marker,
                           s=80, zorder=5, edgecolors="black", linewidths=0.5)

            x_positions.append(group_idx + 0.25)
            x_labels.append(f"{alg_label}\n{ckpt_label}")
            group_idx += 1.2

    ax.set_xticks(x_positions)
    ax.set_xticklabels(x_labels, fontsize=8)
    ax.set_ylabel("Efficacy")
    ax.set_title("(c) Latest vs First Cohort (seed 42)")
    ax.set_ylim(0.4, 1.05)

    # Legend for metric colors
    for mlabel, mcolor in zip(metric_labels, metric_colors):
        ax.scatter([], [], color=mcolor, s=50, label=mlabel)
    ax.legend(loc="lower left", fontsize=7, ncol=3)


def panel_d_order_sensitivity(ax):
    """Panel D: Order sensitivity at 3K and 5K (dot plots)."""
    checkpoints = ["3000_edits", "5000_edits"]
    checkpoint_labels = ["3K", "5K"]
    orderings = ["key_clustered", "key_dispersed"]
    ordering_labels = ["Clustered", "Dispersed"]
    ordering_colors = ["#E91E63", "#2196F3"]
    alg = "AlphaEdit"

    x_idx = 0
    x_ticks = []
    x_tick_labels = []

    for ci, (ckpt, ckpt_label) in enumerate(zip(checkpoints, checkpoint_labels)):
        for seed in SEEDS:
            vals = []
            for oi, (ordering, olabel) in enumerate(zip(orderings, ordering_labels)):
                data = load_matched_ordering_full_eval(seed, ordering, alg)
                if data is None or ckpt not in data:
                    continue
                eff = data[ckpt]["all_facts"]["efficacy"]
                vals.append(eff)
                ax.scatter(x_idx, eff, color=ordering_colors[oi],
                           s=100, zorder=5, edgecolors="black", linewidths=0.5,
                           marker="o")

            # Connect paired points
            if len(vals) == 2:
                ax.plot([x_idx, x_idx], vals, color="gray",
                        linewidth=1, alpha=0.5)
                spread = abs(vals[0] - vals[1])
                ax.annotate(f"\u0394={spread:.3f}",
                            xy=(x_idx + 0.15, np.mean(vals)),
                            fontsize=7, color="gray")

            x_ticks.append(x_idx)
            x_tick_labels.append(f"s{seed}\n{ckpt_label}")
            x_idx += 1

    ax.set_xticks(x_ticks)
    ax.set_xticklabels(x_tick_labels, fontsize=8)
    ax.set_ylabel("Aggregate Efficacy")
    ax.set_title("(d) Order Sensitivity Grows with Horizon")
    ax.set_ylim(0.5, 1.05)

    # Legend
    for olabel, ocolor in zip(ordering_labels, ordering_colors):
        ax.scatter([], [], color=ocolor, s=60, label=olabel)
    ax.legend(loc="lower left", fontsize=8)


# ─── Main ─────────────────────────────────────────────────────────────────────


def generate(output_dir: Path = PAPER_OUTPUT):
    """Generate Figure 3."""
    setup_style()

    fig, axes = plt.subplots(2, 2, figsize=(12, 9))

    panel_a_age_binned_retention(axes[0, 0])
    panel_b_first_cohort_trajectory(axes[0, 1])
    panel_c_latest_vs_first(axes[1, 0])
    panel_d_order_sensitivity(axes[1, 1])

    plt.tight_layout()
    save_figure(fig, "fig3_plasticity", output_dir)


def main():
    parser = argparse.ArgumentParser(
        description="Generate Figure 3: Local plasticity vs global retention")
    parser.add_argument("--output-dir", type=Path, default=PAPER_OUTPUT)
    args = parser.parse_args()
    generate(args.output_dir)


if __name__ == "__main__":
    main()

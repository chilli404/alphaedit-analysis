"""Figure 2 — The anatomy of forgetting.

Question answered: What does long-horizon failure look like internally?

Panels:
  A. Cohort-retention heatmap (x=checkpoint, y=cohort, color=efficacy)
  B. First-1K retention trajectory (seeds 42 and 2024 separately)
  C. First-1K, middle cohort, latest-1K curves (representative seed)
  D. Order sensitivity CV at 3K and 7K

Usage:
    uv run python -m analysis.fig2_forgetting
    uv run python -m analysis.fig2_forgetting --output-dir results/figures/paper
"""

import argparse
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable

from analysis.style import (
    ALGO_COLORS, SEED_COLORS, setup_style, save_figure, PAPER_OUTPUT,
)
from analysis.loaders import (
    load_checkpoint_cohorts,
    load_checkpoint_metrics,
    load_comparison_ordered,
)

# ─── Configuration ────────────────────────────────────────────────────────────

SEEDS = [42, 2024]
EDIT_POINTS = [2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000, 10000]
BATCH_SIZE = 100


# ─── Panel Functions ──────────────────────────────────────────────────────────


def panel_a_cohort_heatmap(ax):
    """Panel A: Cohort-retention heatmap for AlphaEdit seed 42."""
    seed = 42
    alg = "AlphaEdit"

    # Build matrix: rows = cohorts (bands of 1000 edits), cols = checkpoints
    band_size = 10  # 10 batches of 100 = 1000 edits per band
    max_bands = 10  # Up to 10K edits
    n_checkpoints = len(EDIT_POINTS)

    matrix = np.full((max_bands, n_checkpoints), np.nan)

    for col, edits in enumerate(EDIT_POINTS):
        cohorts = load_checkpoint_cohorts(seed, edits, alg, batch_size=BATCH_SIZE)
        if cohorts is None:
            continue

        # Group cohorts into bands of 1000 edits
        for band_idx in range(max_bands):
            band_start = band_idx * band_size
            band_end = band_start + band_size
            band_effs = []
            for cohort_idx in range(band_start, band_end):
                if cohort_idx in cohorts:
                    band_effs.append(cohorts[cohort_idx]["efficacy"])

            # Only fill if this band has been edited by this checkpoint
            if band_effs and (band_idx + 1) * 1000 <= edits:
                matrix[band_idx, col] = np.mean(band_effs)

    im = ax.imshow(matrix, aspect="auto", cmap="RdYlGn", vmin=0, vmax=1,
                   origin="lower", interpolation="nearest")

    ax.set_xticks(range(n_checkpoints))
    ax.set_xticklabels([f"{e // 1000}K" for e in EDIT_POINTS], fontsize=8)
    ax.set_yticks(range(max_bands))
    ax.set_yticklabels([f"{i}K-{i+1}K" for i in range(max_bands)], fontsize=8)
    ax.set_xlabel("Evaluation Checkpoint")
    ax.set_ylabel("Edit Cohort (insertion time)")
    ax.set_title("(A) Cohort Retention Heatmap (AlphaEdit, seed 42)")

    plt.colorbar(im, ax=ax, label="Efficacy", shrink=0.8)


def panel_b_first_1k_trajectory(ax):
    """Panel B: First-1K retention vs total edits, per seed."""
    alg = "AlphaEdit"

    for seed in SEEDS:
        trajectory = []
        for edits in EDIT_POINTS:
            cohorts = load_checkpoint_cohorts(seed, edits, alg, batch_size=BATCH_SIZE)
            if cohorts is None:
                continue
            # First 1K = cohort indices 0-9 (batches 0-9, 100 edits each)
            first_1k = []
            for idx in range(10):
                if idx in cohorts:
                    first_1k.append(cohorts[idx]["efficacy"])
            if first_1k:
                trajectory.append((edits, np.mean(first_1k)))

        if trajectory:
            xs, ys = zip(*trajectory)
            ax.plot(xs, ys, color=SEED_COLORS[seed], linewidth=2,
                    marker="o", markersize=4, label=f"Seed {seed}")

    ax.set_xlabel("Total Edits")
    ax.set_ylabel("First-1K Efficacy")
    ax.set_title("(B) First-1K Retention Trajectory")
    ax.legend()
    ax.set_ylim(-0.05, 1.05)
    ax.axhline(0.5, color="gray", linestyle=":", alpha=0.4)


def panel_c_cohort_comparison(ax):
    """Panel C: First-1K, middle, latest-1K curves for seed 42."""
    seed = 42
    alg = "AlphaEdit"

    first_1k, middle, latest_1k = [], [], []

    for edits in EDIT_POINTS:
        cohorts = load_checkpoint_cohorts(seed, edits, alg, batch_size=BATCH_SIZE)
        if cohorts is None:
            continue

        n_total_batches = edits // BATCH_SIZE

        # First 1K (cohorts 0-9)
        f1k = [cohorts[i]["efficacy"] for i in range(10) if i in cohorts]
        if f1k:
            first_1k.append((edits, np.mean(f1k)))

        # Latest 1K (last 10 cohorts)
        l1k_start = max(0, n_total_batches - 10)
        l1k = [cohorts[i]["efficacy"] for i in range(l1k_start, n_total_batches) if i in cohorts]
        if l1k:
            latest_1k.append((edits, np.mean(l1k)))

        # Middle cohort (middle 1K)
        mid_start = n_total_batches // 2 - 5
        mid_end = mid_start + 10
        mid = [cohorts[i]["efficacy"] for i in range(mid_start, mid_end) if i in cohorts]
        if mid:
            middle.append((edits, np.mean(mid)))

    # Plot
    for data, label, color in [
        (first_1k, "First 1K", "#E91E63"),
        (middle, "Middle 1K", "#9C27B0"),
        (latest_1k, "Latest 1K", "#4CAF50"),
    ]:
        if data:
            xs, ys = zip(*data)
            ax.plot(xs, ys, linewidth=2, marker="o", markersize=4,
                    label=label, color=color)

    ax.set_xlabel("Total Edits")
    ax.set_ylabel("Efficacy")
    ax.set_title("(C) Cohort-Age Retention (seed 42)")
    ax.legend()
    ax.set_ylim(-0.05, 1.05)
    ax.axhline(0.5, color="gray", linestyle=":", alpha=0.4)


def panel_d_order_sensitivity(ax):
    """Panel D: Order sensitivity (CV) at 3K and 7K."""
    edit_levels = [3000, 7000]
    seed = 42

    ae_data = {}
    for edits in edit_levels:
        orders = load_comparison_ordered(seed, edits)
        ae_orders = [o for o in orders if o["algorithm"] == "AlphaEdit"]
        if ae_orders:
            ae_data[edits] = ae_orders

    if not ae_data:
        ax.text(0.5, 0.5, "No order sensitivity data", transform=ax.transAxes,
                ha="center", va="center", fontsize=11)
        ax.set_title("(D) Order Sensitivity vs Scale")
        return

    metrics = ["efficacy", "paraphrase", "neighborhood"]
    metric_labels = ["Efficacy", "Paraphrase", "Specificity"]

    x = np.arange(len(metrics))
    width = 0.35
    colors = ["#2196F3", "#E91E63"]

    for i, edits in enumerate(edit_levels):
        if edits not in ae_data:
            continue
        orders = ae_data[edits]
        cvs = []
        for m in metrics:
            vals = [o[m] for o in orders if m in o]
            if vals and np.mean(vals) > 0:
                cvs.append(np.std(vals) / np.mean(vals) * 100)
            else:
                cvs.append(0)

        offset = (i - 0.5) * width
        ax.bar(x + offset, cvs, width, label=f"{edits // 1000}K edits",
               color=colors[i], alpha=0.8, edgecolor="black", linewidth=0.5)

    ax.set_xticks(x)
    ax.set_xticklabels(metric_labels)
    ax.set_ylabel("Coefficient of Variation (%)")
    ax.set_title("(D) Order Sensitivity vs Scale")
    ax.legend()


# ─── Main ─────────────────────────────────────────────────────────────────────


def generate(output_dir: Path = PAPER_OUTPUT):
    """Generate Figure 2."""
    setup_style()

    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    fig.suptitle(
        "Figure 2: Forgetting is Age-Dependent and Order-Sensitive at Scale",
        fontsize=13, y=0.98,
    )

    panel_a_cohort_heatmap(axes[0, 0])
    panel_b_first_1k_trajectory(axes[0, 1])
    panel_c_cohort_comparison(axes[1, 0])
    panel_d_order_sensitivity(axes[1, 1])

    plt.tight_layout()
    save_figure(fig, "fig2_forgetting", output_dir)


def main():
    parser = argparse.ArgumentParser(description="Generate Figure 2: Anatomy of forgetting")
    parser.add_argument("--output-dir", type=Path, default=PAPER_OUTPUT)
    args = parser.parse_args()
    generate(args.output_dir)


if __name__ == "__main__":
    main()

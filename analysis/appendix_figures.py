"""Appendix figures — supplementary visualizations.

Produces:
  A1. Full per-seed failure curves (all seeds × all orderings)
  A3. Full cohort heatmaps (one per seed/trajectory)
  A8. SeqReg mechanism trajectory (cache size, disruption ratio, update norm)

Usage:
    uv run python -m analysis.appendix_figures
    uv run python -m analysis.appendix_figures --output-dir results/figures/appendix
"""

import argparse
from collections import defaultdict
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from analysis.style import (
    ALGO_COLORS, SEED_COLORS, setup_style, save_figure, APPENDIX_OUTPUT,
)
from analysis.loaders import (
    load_checkpoint_metrics,
    load_checkpoint_cohorts,
    load_seqreg_logs,
)

# ─── Configuration ────────────────────────────────────────────────────────────

SEEDS = [42, 2024, 137]
EDIT_POINTS = [2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000, 10000]
BATCH_SIZE = 100


# ─── A1: Full Per-Seed Failure Curves ────────────────────────────────────────


def figure_a1(output_dir: Path):
    """A1: Full per-seed failure curves with all available data."""
    setup_style()
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    metrics = ["efficacy", "paraphrase", "neighborhood"]
    metric_labels = ["Efficacy", "Paraphrase", "Specificity"]

    for col, (metric, label) in enumerate(zip(metrics, metric_labels)):
        ax = axes[col]

        for alg in ("AlphaEdit", "MEMIT"):
            for seed in SEEDS:
                curve = []
                for edits in EDIT_POINTS:
                    m = load_checkpoint_metrics(seed, edits, alg)
                    if m and metric in m:
                        curve.append((edits, m[metric]))
                if curve:
                    xs, ys = zip(*curve)
                    ax.plot(xs, ys, color=SEED_COLORS.get(seed, "gray"),
                            linewidth=1.5,
                            linestyle="-" if alg == "AlphaEdit" else "--",
                            marker="o", markersize=3,
                            label=f"{alg} s{seed}",
                            alpha=0.8)

        ax.set_xlabel("Total Edits")
        ax.set_ylabel(label)
        ax.set_title(label)
        ax.set_ylim(-0.05, 1.05)
        ax.axhline(0.5, color="gray", linestyle=":", alpha=0.3)
        if col == 0:
            ax.legend(fontsize=7, ncol=2, loc="lower left")

    plt.tight_layout()
    save_figure(fig, "a1_perseed_failure_curves", output_dir)


# ─── A3: Full Cohort Heatmaps ────────────────────────────────────────────────


def figure_a3(output_dir: Path):
    """A3: Full cohort heatmaps for each seed."""
    setup_style()

    seeds_with_data = []
    for seed in SEEDS:
        # Check if any cohort data exists
        test = load_checkpoint_cohorts(seed, EDIT_POINTS[0], "AlphaEdit", BATCH_SIZE)
        if test:
            seeds_with_data.append(seed)

    if not seeds_with_data:
        print("  [A3] SKIP: no cohort data available")
        return

    n_seeds = len(seeds_with_data)
    fig, axes = plt.subplots(1, n_seeds, figsize=(6 * n_seeds, 5))
    if n_seeds == 1:
        axes = [axes]

    band_size = 10  # 10 batches × 100 = 1000 edits per band
    max_bands = 10

    for idx, seed in enumerate(seeds_with_data):
        ax = axes[idx]
        matrix = np.full((max_bands, len(EDIT_POINTS)), np.nan)

        for col, edits in enumerate(EDIT_POINTS):
            cohorts = load_checkpoint_cohorts(seed, edits, "AlphaEdit", BATCH_SIZE)
            if cohorts is None:
                continue

            for band_idx in range(max_bands):
                band_start = band_idx * band_size
                band_end = band_start + band_size
                band_effs = []
                for cohort_idx in range(band_start, band_end):
                    if cohort_idx in cohorts:
                        band_effs.append(cohorts[cohort_idx]["efficacy"])
                if band_effs and (band_idx + 1) * 1000 <= edits:
                    matrix[band_idx, col] = np.mean(band_effs)

        im = ax.imshow(matrix, aspect="auto", cmap="RdYlGn",
                       vmin=0, vmax=1, origin="lower", interpolation="nearest")

        ax.set_xticks(range(len(EDIT_POINTS)))
        ax.set_xticklabels([f"{e // 1000}K" for e in EDIT_POINTS], fontsize=7)
        ax.set_yticks(range(max_bands))
        ax.set_yticklabels([f"{i}K" for i in range(max_bands)], fontsize=8)
        ax.set_xlabel("Checkpoint")
        ax.set_ylabel("Cohort Origin")
        ax.set_title(f"Seed {seed}")

    plt.colorbar(im, ax=axes[-1], label="Efficacy", shrink=0.8)
    fig.suptitle("A3: Cohort Retention Heatmaps (AlphaEdit)", fontsize=13, y=1.02)
    plt.tight_layout()
    save_figure(fig, "a3_cohort_heatmaps", output_dir)


# ─── A8: SeqReg Mechanism Trajectory ─────────────────────────────────────────


def figure_a8(output_dir: Path):
    """A8: MEMIT+SeqReg mechanism trajectory (cache size, norms, ratios)."""
    setup_style()

    records = load_seqreg_logs(42, 1.0, 1.0)
    if not records:
        print("  [A8] SKIP: no SeqReg log data")
        return

    # Aggregate by batch (records are per-layer)
    by_batch = defaultdict(list)
    for r in records:
        by_batch[r["batch"]].append(r)

    batches = sorted(by_batch.keys())
    cache_sizes = []
    mean_upd_norms = []
    mean_disruption = []
    mean_reg_ratio = []

    for batch in batches:
        layers = by_batch[batch]
        cache_sizes.append(layers[0].get("cache_keys", 0))
        mean_upd_norms.append(np.mean([l["upd_norm"] for l in layers]))
        # Disruption ratio: ||ΔW @ K_prev|| / ||ΔW||
        disruptions = []
        for l in layers:
            if l["upd_norm"] > 0:
                disruptions.append(l["dw_kprev_norm"] / l["upd_norm"])
        mean_disruption.append(np.mean(disruptions) if disruptions else 0)
        # Reg/base ratio: ||K_prev @ K_prev^T|| / ||base LHS||
        ratios = []
        for l in layers:
            if l.get("base_lhs_norm", 0) > 0:
                ratios.append(l.get("kpkp_norm", 0) / l["base_lhs_norm"])
        mean_reg_ratio.append(np.mean(ratios) if ratios else 0)

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle("A8: MEMIT+SeqReg Mechanism Trajectory (seed 42, λ_prev=1, λ_delta=1)",
                 fontsize=12, y=0.98)

    # Panel 1: Cache size
    ax = axes[0, 0]
    ax.plot(batches, cache_sizes, linewidth=2, color="#4CAF50", marker="o", markersize=3)
    ax.set_xlabel("Batch")
    ax.set_ylabel("Cache Keys")
    ax.set_title("Cache Size vs Edit Batch")

    # Panel 2: Update norm
    ax = axes[0, 1]
    ax.plot(batches, mean_upd_norms, linewidth=2, color="#2196F3", marker="o", markersize=3)
    ax.set_xlabel("Batch")
    ax.set_ylabel("Mean ||ΔW|| (across layers)")
    ax.set_title("Update Norm vs Edit Batch")

    # Panel 3: Disruption ratio
    ax = axes[1, 0]
    ax.plot(batches, mean_disruption, linewidth=2, color="#E91E63", marker="o", markersize=3)
    ax.set_xlabel("Batch")
    ax.set_ylabel("||ΔW @ K_prev|| / ||ΔW||")
    ax.set_title("Disruption Ratio vs Edit Batch")

    # Panel 4: Regularization/base ratio
    ax = axes[1, 1]
    ax.plot(batches, mean_reg_ratio, linewidth=2, color="#FF9800", marker="o", markersize=3)
    ax.set_xlabel("Batch")
    ax.set_ylabel("||K_prev K_prev^T|| / ||base LHS||")
    ax.set_title("Regularization Strength Ratio")

    plt.tight_layout()
    save_figure(fig, "a8_seqreg_mechanism", output_dir)


# ─── Main ─────────────────────────────────────────────────────────────────────


def generate(output_dir: Path = APPENDIX_OUTPUT):
    """Generate all appendix figures."""
    output_dir.mkdir(parents=True, exist_ok=True)
    print("Generating appendix figures...")
    figure_a1(output_dir)
    figure_a3(output_dir)
    figure_a8(output_dir)


def main():
    parser = argparse.ArgumentParser(description="Generate appendix figures (A1, A3, A8)")
    parser.add_argument("--output-dir", type=Path, default=APPENDIX_OUTPUT)
    args = parser.parse_args()
    generate(args.output_dir)


if __name__ == "__main__":
    main()

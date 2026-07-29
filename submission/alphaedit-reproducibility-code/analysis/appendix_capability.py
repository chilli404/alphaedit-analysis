"""Appendix figure: Capability probe (perplexity + MMLU vs edit count).

Produces one figure with 3 panels:
  A. Factual efficacy (first-cohort retention) vs edit count
  B. Corpus-level WikiText-103 perplexity vs edit count
  C. Four-subject MMLU accuracy vs edit count

Each panel shows per-algorithm traces with per-seed thin lines and an
aggregate mean. Supports AlphaEdit, MEMIT, and MEMIT-Seq variants.

Usage:
    uv run python -m analysis.appendix_capability
    uv run python -m analysis.appendix_capability --output-dir results/figures/appendix
    uv run python -m analysis.appendix_capability --algorithms AlphaEdit MEMIT-Seq-lp1.0-ld0.0-cache0
"""

import argparse
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from analysis.style import (
    ALGO_COLORS, SEED_COLORS, setup_style, save_figure,
    APPENDIX_OUTPUT,
)
from analysis.loaders import (
    load_capability_probe,
    load_checkpoint_metrics,
)

# ─── Configuration ────────────────────────────────────────────────────────────

SEEDS = [42, 137, 2024]

# Algorithms to plot (in display order). Each entry is (alg_name, display_label, color).
DEFAULT_ALGORITHMS = [
    "AlphaEdit",
    "MEMIT",
    "MEMIT-Seq-lp1.0-ld0.0-cache0",
]

# Display labels and colors for algorithm variants
ALG_DISPLAY = {
    "AlphaEdit": ("AlphaEdit", ALGO_COLORS.get("AlphaEdit", "#2196F3")),
    "MEMIT": ("MEMIT", ALGO_COLORS.get("MEMIT", "#FF9800")),
    "MEMIT-Seq-lp1.0-ld0.0-cache0": ("MEMIT+SeqReg", ALGO_COLORS.get("MEMIT+SeqReg", "#4CAF50")),
}

# Edit points for factual efficacy (from failure curve)
EDIT_POINTS = [2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000, 10000]


# ─── Helpers ──────────────────────────────────────────────────────────────────


def _get_alg_display(alg: str):
    """Get display label and color for an algorithm."""
    if alg in ALG_DISPLAY:
        return ALG_DISPLAY[alg]
    # Fallback for unknown MEMIT-Seq variants
    if alg.startswith("MEMIT-Seq"):
        return (alg.replace("MEMIT-Seq-", "SeqReg "), "#4CAF50")
    return (alg, "#607D8B")


def _get_efficacy_curves(alg: str):
    """Load first-cohort efficacy from failure curve checkpoints."""
    curves = {}
    for seed in SEEDS:
        points = []
        for edits in EDIT_POINTS:
            m = load_checkpoint_metrics(seed, edits, alg)
            if m and "efficacy" in m:
                points.append((edits, m["efficacy"]))
        if points:
            curves[seed] = points
    return curves


def _get_probe_curves(alg: str):
    """Load capability probe data (perplexity + MMLU) per seed."""
    ppl_curves = {}
    mmlu_curves = {}
    for seed in SEEDS:
        records = load_capability_probe(seed, alg)
        if not records:
            continue
        ppl_points = []
        mmlu_points = []
        for r in records:
            ec = r.get("edit_count", 0)
            ppl = r.get("mean_perplexity")
            if ppl is not None and not np.isnan(ppl):
                ppl_points.append((ec, ppl))
            mmlu = r.get("mmlu_accuracy")
            if mmlu is not None and not np.isnan(mmlu):
                mmlu_points.append((ec, mmlu))
        if ppl_points:
            ppl_curves[seed] = ppl_points
        if mmlu_points:
            mmlu_curves[seed] = mmlu_points
    return ppl_curves, mmlu_curves


def _compute_aggregate(curves):
    """Compute mean ± std across seeds at shared x-values."""
    if not curves:
        return None
    # Collect all x values that appear in at least 2 seeds
    from collections import Counter
    x_counts = Counter()
    for points in curves.values():
        for x, _ in points:
            x_counts[x] += 1
    # For single-seed algorithms, accept x values appearing at least once
    min_count = 2 if len(curves) >= 2 else 1
    shared_xs = sorted(x for x, c in x_counts.items() if c >= min_count)
    if not shared_xs:
        return None

    means = []
    stds = []
    for x in shared_xs:
        vals = []
        for points in curves.values():
            for px, py in points:
                if px == x:
                    vals.append(py)
                    break
        means.append(np.mean(vals))
        stds.append(np.std(vals) if len(vals) > 1 else 0.0)
    return shared_xs, means, stds


# ─── Figure ───────────────────────────────────────────────────────────────────


def figure_capability_probe(output_dir: Path, algorithms: list[str] | None = None):
    """Generate 3-panel capability probe appendix figure with multi-algorithm support."""
    setup_style()

    if algorithms is None:
        algorithms = DEFAULT_ALGORITHMS

    # Collect data per algorithm
    all_efficacy = {}  # alg -> {seed -> points}
    all_ppl = {}       # alg -> {seed -> points}
    all_mmlu = {}      # alg -> {seed -> points}

    for alg in algorithms:
        efficacy_curves = _get_efficacy_curves(alg)
        ppl_curves, mmlu_curves = _get_probe_curves(alg)

        if efficacy_curves:
            all_efficacy[alg] = efficacy_curves
        if ppl_curves:
            all_ppl[alg] = ppl_curves
        if mmlu_curves:
            all_mmlu[alg] = mmlu_curves

    has_efficacy = bool(all_efficacy)
    has_ppl = bool(all_ppl)
    has_mmlu = bool(all_mmlu)

    if not has_efficacy and not has_ppl and not has_mmlu:
        print("  [SKIP] No capability probe data available.")
        print("         Run: bash scripts/run_capability_probe_offline.sh <seed> all")
        return

    n_panels = sum([has_efficacy, has_ppl, has_mmlu])
    fig, axes = plt.subplots(1, n_panels, figsize=(5 * n_panels, 4.5))
    if n_panels == 1:
        axes = [axes]

    panel_idx = 0

    # Panel A: Factual efficacy
    if has_efficacy:
        ax = axes[panel_idx]
        for alg, curves in all_efficacy.items():
            label, color = _get_alg_display(alg)
            # Per-seed thin lines
            for seed, points in curves.items():
                xs, ys = zip(*points)
                ax.plot(xs, ys, color=color,
                        linewidth=0.8, marker="o", markersize=2,
                        alpha=0.35)
            # Aggregate mean
            agg = _compute_aggregate(curves)
            if agg:
                xs, means, stds = agg
                ax.plot(xs, means, color=color, linewidth=2.5,
                        label=label, zorder=5)
                if any(s > 0 for s in stds):
                    ax.fill_between(xs,
                                    [m - s for m, s in zip(means, stds)],
                                    [m + s for m, s in zip(means, stds)],
                                    color=color, alpha=0.1)
        ax.set_xlabel("Total Edits")
        ax.set_ylabel("Factual Efficacy")
        ax.set_title("A. Factual Retention")
        ax.set_ylim(-0.05, 1.05)
        ax.axhline(0.5, color="gray", linestyle=":", alpha=0.3)
        ax.legend(fontsize=7, loc="lower left")
        panel_idx += 1

    # Panel B: Perplexity
    if has_ppl:
        ax = axes[panel_idx]
        for alg, curves in all_ppl.items():
            label, color = _get_alg_display(alg)
            # Per-seed thin lines
            for seed, points in curves.items():
                xs, ys = zip(*points)
                ax.plot(xs, ys, color=color,
                        linewidth=0.8, marker="s", markersize=2,
                        alpha=0.35)
            # Aggregate mean
            agg = _compute_aggregate(curves)
            if agg:
                xs, means, stds = agg
                ax.plot(xs, means, color=color, linewidth=2.5,
                        label=label, zorder=5)
                if any(s > 0 for s in stds):
                    ax.fill_between(xs,
                                    [m - s for m, s in zip(means, stds)],
                                    [m + s for m, s in zip(means, stds)],
                                    color=color, alpha=0.1)
        ax.set_xlabel("Total Edits")
        ax.set_ylabel("Perplexity (WikiText-2)")
        ax.set_title("B. Corpus Perplexity")
        ax.set_yscale("log")
        ax.legend(fontsize=7, loc="upper left")
        panel_idx += 1

    # Panel C: MMLU
    if has_mmlu:
        ax = axes[panel_idx]
        for alg, curves in all_mmlu.items():
            label, color = _get_alg_display(alg)
            # Per-seed thin lines
            for seed, points in curves.items():
                xs, ys = zip(*points)
                ax.plot(xs, ys, color=color,
                        linewidth=0.8, marker="^", markersize=2,
                        alpha=0.35)
            # Aggregate mean
            agg = _compute_aggregate(curves)
            if agg:
                xs, means, stds = agg
                ax.plot(xs, means, color=color, linewidth=2.5,
                        label=label, zorder=5)
                if any(s > 0 for s in stds):
                    ax.fill_between(xs,
                                    [m - s for m, s in zip(means, stds)],
                                    [m + s for m, s in zip(means, stds)],
                                    color=color, alpha=0.1)
        ax.set_xlabel("Total Edits")
        ax.set_ylabel("Accuracy (4-subject MMLU)")
        ax.set_title("C. MMLU Accuracy")
        ax.set_ylim(-0.05, 1.05)
        ax.axhline(0.25, color="gray", linestyle=":", alpha=0.3, label="Random")
        ax.legend(fontsize=7, loc="lower left")
        panel_idx += 1

    plt.tight_layout()
    save_figure(fig, "a_capability_probe", output_dir)


# ─── CLI ──────────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(description="Generate capability probe appendix figure")
    parser.add_argument("--output-dir", type=Path, default=APPENDIX_OUTPUT)
    parser.add_argument(
        "--algorithms", nargs="+", default=None,
        help="Algorithms to include (default: AlphaEdit MEMIT MEMIT-Seq-lp1.0-ld0.0-cache0)"
    )
    args = parser.parse_args()
    figure_capability_probe(args.output_dir, args.algorithms)


if __name__ == "__main__":
    main()

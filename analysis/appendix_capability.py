"""Appendix figure: Capability probe (perplexity + MMLU vs edit count).

Produces one figure with 3 panels:
  A. Factual efficacy (first-cohort retention) vs edit count
  B. Corpus-level WikiText-103 perplexity vs edit count
  C. Four-subject MMLU accuracy vs edit count

Each panel shows per-seed lines (42, 137, 2024) and an aggregate mean.

Claim supported (appendix only):
  Factual retention degrades substantially before broad language-model
  capability collapses.

Usage:
    uv run python -m analysis.appendix_capability
    uv run python -m analysis.appendix_capability --output-dir results/figures/appendix
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
ALG = "AlphaEdit"
# Edit points for factual efficacy (from failure curve)
EDIT_POINTS = [2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000, 10000]


# ─── Helpers ──────────────────────────────────────────────────────────────────


def _get_efficacy_curves():
    """Load first-cohort efficacy from failure curve checkpoints."""
    curves = {}
    for seed in SEEDS:
        points = []
        for edits in EDIT_POINTS:
            m = load_checkpoint_metrics(seed, edits, ALG)
            if m and "efficacy" in m:
                points.append((edits, m["efficacy"]))
        if points:
            curves[seed] = points
    return curves


def _get_probe_curves():
    """Load capability probe data (perplexity + MMLU) per seed."""
    ppl_curves = {}
    mmlu_curves = {}
    for seed in SEEDS:
        records = load_capability_probe(seed, ALG)
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
    shared_xs = sorted(x for x, c in x_counts.items() if c >= 2)
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
        stds.append(np.std(vals))
    return shared_xs, means, stds


# ─── Figure ───────────────────────────────────────────────────────────────────


def figure_capability_probe(output_dir: Path):
    """Generate 3-panel capability probe appendix figure."""
    setup_style()

    efficacy_curves = _get_efficacy_curves()
    ppl_curves, mmlu_curves = _get_probe_curves()

    # Check we have data
    has_efficacy = bool(efficacy_curves)
    has_ppl = bool(ppl_curves)
    has_mmlu = bool(mmlu_curves)

    if not has_efficacy and not has_ppl and not has_mmlu:
        print("  [SKIP] No capability probe data available.")
        print("         Run: python src/mechanism/capability_probe_offline.py --seed 42 --alg_name AlphaEdit")
        return

    n_panels = sum([has_efficacy, has_ppl, has_mmlu])
    fig, axes = plt.subplots(1, n_panels, figsize=(5 * n_panels, 4.5))
    if n_panels == 1:
        axes = [axes]

    panel_idx = 0

    # Panel A: Factual efficacy
    if has_efficacy:
        ax = axes[panel_idx]
        for seed, points in efficacy_curves.items():
            xs, ys = zip(*points)
            ax.plot(xs, ys, color=SEED_COLORS.get(seed, "gray"),
                    linewidth=1.2, marker="o", markersize=3,
                    alpha=0.6, label=f"seed {seed}")
        agg = _compute_aggregate(efficacy_curves)
        if agg:
            xs, means, stds = agg
            ax.plot(xs, means, color=ALGO_COLORS[ALG], linewidth=2.5,
                    label="Mean", zorder=5)
            ax.fill_between(xs,
                            [m - s for m, s in zip(means, stds)],
                            [m + s for m, s in zip(means, stds)],
                            color=ALGO_COLORS[ALG], alpha=0.15)
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
        for seed, points in ppl_curves.items():
            xs, ys = zip(*points)
            ax.plot(xs, ys, color=SEED_COLORS.get(seed, "gray"),
                    linewidth=1.2, marker="s", markersize=3,
                    alpha=0.6, label=f"seed {seed}")
        agg = _compute_aggregate(ppl_curves)
        if agg:
            xs, means, stds = agg
            ax.plot(xs, means, color="black", linewidth=2.5,
                    label="Mean", zorder=5)
            ax.fill_between(xs,
                            [m - s for m, s in zip(means, stds)],
                            [m + s for m, s in zip(means, stds)],
                            color="black", alpha=0.1)
        ax.set_xlabel("Total Edits")
        ax.set_ylabel("Perplexity (WikiText-103)")
        ax.set_title("B. Corpus Perplexity")
        ax.legend(fontsize=7, loc="upper left")
        panel_idx += 1

    # Panel C: MMLU
    if has_mmlu:
        ax = axes[panel_idx]
        for seed, points in mmlu_curves.items():
            xs, ys = zip(*points)
            ax.plot(xs, ys, color=SEED_COLORS.get(seed, "gray"),
                    linewidth=1.2, marker="^", markersize=3,
                    alpha=0.6, label=f"seed {seed}")
        agg = _compute_aggregate(mmlu_curves)
        if agg:
            xs, means, stds = agg
            ax.plot(xs, means, color="black", linewidth=2.5,
                    label="Mean", zorder=5)
            ax.fill_between(xs,
                            [m - s for m, s in zip(means, stds)],
                            [m + s for m, s in zip(means, stds)],
                            color="black", alpha=0.1)
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
    args = parser.parse_args()
    figure_capability_probe(args.output_dir)


if __name__ == "__main__":
    main()

"""Figure 2 — Reproduction and long-horizon boundary.

Question answered: Does AlphaEdit reproduce its claimed advantage, and where
does that advantage begin to break down?

Panels:
  A. Standard-scale reproduction (AlphaEdit vs MEMIT at 2K, 5 seeds, bar chart)
  B. Long-horizon efficacy (2K → 10K, per-seed traces + mean)
  C. Paraphrase and locality trajectories (AlphaEdit mean across seeds)

Usage:
    uv run python -m analysis.fig2_reproduction
    uv run python -m analysis.fig2_reproduction --output-dir results/figures/paper
"""

import argparse
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from analysis.style import (
    ALGO_COLORS, SEED_COLORS, setup_style, save_figure, PAPER_OUTPUT,
)
from analysis.loaders import load_checkpoint_metrics, load_mve_metrics

# ─── Configuration ────────────────────────────────────────────────────────────

SEEDS = [42, 2024, 137]
MVE_SEEDS = [42, 2024, 137, 7, 99]
EDIT_POINTS = [2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000, 10000]
ALGOS = ["AlphaEdit", "MEMIT"]


# ─── Helpers ──────────────────────────────────────────────────────────────────


def _collect_curves(metric: str, alg: str):
    """Collect per-seed curves for a metric/algorithm pair."""
    seed_curves = {}
    for seed in SEEDS:
        curve = []
        for edits in EDIT_POINTS:
            m = load_checkpoint_metrics(seed, edits, alg)
            if m is not None and metric in m:
                curve.append((edits, m[metric]))
        if curve:
            seed_curves[seed] = curve
    return seed_curves


def _plot_algo_curves(ax, metric: str, alg: str, show_individual=True):
    """Plot individual seed traces (thin) and mean curve (thick) for one algo."""
    seed_curves = _collect_curves(metric, alg)
    if not seed_curves:
        return

    color = ALGO_COLORS[alg]

    # Individual seeds (thin, dashed)
    if show_individual:
        for seed, curve in seed_curves.items():
            xs, ys = zip(*curve)
            ax.plot(xs, ys, color=SEED_COLORS.get(seed, color),
                    alpha=0.3, linewidth=1, linestyle="--")

    # Mean curve (thick)
    all_edits = sorted(set(e for c in seed_curves.values() for e, _ in c))
    mean_vals = []
    for e in all_edits:
        vals = [v for curve in seed_curves.values() for x, v in curve if x == e]
        if vals:
            mean_vals.append((e, np.mean(vals), np.std(vals)))

    if mean_vals:
        xs, ys, stds = zip(*mean_vals)
        ax.plot(xs, ys, color=color, linewidth=2.5, label=alg,
                marker="o", markersize=4)
        ax.fill_between(xs, np.array(ys) - np.array(stds),
                        np.array(ys) + np.array(stds),
                        color=color, alpha=0.1)


# ─── Panel Functions ──────────────────────────────────────────────────────────


def panel_a_reproduction(ax):
    """Panel A: Standard-scale reproduction at 2K (5 seeds, bar chart)."""
    metrics_list = ["efficacy", "paraphrase", "neighborhood"]
    metric_labels = ["Efficacy", "Paraphrase", "Specificity"]
    x = np.arange(len(metrics_list))
    width = 0.35

    for i, (alg, mve_exp) in enumerate([
        ("AlphaEdit", "mve1_alphaedit_mcf"),
        ("MEMIT", "mve2_memit_mcf"),
    ]):
        seed_values = {m: [] for m in metrics_list}
        for seed in MVE_SEEDS:
            m = load_mve_metrics(mve_exp, seed, alg)
            if m is None:
                m = load_checkpoint_metrics(seed, 2000, alg)
            if m:
                for metric in metrics_list:
                    if metric in m:
                        seed_values[metric].append(m[metric])

        means = [np.mean(seed_values[m]) if seed_values[m] else 0
                 for m in metrics_list]
        stds = [np.std(seed_values[m]) if seed_values[m] else 0
                for m in metrics_list]

        offset = (i - 0.5) * width
        color = ALGO_COLORS[alg]
        ax.bar(x + offset, means, width, yerr=stds, label=alg,
               color=color, alpha=0.8, edgecolor="black", linewidth=0.5,
               capsize=3)

    ax.set_xticks(x)
    ax.set_xticklabels(metric_labels)
    ax.set_ylabel("Score")
    ax.set_title("(A) Reproduction at 2K Edits (5 seeds)")
    ax.legend(loc="upper right")
    ax.set_ylim(0, 1.1)
    ax.axhline(0.5, color="gray", linestyle=":", alpha=0.3)


def panel_b_long_horizon(ax):
    """Panel B: Long-horizon efficacy with individual seed traces."""
    for alg in ALGOS:
        _plot_algo_curves(ax, "efficacy", alg, show_individual=True)

    # Shaded region for trajectory sensitivity
    ax.axvspan(7000, 8000, alpha=0.08, color="red",
               label="Increasing trajectory sensitivity")

    ax.set_xlabel("Cumulative Edits")
    ax.set_ylabel("Aggregate Efficacy")
    ax.set_title("(B) Long-Horizon Failure Curve (2K\u201310K)")
    ax.legend(loc="lower left", fontsize=8)
    ax.set_ylim(-0.05, 1.05)
    ax.axhline(0.5, color="gray", linestyle=":", alpha=0.4)


def panel_c_paraphrase_locality(ax):
    """Panel C: Paraphrase and locality trajectories (AlphaEdit, mean)."""
    alg = "AlphaEdit"

    for metric, label, ls in [
        ("paraphrase", "Paraphrase", "-"),
        ("neighborhood_prob", "Neighborhood P(new)", "--"),
    ]:
        seed_curves = _collect_curves(metric, alg)
        if not seed_curves:
            continue

        all_edits = sorted(set(e for c in seed_curves.values() for e, _ in c))
        mean_vals = []
        for e in all_edits:
            vals = [v for curve in seed_curves.values()
                    for x, v in curve if x == e]
            if vals:
                mean_vals.append((e, np.mean(vals), np.std(vals)))

        if mean_vals:
            xs, ys, stds = zip(*mean_vals)
            color = ALGO_COLORS[alg] if metric == "paraphrase" else "#9C27B0"
            ax.plot(xs, ys, color=color, linewidth=2, label=label,
                    marker="s", markersize=3, linestyle=ls)
            ax.fill_between(xs, np.array(ys) - np.array(stds),
                            np.array(ys) + np.array(stds),
                            color=color, alpha=0.1)

    ax.set_xlabel("Cumulative Edits")
    ax.set_ylabel("Score")
    ax.set_title("(C) Paraphrase & Locality (AlphaEdit)")
    ax.legend(loc="upper right", fontsize=8)
    ax.set_ylim(-0.05, 1.05)
    ax.axhline(0.5, color="gray", linestyle=":", alpha=0.4)


# ─── Main ─────────────────────────────────────────────────────────────────────


def generate(output_dir: Path = PAPER_OUTPUT):
    """Generate Figure 2."""
    setup_style()

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    fig.suptitle(
        "Figure 2: AlphaEdit Reproduces at 2K but Degrades Beyond 7K Edits",
        fontsize=13, y=1.02,
    )

    panel_a_reproduction(axes[0])
    panel_b_long_horizon(axes[1])
    panel_c_paraphrase_locality(axes[2])

    plt.tight_layout()
    save_figure(fig, "fig2_reproduction", output_dir)


def main():
    parser = argparse.ArgumentParser(
        description="Generate Figure 2: Reproduction & long-horizon boundary")
    parser.add_argument("--output-dir", type=Path, default=PAPER_OUTPUT)
    args = parser.parse_args()
    generate(args.output_dir)


if __name__ == "__main__":
    main()

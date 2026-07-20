"""Figure 1 — Faithful reproduction and long-horizon boundary.

Question answered: Does AlphaEdit reproduce, and where does it begin to fail?

Panels:
  A. Standard-scale reproduction (AlphaEdit vs MEMIT through 3K, multi-seed)
  B. Long-horizon efficacy (1K → 10K, individual seed traces + mean)
  C. Probability locality (neighborhood_prob trajectory)
  D. Capability / locality comparison (neighborhood_prob normalized)

Usage:
    uv run python -m analysis.fig1_reproduction
    uv run python -m analysis.fig1_reproduction --output-dir results/figures/paper
"""

import argparse
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from analysis.style import (
    ALGO_COLORS, SEED_COLORS, setup_style, save_figure, PAPER_OUTPUT,
)
from analysis.loaders import load_checkpoint_metrics

# ─── Configuration ────────────────────────────────────────────────────────────

SEEDS = [42, 2024, 137]
EDIT_POINTS = [2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000, 10000]
ALGOS = ["AlphaEdit", "MEMIT"]


# ─── Helpers ──────────────────────────────────────────────────────────────────


def _collect_curves(metric: str, alg: str):
    """Collect per-seed curves for a metric/algorithm pair.

    Returns dict: seed → list of (edits, value).
    """
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
    """Panel A: Standard-scale reproduction through 3K."""
    short_points = [e for e in EDIT_POINTS if e <= 3000]

    for alg in ALGOS:
        seed_curves = {}
        for seed in SEEDS:
            curve = []
            for edits in short_points:
                m = load_checkpoint_metrics(seed, edits, alg)
                if m is not None:
                    curve.append((edits, m["efficacy"]))
            if curve:
                seed_curves[seed] = curve

        if not seed_curves:
            continue

        color = ALGO_COLORS[alg]

        # Mean + uncertainty band
        all_edits = sorted(set(e for c in seed_curves.values() for e, _ in c))
        mean_vals = []
        for e in all_edits:
            vals = [v for curve in seed_curves.values() for x, v in curve if x == e]
            if vals:
                mean_vals.append((e, np.mean(vals), np.std(vals)))

        if mean_vals:
            xs, ys, stds = zip(*mean_vals)
            ax.plot(xs, ys, color=color, linewidth=2.5, label=alg,
                    marker="o", markersize=5)
            ax.fill_between(xs, np.array(ys) - np.array(stds),
                            np.array(ys) + np.array(stds),
                            color=color, alpha=0.15)

    ax.set_xlabel("Total Edits")
    ax.set_ylabel("Efficacy")
    ax.set_title("(A) Standard-Scale Reproduction")
    ax.legend(loc="lower left")
    ax.set_ylim(-0.05, 1.05)
    ax.axhline(0.5, color="gray", linestyle=":", alpha=0.4)


def panel_b_long_horizon(ax):
    """Panel B: Long-horizon efficacy with individual seed traces."""
    for alg in ALGOS:
        _plot_algo_curves(ax, "efficacy", alg, show_individual=True)

    ax.set_xlabel("Total Edits")
    ax.set_ylabel("Efficacy")
    ax.set_title("(B) Long-Horizon Efficacy (2K → 10K)")
    ax.legend(loc="upper right")
    ax.set_ylim(-0.05, 1.05)
    ax.axhline(0.5, color="gray", linestyle=":", alpha=0.4)


def panel_c_probability_locality(ax):
    """Panel C: Probability locality (neighborhood_prob) trajectory."""
    for alg in ALGOS:
        _plot_algo_curves(ax, "neighborhood_prob", alg, show_individual=False)

    ax.set_xlabel("Total Edits")
    ax.set_ylabel("P(target_new | neighborhood)")
    ax.set_title("(C) Probability Locality")
    ax.legend(loc="upper right")
    ax.axhline(0.5, color="gray", linestyle=":", alpha=0.4, label="chance")


def panel_d_paraphrase(ax):
    """Panel D: Paraphrase (generalization) trajectory."""
    for alg in ALGOS:
        _plot_algo_curves(ax, "paraphrase", alg, show_individual=True)

    ax.set_xlabel("Total Edits")
    ax.set_ylabel("Paraphrase Success")
    ax.set_title("(D) Paraphrase Generalization")
    ax.legend(loc="upper right")
    ax.set_ylim(-0.05, 1.05)
    ax.axhline(0.5, color="gray", linestyle=":", alpha=0.4)


# ─── Main ─────────────────────────────────────────────────────────────────────


def generate(output_dir: Path = PAPER_OUTPUT):
    """Generate Figure 1."""
    setup_style()

    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    fig.suptitle(
        "Figure 1: AlphaEdit Reproduces Its Advantage but Degrades at Scale",
        fontsize=13, y=0.98,
    )

    panel_a_reproduction(axes[0, 0])
    panel_b_long_horizon(axes[0, 1])
    panel_c_probability_locality(axes[1, 0])
    panel_d_paraphrase(axes[1, 1])

    plt.tight_layout()
    save_figure(fig, "fig1_reproduction", output_dir)


def main():
    parser = argparse.ArgumentParser(description="Generate Figure 1: Reproduction & long-horizon boundary")
    parser.add_argument("--output-dir", type=Path, default=PAPER_OUTPUT)
    args = parser.parse_args()
    generate(args.output_dir)


if __name__ == "__main__":
    main()

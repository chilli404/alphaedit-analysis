"""Figure 4 — Stream concentration and individual interference.

Question answered: Does key-space concentration (batch geometry) causally drive
forgetting, and can we observe individual-edit interference?

Panels:
  A. Retention outcome under controlled concentration (paired points)
  B. Geometric response (within-batch cosine by ordering)
  C. Age-matched retained vs forgotten future-key cosine
  D. Effect summary (forest plot: OR per +0.1 cosine with 95% CI)

Usage:
    uv run python -m analysis.fig4_mechanism
    uv run python -m analysis.fig4_mechanism --output-dir results/figures/paper
"""

import argparse
import json
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from analysis.style import (
    SEED_COLORS, setup_style, save_figure, PAPER_OUTPUT, RESULTS,
)
from analysis.loaders import (
    load_matched_ordering_full_eval,
    load_matched_ordering_properties,
)

# ─── Configuration ────────────────────────────────────────────────────────────

SEEDS = [42, 2024]
INTERFERENCE_JSON = RESULTS / "figures" / "paper" / "interference_panel_results.json"


def _load_interference_results(output_dir: Path) -> dict:
    """Load interference panel results."""
    path = output_dir / "interference_panel_results.json"
    if not path.exists():
        path = INTERFERENCE_JSON
    if not path.exists():
        return {}
    with open(path) as f:
        return json.load(f)


# ─── Panel Functions ──────────────────────────────────────────────────────────


def panel_a_retention_under_concentration(ax):
    """Panel A: Retention under controlled concentration (paired low/high)."""
    alg = "AlphaEdit"
    ckpt = "10000_edits"
    orderings = ["key_dispersed", "key_clustered"]
    ordering_labels = ["Dispersed\n(low conc.)", "Clustered\n(high conc.)"]
    ordering_colors = ["#2196F3", "#E91E63"]

    x_positions = [0, 1]

    for si, seed in enumerate(SEEDS):
        first_1k_vals = []
        latest_1k_vals = []

        for ordering in orderings:
            data = load_matched_ordering_full_eval(seed, ordering, alg)
            if data is None or ckpt not in data:
                first_1k_vals.append(np.nan)
                latest_1k_vals.append(np.nan)
                continue
            first_1k_vals.append(data[ckpt]["first_1k"]["efficacy"])
            latest_1k_vals.append(data[ckpt]["latest_1k"]["efficacy"])

        # Plot first-1K (filled markers, connected)
        offset = si * 0.15
        ax.plot([x + offset for x in x_positions], first_1k_vals,
                color=SEED_COLORS[seed], linewidth=2, marker="o",
                markersize=8, label=f"First 1K (s{seed})")

        # Plot latest-1K (hollow markers)
        ax.plot([x + offset for x in x_positions], latest_1k_vals,
                color=SEED_COLORS[seed], linewidth=1, marker="o",
                markersize=8, markerfacecolor="none", linestyle="--",
                label=f"Latest 1K (s{seed})")

    ax.set_xticks(x_positions)
    ax.set_xticklabels(ordering_labels)
    ax.set_ylabel("Efficacy at 10K")
    ax.set_title("(a) Retention Under Controlled Concentration")
    ax.legend(loc="lower left", fontsize=7, ncol=2)
    ax.set_ylim(0.4, 1.05)
    ax.axhline(0.5, color="gray", linestyle=":", alpha=0.3)


def panel_b_geometric_response(ax):
    """Panel B: Within-batch cosine similarity by ordering."""
    bar_data = []

    for seed in SEEDS:
        props = load_matched_ordering_properties(seed)
        if props is None:
            continue
        for ordering in ["key_clustered", "key_dispersed"]:
            if ordering in props:
                bar_data.append({
                    "seed": seed,
                    "ordering": ordering,
                    "mean_cos": props[ordering]["mean_within_batch_cosine"],
                    "std_cos": props[ordering]["std_within_batch_cosine"],
                    "clusters_per_batch": props[ordering]["mean_clusters_per_batch"],
                })

    if not bar_data:
        ax.text(0.5, 0.5, "No stream property data", transform=ax.transAxes,
                ha="center", va="center", fontsize=11)
        ax.set_title("(b) Geometric Response")
        return

    # Group by ordering
    clustered = [d for d in bar_data if d["ordering"] == "key_clustered"]
    dispersed = [d for d in bar_data if d["ordering"] == "key_dispersed"]

    x = np.arange(len(SEEDS))
    width = 0.35

    clust_means = [d["mean_cos"] for d in clustered]
    clust_stds = [d["std_cos"] for d in clustered]
    disp_means = [d["mean_cos"] for d in dispersed]
    disp_stds = [d["std_cos"] for d in dispersed]

    ax.bar(x - width / 2, clust_means, width, yerr=clust_stds,
           label="Clustered", color="#E91E63", alpha=0.8,
           edgecolor="black", linewidth=0.5, capsize=3)
    ax.bar(x + width / 2, disp_means, width, yerr=disp_stds,
           label="Dispersed", color="#2196F3", alpha=0.8,
           edgecolor="black", linewidth=0.5, capsize=3)

    ax.set_xticks(x)
    ax.set_xticklabels([f"Seed {s}" for s in SEEDS])
    ax.set_ylabel("Mean Within-Batch Cosine")
    ax.set_title("(b) Stream Geometric Concentration")
    ax.legend(fontsize=8)

    # Annotate ratio
    if clust_means and disp_means:
        ratio = np.mean(clust_means) / np.mean(disp_means)
        ax.text(0.95, 0.95, f"Ratio: {ratio:.2f}\u00d7",
                transform=ax.transAxes, ha="right", va="top",
                fontsize=9, style="italic", color="gray")


def panel_c_age_matched_cosine(ax, results: dict):
    """Panel C: Age-matched retained vs forgotten future-key cosine."""
    ks = results.get("age_matched", {}).get("key_similarity", {})
    if not ks:
        ax.text(0.5, 0.5, "No key similarity data", transform=ax.transAxes,
                ha="center", va="center", fontsize=11)
        ax.set_title("(c) Key Cosine: Survived vs Forgotten")
        return

    bins = ["Q1_young", "Q2", "Q3", "Q4_old"]
    bin_labels = ["Q1\n(young)", "Q2", "Q3", "Q4\n(old)"]

    survived_vals = [ks[b]["mean_cos_survived"] for b in bins if b in ks]
    forgotten_vals = [ks[b]["mean_cos_forgotten"] for b in bins if b in ks]

    x = np.arange(len(bins))
    width = 0.35

    ax.bar(x - width / 2, survived_vals, width, label="Survived",
           color="#4CAF50", alpha=0.8, edgecolor="black", linewidth=0.5)
    ax.bar(x + width / 2, forgotten_vals, width, label="Forgotten",
           color="#E91E63", alpha=0.8, edgecolor="black", linewidth=0.5)

    # Annotate deltas
    for i, (s, f) in enumerate(zip(survived_vals, forgotten_vals)):
        delta = f - s
        ax.annotate(f"+{delta:.3f}", xy=(x[i] + width / 2, f),
                    xytext=(0, 4), textcoords="offset points",
                    ha="center", fontsize=7, color="#E91E63", weight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels(bin_labels)
    ax.set_ylabel("Max Cosine to Subsequent Edits")
    ax.set_title("(c) Key Geometry Predicts Forgetting")
    ax.legend(loc="upper left", fontsize=8)
    ax.set_ylim(0, max(forgotten_vals) * 1.15 if forgotten_vals else 1.0)


def panel_d_forest_plot(ax, results: dict):
    """Panel D: Forest plot — OR per +0.1 future-key cosine with 95% CI."""
    bootstrap = results.get("bootstrap", {})
    if not bootstrap:
        ax.text(0.5, 0.5, "No bootstrap data", transform=ax.transAxes,
                ha="center", va="center", fontsize=11)
        ax.set_title("(d) Effect Size: OR per +0.1 Cosine")
        return

    seeds = sorted(bootstrap.keys(), key=int)
    labels = []
    ors = []
    ci_los = []
    ci_his = []
    colors = []

    for seed_str in seeds:
        data = bootstrap[seed_str]
        seed = int(seed_str)
        labels.append(f"Seed {seed}")
        ors.append(data["or_per_0.1_mean"])
        ci_los.append(data["or_per_0.1_ci_025"])
        ci_his.append(data["or_per_0.1_ci_975"])
        colors.append(SEED_COLORS.get(seed, "#666666"))

    y = np.arange(len(labels))

    for i in range(len(labels)):
        ax.plot([ci_los[i], ci_his[i]], [y[i], y[i]], color=colors[i],
                linewidth=2.5, solid_capstyle="round")
        ax.plot(ors[i], y[i], "o", color=colors[i], markersize=10,
                markeredgecolor="black", markeredgewidth=0.5)

    # Reference line at OR=1.0
    ax.axvline(1.0, color="gray", linestyle="--", alpha=0.5, linewidth=1)

    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.set_xlabel("OR per +0.1 max cosine (survival)")
    ax.set_title("(d) Effect Size (Block Bootstrap 95% CI)")

    # Annotate OR values
    for i in range(len(labels)):
        ax.annotate(f"OR={ors[i]:.3f}",
                    xy=(ors[i], y[i]),
                    xytext=(0, 12), textcoords="offset points",
                    ha="center", fontsize=8, color=colors[i])

    ax.set_xlim(0.5, 1.1)


# ─── Main ─────────────────────────────────────────────────────────────────────


def generate(output_dir: Path = PAPER_OUTPUT):
    """Generate Figure 4."""
    setup_style()

    results = _load_interference_results(output_dir)

    fig, axes = plt.subplots(2, 2, figsize=(12, 9))

    panel_a_retention_under_concentration(axes[0, 0])
    panel_b_geometric_response(axes[0, 1])
    panel_c_age_matched_cosine(axes[1, 0], results)
    panel_d_forest_plot(axes[1, 1], results)

    plt.tight_layout()
    save_figure(fig, "fig4_mechanism", output_dir)


def main():
    parser = argparse.ArgumentParser(
        description="Generate Figure 4: Stream concentration & interference")
    parser.add_argument("--output-dir", type=Path, default=PAPER_OUTPUT)
    args = parser.parse_args()
    generate(args.output_dir)


if __name__ == "__main__":
    main()

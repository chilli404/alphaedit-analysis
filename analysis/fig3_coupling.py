"""Figure 3 — Controlled semantic concentration.

Question answered: Does semantic concentration causally accelerate forgetting?

Panels:
  A. Stream construction summary (subject reuse, unique subjects, overlap)
  B. Old vs recent retention (first-1K / latest-1K, seed 42 and 137)
  C. Retention trajectories + AUC (both seeds, both streams)
  D. Effective rank over edit count (low vs high coupling)

Usage:
    uv run python -m analysis.fig3_coupling
    uv run python -m analysis.fig3_coupling --output-dir results/figures/paper
"""

import argparse
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from analysis.style import (
    STREAM_COLORS, SEED_COLORS, setup_style, save_figure, PAPER_OUTPUT,
)
from analysis.loaders import (
    load_controlled_coupling_behavioral,
    load_controlled_coupling_jsonl,
    load_stream_audit,
)

# ─── Configuration ────────────────────────────────────────────────────────────

SEEDS = [42, 137]


# ─── Panel Functions ──────────────────────────────────────────────────────────


def panel_a_stream_construction(ax):
    """Panel A: Stream construction summary showing matched vs manipulated properties."""
    audit = load_stream_audit(42)
    if audit is None:
        ax.text(0.5, 0.5, "No stream audit data", transform=ax.transAxes,
                ha="center", va="center", fontsize=11)
        ax.set_title("(A) Stream Construction")
        return

    # Use whichever key names the audit has
    low_key = "low_structure" if "low_structure" in audit else "low_coupling"
    high_key = "high_structure" if "high_structure" in audit else "high_coupling"
    low = audit.get(low_key, {})
    high = audit.get(high_key, {})

    # Properties to show (manipulated)
    labels = ["Subject\nReuse Rate", "Unique\nSubjects\n(÷1000)", "Mean Intra-Batch\nOverlap"]
    low_vals = [
        low.get("subject_reuse_rate", 0),
        low.get("n_unique_subjects", 0) / 1000,
        low.get("mean_intra_batch_overlap", 0),
    ]
    high_vals = [
        high.get("subject_reuse_rate", 0),
        high.get("n_unique_subjects", 0) / 1000,
        high.get("mean_intra_batch_overlap", 0),
    ]

    x = np.arange(len(labels))
    width = 0.35

    ax.bar(x - width / 2, low_vals, width, label="Low Coupling",
           color=STREAM_COLORS["low_coupling"], alpha=0.8, edgecolor="black", linewidth=0.5)
    ax.bar(x + width / 2, high_vals, width, label="High Coupling",
           color=STREAM_COLORS["high_coupling"], alpha=0.8, edgecolor="black", linewidth=0.5)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel("Value")
    ax.set_title("(A) Stream Construction (Manipulated Properties)")
    ax.legend(loc="upper left")

    # Annotate that prompt/target/relation are MATCHED
    ax.text(0.98, 0.02, "Prompt length, target length,\nrelation distribution: MATCHED",
            transform=ax.transAxes, fontsize=7, ha="right", va="bottom",
            style="italic", color="gray")


def panel_b_retention_paired(ax):
    """Panel B: Old vs recent retention for both seeds, paired comparison."""
    categories = []
    low_vals = []
    high_vals = []

    for seed in SEEDS:
        behav = load_controlled_coupling_behavioral(seed)
        if behav is None:
            continue

        low = behav.get("low_coupling", {})
        high = behav.get("high_coupling", {})

        categories.append(f"Seed {seed}\nFirst-1K")
        low_vals.append(low.get("first_1k_mean_efficacy", 0))
        high_vals.append(high.get("first_1k_mean_efficacy", 0))

        categories.append(f"Seed {seed}\nLatest-1K")
        low_vals.append(low.get("last_1k_mean_efficacy", 0))
        high_vals.append(high.get("last_1k_mean_efficacy", 0))

    if not categories:
        ax.text(0.5, 0.5, "No behavioral eval data", transform=ax.transAxes,
                ha="center", va="center", fontsize=11)
        ax.set_title("(B) Old vs Recent Retention")
        return

    x = np.arange(len(categories))
    width = 0.35

    bars_low = ax.bar(x - width / 2, low_vals, width, label="Low Coupling",
                      color=STREAM_COLORS["low_coupling"], alpha=0.8,
                      edgecolor="black", linewidth=0.5)
    bars_high = ax.bar(x + width / 2, high_vals, width, label="High Coupling",
                       color=STREAM_COLORS["high_coupling"], alpha=0.8,
                       edgecolor="black", linewidth=0.5)

    # Annotate gaps on first-1K bars
    for i in range(0, len(categories), 2):
        gap = low_vals[i] - high_vals[i]
        if gap != 0:
            y_pos = max(low_vals[i], high_vals[i]) + 0.03
            ax.annotate(f"Δ={gap:.1%}", xy=(x[i], y_pos),
                        fontsize=8, ha="center", color="#333")

    ax.set_xticks(x)
    ax.set_xticklabels(categories, fontsize=8)
    ax.set_ylabel("Efficacy")
    ax.set_title("(B) Old vs Recent Retention")
    ax.legend(loc="lower right")
    ax.set_ylim(0, 1.15)
    ax.axhline(0.5, color="gray", linestyle=":", alpha=0.3)


def panel_c_retention_auc(ax):
    """Panel C: Retention AUC comparison across seeds."""
    results = []
    for seed in SEEDS:
        behav = load_controlled_coupling_behavioral(seed)
        if behav is None:
            continue
        for stream in ("low_coupling", "high_coupling"):
            data = behav.get(stream, {})
            if "retention_auc" in data:
                results.append({
                    "seed": seed,
                    "stream": stream,
                    "auc": data["retention_auc"],
                    "overall_eff": data.get("overall_efficacy", 0),
                })

    if not results:
        ax.text(0.5, 0.5, "No AUC data", transform=ax.transAxes,
                ha="center", va="center", fontsize=11)
        ax.set_title("(C) Retention AUC")
        return

    # Grouped bar: seeds on x-axis, streams as groups
    x_labels = [f"Seed {s}" for s in SEEDS]
    x = np.arange(len(SEEDS))
    width = 0.35

    for i, stream in enumerate(("low_coupling", "high_coupling")):
        vals = []
        for seed in SEEDS:
            match = [r for r in results if r["seed"] == seed and r["stream"] == stream]
            vals.append(match[0]["auc"] if match else 0)

        offset = (i - 0.5) * width
        ax.bar(x + offset, vals, width,
               label=stream.replace("_", " ").title(),
               color=STREAM_COLORS[stream], alpha=0.8,
               edgecolor="black", linewidth=0.5)

    ax.set_xticks(x)
    ax.set_xticklabels(x_labels)
    ax.set_ylabel("Retention AUC")
    ax.set_title("(C) Retention AUC (Higher = Better)")
    ax.legend()
    ax.set_ylim(0, 1.1)


def panel_d_effective_rank(ax):
    """Panel D: Cache effective rank over edit count (low vs high coupling)."""
    for seed in SEEDS:
        for stream in ("low_coupling", "high_coupling"):
            records = load_controlled_coupling_jsonl(stream, seed)
            if not records:
                continue

            edits = []
            ranks = []
            for r in records:
                agg = r.get("mechanism", {}).get("aggregate", {})
                eff_rank = agg.get("mean_cache_effective_rank")
                if eff_rank is not None:
                    edits.append(r["total_edits"])
                    ranks.append(eff_rank)

            if edits:
                color = STREAM_COLORS[stream]
                alpha = 1.0 if seed == 42 else 0.5
                linestyle = "-" if seed == 42 else "--"
                label = f"{stream.replace('_', ' ').title()} (s{seed})" if seed == 42 else None
                ax.plot(edits, ranks, color=color, linewidth=2,
                        alpha=alpha, linestyle=linestyle, label=label)

    ax.set_xlabel("Total Edits")
    ax.set_ylabel("Cache Effective Rank")
    ax.set_title("(D) Cache Spectral Concentration")
    ax.legend(loc="upper left")


# ─── Main ─────────────────────────────────────────────────────────────────────


def generate(output_dir: Path = PAPER_OUTPUT):
    """Generate Figure 3."""
    setup_style()

    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    fig.suptitle(
        "Figure 3: Semantic Concentration Causally Accelerates Forgetting",
        fontsize=13, y=0.98,
    )

    panel_a_stream_construction(axes[0, 0])
    panel_b_retention_paired(axes[0, 1])
    panel_c_retention_auc(axes[1, 0])
    panel_d_effective_rank(axes[1, 1])

    plt.tight_layout()
    save_figure(fig, "fig3_coupling", output_dir)


def main():
    parser = argparse.ArgumentParser(description="Generate Figure 3: Controlled semantic concentration")
    parser.add_argument("--output-dir", type=Path, default=PAPER_OUTPUT)
    args = parser.parse_args()
    generate(args.output_dir)


if __name__ == "__main__":
    main()

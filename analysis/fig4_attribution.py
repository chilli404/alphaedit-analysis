"""Figure 4 — Local editability, global interference, and baseline attribution.

Question answered: Is failure caused by projection blocking new edits,
and is projection actually responsible for AlphaEdit's advantage?

Panels:
  A. Functional projection signal (removed fraction + latest-cohort efficacy)
  B. Weight drift negative control (Frobenius drift low/high + retention)
  C. Matched method comparison at 3K (MEMIT vs AlphaEdit vs MEMIT+SeqReg)
  D. Matched method comparison at 5K (AlphaEdit vs MEMIT+SeqReg)

Usage:
    uv run python -m analysis.fig4_attribution
    uv run python -m analysis.fig4_attribution --output-dir results/figures/paper
"""

import argparse
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from analysis.style import (
    ALGO_COLORS, STREAM_COLORS, setup_style, save_figure, PAPER_OUTPUT,
)
from analysis.loaders import (
    load_checkpoint_metrics,
    load_controlled_coupling_jsonl,
    load_controlled_coupling_behavioral,
    load_seqreg_eval,
    load_seqreg_behavioral,
)

# ─── Configuration ────────────────────────────────────────────────────────────

SEED = 42


# ─── Panel Functions ──────────────────────────────────────────────────────────


def panel_a_projection_signal(ax):
    """Panel A: Projection preserves editability (removed fraction stays low)."""
    ax2 = ax.twinx()

    for stream in ("low_coupling", "high_coupling"):
        records = load_controlled_coupling_jsonl(stream, SEED)
        if not records:
            continue

        edits = []
        removed_fracs = []
        for r in records:
            agg = r.get("mechanism", {}).get("aggregate", {})
            rf = agg.get("mean_removed_fraction")
            if rf is not None:
                edits.append(r["total_edits"])
                removed_fracs.append(rf)

        if edits:
            color = STREAM_COLORS[stream]
            label = stream.replace("_", " ").title()
            ax.plot(edits, removed_fracs, color=color, linewidth=2,
                    label=f"{label} (removed fraction)")

    # Overlay: latest-cohort efficacy as proxy for q_t
    behav = load_controlled_coupling_behavioral(SEED)
    if behav:
        for stream in ("low_coupling", "high_coupling"):
            data = behav.get(stream, {})
            last_1k = data.get("last_1k_mean_efficacy")
            if last_1k is not None:
                color = STREAM_COLORS[stream]
                # Plot as horizontal reference line (single endpoint measurement)
                ax2.axhline(last_1k, color=color, linestyle="--", alpha=0.6)
                ax2.text(500, last_1k + 0.005,
                         f"Latest-1K eff: {last_1k:.3f}",
                         fontsize=7, color=color)

    ax.set_xlabel("Total Edits")
    ax.set_ylabel("Removed Fraction (projection loss)")
    ax.set_title("(A) Projection Signal: Low Removal, High Success")
    ax.legend(loc="upper left", fontsize=8)
    ax.set_ylim(0, 0.2)

    ax2.set_ylabel("Latest-Cohort Efficacy", color="gray")
    ax2.set_ylim(0.9, 1.01)
    ax2.tick_params(axis="y", labelcolor="gray")


def panel_b_weight_drift(ax):
    """Panel B: Weight drift negative control — more drift ≠ more forgetting.

    Uses cumulative mean_update_norm from controlled coupling JSONL as a proxy
    for Frobenius drift. This is an upper bound (updates could partially cancel),
    but sufficient for the negative-control argument.
    """
    has_data = False

    for stream in ("low_coupling", "high_coupling"):
        records = load_controlled_coupling_jsonl(stream, SEED)
        if not records:
            continue

        edits = []
        cumulative_norm = []
        running_sum = 0.0
        for r in records:
            agg = r.get("mechanism", {}).get("aggregate", {})
            upd_norm = agg.get("mean_update_norm")
            if upd_norm is not None:
                running_sum += upd_norm
                edits.append(r["total_edits"])
                cumulative_norm.append(running_sum)

        if edits:
            has_data = True
            color = STREAM_COLORS[stream]
            label = stream.replace("_", " ").title()
            ax.plot(edits, cumulative_norm, color=color, linewidth=2,
                    marker="o", markersize=3, label=label)

    if not has_data:
        ax.text(0.5, 0.5, "No update norm data", transform=ax.transAxes,
                ha="center", va="center", fontsize=11)

    ax.set_xlabel("Total Edits")
    ax.set_ylabel("Cumulative Update Norm (∑ ||ΔW||)")
    ax.set_title("(B) Weight Drift (Negative Control)")
    ax.legend(loc="upper left")

    # Annotate key insight
    ax.text(0.98, 0.98, "Low coupling: more drift,\nbetter retention",
            transform=ax.transAxes, fontsize=8, ha="right", va="top",
            style="italic", color="gray",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow", alpha=0.5))


def panel_c_comparison_3k(ax):
    """Panel C: Matched method comparison at 3K edits."""
    methods = {}

    # AlphaEdit at 3K
    ae = load_checkpoint_metrics(SEED, 3000, "AlphaEdit")
    if ae:
        methods["AlphaEdit"] = ae

    # MEMIT at 3K
    memit = load_checkpoint_metrics(SEED, 3000, "MEMIT")
    if memit:
        methods["MEMIT"] = memit

    # MEMIT+SeqReg at 3K (from full_eval JSON, "3000_edits" key)
    seqreg_eval = load_seqreg_eval(SEED)
    if seqreg_eval and "3000_edits" in seqreg_eval:
        sr = seqreg_eval["3000_edits"]
        methods["MEMIT+SeqReg"] = {
            "efficacy": sr.get("all_facts", {}).get("efficacy"),
            "paraphrase": sr.get("all_facts", {}).get("paraphrase"),
            "neighborhood": sr.get("all_facts", {}).get("neighborhood"),
        }
    # If no 3K in full eval, try 2K
    elif seqreg_eval and "2000_edits" in seqreg_eval:
        sr = seqreg_eval["2000_edits"]
        methods["MEMIT+SeqReg\n(2K)"] = {
            "efficacy": sr.get("all_facts", {}).get("efficacy"),
            "paraphrase": sr.get("all_facts", {}).get("paraphrase"),
            "neighborhood": sr.get("all_facts", {}).get("neighborhood"),
        }

    if not methods:
        ax.text(0.5, 0.5, "No comparison data at 3K", transform=ax.transAxes,
                ha="center", va="center", fontsize=11)
        ax.set_title("(C) Matched Comparison at 3K")
        return

    # Grouped bar chart
    metrics = ["efficacy", "paraphrase", "neighborhood"]
    metric_labels = ["Efficacy", "Paraphrase", "Specificity"]
    x = np.arange(len(metrics))
    n_methods = len(methods)
    width = 0.8 / n_methods

    for i, (method_name, data) in enumerate(methods.items()):
        vals = [data.get(m, 0) or 0 for m in metrics]
        offset = (i - n_methods / 2 + 0.5) * width
        color = ALGO_COLORS.get(method_name, f"C{i}")
        ax.bar(x + offset, vals, width, label=method_name,
               color=color, alpha=0.8, edgecolor="black", linewidth=0.5)

    ax.set_xticks(x)
    ax.set_xticklabels(metric_labels)
    ax.set_ylabel("Score")
    ax.set_title("(C) Matched Comparison at 3K Edits")
    ax.legend(loc="upper right", fontsize=8)
    ax.set_ylim(0, 1.1)


def panel_d_comparison_5k(ax):
    """Panel D: Matched comparison at 5K (AlphaEdit vs MEMIT+SeqReg)."""
    methods = {}

    # AlphaEdit at 5K
    ae = load_checkpoint_metrics(SEED, 5000, "AlphaEdit")
    if ae:
        methods["AlphaEdit"] = ae

    # MEMIT+SeqReg at 5K
    seqreg_eval = load_seqreg_eval(SEED)
    if seqreg_eval and "5000_edits" in seqreg_eval:
        sr = seqreg_eval["5000_edits"]
        methods["MEMIT+SeqReg"] = {
            "efficacy": sr.get("all_facts", {}).get("efficacy"),
            "paraphrase": sr.get("all_facts", {}).get("paraphrase"),
            "neighborhood": sr.get("all_facts", {}).get("neighborhood"),
        }

    # Also try behavioral directory
    if "MEMIT+SeqReg" not in methods:
        sr_behav = load_seqreg_behavioral(SEED, 5000)
        if sr_behav:
            methods["MEMIT+SeqReg"] = sr_behav

    if not methods:
        ax.text(0.5, 0.5, "No comparison data at 5K", transform=ax.transAxes,
                ha="center", va="center", fontsize=11)
        ax.set_title("(D) Matched Comparison at 5K")
        return

    # Grouped bar chart with more detailed metrics
    metrics = ["efficacy", "paraphrase", "neighborhood"]
    metric_labels = ["Efficacy", "Paraphrase", "Specificity"]
    x = np.arange(len(metrics))
    n_methods = len(methods)
    width = 0.8 / n_methods

    for i, (method_name, data) in enumerate(methods.items()):
        vals = [data.get(m, 0) or 0 for m in metrics]
        offset = (i - n_methods / 2 + 0.5) * width
        color = ALGO_COLORS.get(method_name, f"C{i}")
        ax.bar(x + offset, vals, width, label=method_name,
               color=color, alpha=0.8, edgecolor="black", linewidth=0.5)

    ax.set_xticks(x)
    ax.set_xticklabels(metric_labels)
    ax.set_ylabel("Score")
    ax.set_title("(D) Matched Comparison at 5K Edits")
    ax.legend(loc="upper right", fontsize=8)
    ax.set_ylim(0, 1.1)

    # Annotate key finding
    if "AlphaEdit" in methods and "MEMIT+SeqReg" in methods:
        ae_eff = methods["AlphaEdit"].get("efficacy", 0)
        sr_eff = methods["MEMIT+SeqReg"].get("efficacy", 0)
        if sr_eff and ae_eff:
            delta = sr_eff - ae_eff
            sign = "+" if delta > 0 else ""
            ax.text(0.02, 0.02, f"SeqReg vs AlphaEdit efficacy: {sign}{delta:.1%}",
                    transform=ax.transAxes, fontsize=8, ha="left", va="bottom",
                    weight="bold",
                    color="#4CAF50" if delta > 0 else "#E91E63")


# ─── Main ─────────────────────────────────────────────────────────────────────


def generate(output_dir: Path = PAPER_OUTPUT):
    """Generate Figure 4."""
    setup_style()

    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    fig.suptitle(
        "Figure 4: Projection Preserves Editability; Regularization Drives Stability",
        fontsize=13, y=0.98,
    )

    panel_a_projection_signal(axes[0, 0])
    panel_b_weight_drift(axes[0, 1])
    panel_c_comparison_3k(axes[1, 0])
    panel_d_comparison_5k(axes[1, 1])

    plt.tight_layout()
    save_figure(fig, "fig4_attribution", output_dir)


def main():
    parser = argparse.ArgumentParser(description="Generate Figure 4: Editability, interference, attribution")
    parser.add_argument("--output-dir", type=Path, default=PAPER_OUTPUT)
    args = parser.parse_args()
    generate(args.output_dir)


if __name__ == "__main__":
    main()

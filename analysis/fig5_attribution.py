"""Figure 5 — Attribution correction: projection vs historical constraints.

Question answered: Does null-space projection (AlphaEdit) or full-history
sequential regularization (MEMIT-Seq) better preserve edits, and what
does the functional preservation decomposition reveal?

Panels (Panel A is a manual schematic — skipped here):
  B. Cumulative performance at 5K (grouped dot plot: MEMIT vs AlphaEdit vs SeqReg)
  C. Retention/plasticity decomposition at 5K
  D. Functional projection preservation (survival fraction by method × ordering)

Usage:
    uv run python -m analysis.fig5_attribution
    uv run python -m analysis.fig5_attribution --output-dir results/figures/paper
"""

import argparse
import json
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from analysis.style import (
    ALGO_COLORS, setup_style, save_figure, PAPER_OUTPUT, RESULTS,
)
from analysis.loaders import (
    load_checkpoint_metrics,
    load_matched_ordering_full_eval,
)

# ─── Configuration ────────────────────────────────────────────────────────────

MECHANISM_JSON = RESULTS / "figures" / "paper" / "interference_mechanism_summary.json"

METHOD_COLORS = {
    "MEMIT": ALGO_COLORS["MEMIT"],
    "AlphaEdit": ALGO_COLORS["AlphaEdit"],
    "MEMIT-Seq": ALGO_COLORS["MEMIT+SeqReg"],
}


def _load_mechanism_summary(output_dir: Path) -> dict:
    """Load interference mechanism summary."""
    path = output_dir / "interference_mechanism_summary.json"
    if not path.exists():
        path = MECHANISM_JSON
    if not path.exists():
        return {}
    with open(path) as f:
        return json.load(f)


# ─── Panel Functions ──────────────────────────────────────────────────────────


def panel_b_cumulative_performance(ax):
    """Panel B: Cumulative performance at 10K (grouped dot plot)."""
    seed = 42
    ordering = "key_clustered"
    ckpt = "10000_edits"

    # Load data for each method
    methods = {}

    # MEMIT from failure curve
    memit_data = load_checkpoint_metrics(seed, 10000, "MEMIT")
    if memit_data:
        methods["MEMIT"] = memit_data

    # AlphaEdit from matched ordering
    ae_data = load_matched_ordering_full_eval(seed, ordering, "AlphaEdit")
    if ae_data and ckpt in ae_data:
        methods["AlphaEdit"] = ae_data[ckpt]["all_facts"]

    # MEMIT-Seq from matched ordering
    seq_data = load_matched_ordering_full_eval(
        seed, ordering, "MEMIT-Seq-lp1.0-ld0.0-cache0")
    if seq_data and ckpt in seq_data:
        methods["MEMIT-Seq"] = seq_data[ckpt]["all_facts"]

    if not methods:
        ax.text(0.5, 0.5, "No data available", transform=ax.transAxes,
                ha="center", va="center", fontsize=11)
        ax.set_title("(B) Cumulative Performance at 5K")
        return

    metrics = ["efficacy", "paraphrase", "neighborhood"]
    metric_labels = ["Efficacy", "Paraphrase", "Specificity"]
    x = np.arange(len(metrics))

    n_methods = len(methods)
    width = 0.8 / n_methods
    offsets = np.linspace(-(n_methods - 1) * width / 2,
                          (n_methods - 1) * width / 2, n_methods)

    for i, (method, data) in enumerate(methods.items()):
        vals = [data.get(m, 0) for m in metrics]
        color = METHOD_COLORS.get(method, "#666666")
        ax.bar(x + offsets[i], vals, width, label=method,
               color=color, alpha=0.8, edgecolor="black", linewidth=0.5)

    ax.set_xticks(x)
    ax.set_xticklabels(metric_labels)
    ax.set_ylabel("Score")
    ax.set_title("(B) Cumulative Performance at 5K Edits")
    ax.legend(loc="upper right", fontsize=8)
    ax.set_ylim(0, 1.1)
    ax.axhline(0.5, color="gray", linestyle=":", alpha=0.3)


def panel_c_retention_decomposition(ax):
    """Panel C: Retention/plasticity decomposition at 10K."""
    seed = 42
    ordering = "key_clustered"
    ckpt = "10000_edits"

    methods = {}
    method_labels = []

    # AlphaEdit
    ae_data = load_matched_ordering_full_eval(seed, ordering, "AlphaEdit")
    if ae_data and ckpt in ae_data:
        entry = ae_data[ckpt]
        methods["AlphaEdit"] = {
            "First-1K\nRetention": entry["first_1k"]["efficacy"],
            "Latest-1K\nEfficacy": entry["latest_1k"]["efficacy"],
            "Latest-100\nEfficacy": entry["latest_100"]["efficacy"],
            "Retention\nAUC": entry["retention_auc"],
        }
        method_labels.append("AlphaEdit")

    # MEMIT-Seq
    seq_data = load_matched_ordering_full_eval(
        seed, ordering, "MEMIT-Seq-lp1.0-ld0.0-cache0")
    if seq_data and ckpt in seq_data:
        entry = seq_data[ckpt]
        methods["MEMIT-Seq"] = {
            "First-1K\nRetention": entry["first_1k"]["efficacy"],
            "Latest-1K\nEfficacy": entry["latest_1k"]["efficacy"],
            "Latest-100\nEfficacy": entry["latest_100"]["efficacy"],
            "Retention\nAUC": entry["retention_auc"],
        }
        method_labels.append("MEMIT-Seq")

    if not methods:
        ax.text(0.5, 0.5, "No data available", transform=ax.transAxes,
                ha="center", va="center", fontsize=11)
        ax.set_title("(C) Retention/Plasticity Decomposition")
        return

    decomp_metrics = list(next(iter(methods.values())).keys())
    x = np.arange(len(decomp_metrics))
    n_methods = len(methods)
    width = 0.8 / n_methods
    offsets = np.linspace(-(n_methods - 1) * width / 2,
                          (n_methods - 1) * width / 2, n_methods)

    for i, (method, vals_dict) in enumerate(methods.items()):
        vals = list(vals_dict.values())
        color = METHOD_COLORS.get(method, "#666666")
        ax.bar(x + offsets[i], vals, width, label=method,
               color=color, alpha=0.8, edgecolor="black", linewidth=0.5)

    ax.set_xticks(x)
    ax.set_xticklabels(decomp_metrics, fontsize=8)
    ax.set_ylabel("Score")
    ax.set_title("(C) Retention/Plasticity at 5K (seed 42)")
    ax.legend(loc="lower right", fontsize=8)
    ax.set_ylim(0.5, 1.05)


def panel_d_functional_preservation(ax, mechanism: dict):
    """Panel D: Functional projection preservation by method × ordering."""
    layer_data = mechanism.get("joint_model_layer6", {})
    if not layer_data:
        ax.text(0.5, 0.5, "No mechanism summary data", transform=ax.transAxes,
                ha="center", va="center", fontsize=11)
        ax.set_title("(D) Functional Preservation")
        return

    # Extract survival fractions
    methods = ["AlphaEdit", "MEMIT-Seq-lp1.0-ld0.0-cache0"]
    method_labels = ["AlphaEdit", "MEMIT-Seq"]
    orderings = ["clustered", "dispersed"]
    ordering_labels = ["Clustered", "Dispersed"]

    x = np.arange(len(orderings))
    width = 0.35

    for mi, (method, mlabel) in enumerate(zip(methods, method_labels)):
        survival_rates = []
        for ordering in orderings:
            key = f"{method}/{ordering}"
            if key in layer_data:
                entry = layer_data[key]
                n_valid = entry["n_valid"]
                n_retained = entry["n_retained"]
                survival_rates.append(n_retained / n_valid)
            else:
                survival_rates.append(np.nan)

        offset = (mi - 0.5) * width
        color = METHOD_COLORS.get(mlabel, "#666666")
        bars = ax.bar(x + offset, survival_rates, width, label=mlabel,
                      color=color, alpha=0.8, edgecolor="black", linewidth=0.5)

        # Annotate fractions
        for i, (rate, ordering) in enumerate(zip(survival_rates, orderings)):
            if not np.isnan(rate):
                key = f"{method}/{ordering}"
                entry = layer_data[key]
                ax.annotate(
                    f"{entry['n_retained']}/{entry['n_valid']}",
                    xy=(x[i] + offset, rate),
                    xytext=(0, 4), textcoords="offset points",
                    ha="center", fontsize=7, weight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels(ordering_labels)
    ax.set_ylabel("Survival Fraction")
    ax.set_title("(D) Functional Preservation by Method \u00d7 Ordering")
    ax.legend(loc="lower left", fontsize=8)
    ax.set_ylim(0.5, 1.05)
    ax.axhline(0.9, color="gray", linestyle=":", alpha=0.3)


# ─── Main ─────────────────────────────────────────────────────────────────────


def generate(output_dir: Path = PAPER_OUTPUT):
    """Generate Figure 5."""
    setup_style()

    mechanism = _load_mechanism_summary(output_dir)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    fig.suptitle(
        "Figure 5: Attribution Correction \u2014 Projection vs Historical Constraints",
        fontsize=13, y=1.02,
    )

    panel_b_cumulative_performance(axes[0])
    panel_c_retention_decomposition(axes[1])
    panel_d_functional_preservation(axes[2], mechanism)

    plt.tight_layout()
    save_figure(fig, "fig5_attribution", output_dir)


def main():
    parser = argparse.ArgumentParser(
        description="Generate Figure 5: Attribution correction")
    parser.add_argument("--output-dir", type=Path, default=PAPER_OUTPUT)
    args = parser.parse_args()
    generate(args.output_dir)


if __name__ == "__main__":
    main()

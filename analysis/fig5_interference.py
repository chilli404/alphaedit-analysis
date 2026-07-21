"""Figure 5 — Per-edit interference: key geometry predicts individual forgetting.

Question answered: Among edits of comparable age, are those geometrically
closer to subsequent edits more likely to be forgotten?

Panels:
  A. Age-matched key cosine: forgotten vs survived (grouped bar, 4 bins)
  B. Model coefficients (forest plot): β for max_cosine across trajectories
  C. AIC model comparison: age-only → +semantic → +keys (stacked improvement)
  D. Dissociation: relation overlap (protective) vs key cosine (harmful)

Usage:
    uv run python -m analysis.fig5_interference
    uv run python -m analysis.fig5_interference --output-dir results/figures/paper
"""

import argparse
import json
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from analysis.style import (
    SEED_COLORS, setup_style, save_figure, PAPER_OUTPUT, RESULTS,
)

# ─── Configuration ────────────────────────────────────────────────────────────

RESULTS_JSON = RESULTS / "figures" / "paper" / "interference_panel_results.json"


def load_results(output_dir: Path) -> dict:
    """Load interference panel results JSON."""
    path = output_dir / "interference_panel_results.json"
    if not path.exists():
        path = RESULTS_JSON
    if not path.exists():
        return {}
    with open(path) as f:
        return json.load(f)


# ─── Panel Functions ──────────────────────────────────────────────────────────


def panel_a_key_cosine_by_age(ax, results: dict):
    """Panel A: Age-matched key cosine — forgotten edits have higher similarity."""
    ks = results.get("age_matched", {}).get("key_similarity", {})
    if not ks:
        ax.text(0.5, 0.5, "No key similarity data", transform=ax.transAxes,
                ha="center", va="center", fontsize=11)
        ax.set_title("(A) Key Cosine by Age Bin")
        return

    bins = ["Q1_young", "Q2", "Q3", "Q4_old"]
    bin_labels = ["Q1\n(young)", "Q2", "Q3", "Q4\n(old)"]

    survived_vals = [ks[b]["mean_cos_survived"] for b in bins if b in ks]
    forgotten_vals = [ks[b]["mean_cos_forgotten"] for b in bins if b in ks]

    x = np.arange(len(bins))
    width = 0.35

    bars_s = ax.bar(x - width / 2, survived_vals, width, label="Survived",
                    color="#4CAF50", alpha=0.8, edgecolor="black", linewidth=0.5)
    bars_f = ax.bar(x + width / 2, forgotten_vals, width, label="Forgotten",
                    color="#E91E63", alpha=0.8, edgecolor="black", linewidth=0.5)

    # Annotate deltas
    for i, (s, f) in enumerate(zip(survived_vals, forgotten_vals)):
        delta = f - s
        ax.annotate(f"+{delta:.3f}", xy=(x[i] + width / 2, f),
                    xytext=(0, 4), textcoords="offset points",
                    ha="center", fontsize=7, color="#E91E63", weight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels(bin_labels)
    ax.set_ylabel("Max Cosine Similarity\n(to subsequent edits)")
    ax.set_title("(A) Key Geometry Predicts Forgetting")
    ax.legend(loc="upper left", fontsize=8)
    ax.set_ylim(0, max(forgotten_vals) * 1.15)


def panel_b_forest_plot(ax, results: dict):
    """Panel B: Forest plot of max_cosine β across trajectories."""
    per_traj = results.get("per_trajectory", {})
    if not per_traj:
        ax.text(0.5, 0.5, "No trajectory data", transform=ax.transAxes,
                ha="center", va="center", fontsize=11)
        ax.set_title("(B) Key Cosine Effect Size")
        return

    seeds = []
    betas = []
    ses = []
    colors = []

    for seed_str, data in sorted(per_traj.items()):
        if "max_cosine_coef" not in data:
            continue
        seed = int(seed_str)
        seeds.append(f"Seed {seed}")
        betas.append(data["max_cosine_coef"])
        ses.append(data.get("max_cosine_se", 0))
        colors.append(SEED_COLORS.get(seed, "#666666"))

    if not seeds:
        ax.text(0.5, 0.5, "No key cosine coefficients", transform=ax.transAxes,
                ha="center", va="center", fontsize=11)
        ax.set_title("(B) Key Cosine Effect Size")
        return

    y = np.arange(len(seeds))
    betas = np.array(betas)
    ses = np.array(ses)

    # 95% CI
    ci_lo = betas - 1.96 * ses
    ci_hi = betas + 1.96 * ses

    for i in range(len(seeds)):
        ax.plot([ci_lo[i], ci_hi[i]], [y[i], y[i]], color=colors[i],
                linewidth=2, solid_capstyle="round")
        ax.plot(betas[i], y[i], "o", color=colors[i], markersize=8,
                markeredgecolor="black", markeredgewidth=0.5)

    # Null line
    ax.axvline(0, color="gray", linestyle="--", alpha=0.5, linewidth=1)

    ax.set_yticks(y)
    ax.set_yticklabels(seeds)
    ax.set_xlabel("β (max cosine → log-odds of survival)")
    ax.set_title("(B) Effect Size: Key Similarity")

    # Annotate OR
    for i in range(len(seeds)):
        or_val = np.exp(betas[i])
        ax.annotate(f"OR={or_val:.3f}", xy=(betas[i], y[i]),
                    xytext=(10, 0), textcoords="offset points",
                    fontsize=8, va="center")


def panel_c_aic_ladder(ax, results: dict):
    """Panel C: AIC improvement ladder — age → +semantic → +keys."""
    per_traj = results.get("per_trajectory", {})
    if not per_traj:
        ax.text(0.5, 0.5, "No model comparison data", transform=ax.transAxes,
                ha="center", va="center", fontsize=11)
        ax.set_title("(C) Model Comparison (AIC)")
        return

    seeds = sorted(per_traj.keys())
    x = np.arange(len(seeds))
    width = 0.25

    m1_vals = []
    m2_vals = []
    m4_vals = []
    for seed_str in seeds:
        data = per_traj[seed_str]
        m1_vals.append(data.get("m1_aic", 0))
        m2_vals.append(data.get("m2_aic", 0))
        m4_vals.append(data.get("m4_aic", 0))

    # Plot relative AIC (improvement over age-only)
    m1_ref = np.array(m1_vals)
    m2_delta = np.array(m1_vals) - np.array(m2_vals)  # improvement (positive = better)
    m4_delta = np.array(m1_vals) - np.array(m4_vals)

    ax.bar(x - width, m2_delta, width, label="+ Semantic exposure",
           color="#FF9800", alpha=0.8, edgecolor="black", linewidth=0.5)
    ax.bar(x, m4_delta, width, label="+ Key cosine",
           color="#2196F3", alpha=0.8, edgecolor="black", linewidth=0.5)

    # Annotate the key improvement
    for i in range(len(seeds)):
        keys_over_sem = m4_delta[i] - m2_delta[i]
        ax.annotate(f"+{keys_over_sem:.0f}", xy=(x[i], m4_delta[i]),
                    xytext=(0, 4), textcoords="offset points",
                    ha="center", fontsize=8, color="#2196F3", weight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels([f"Seed {s}" for s in seeds])
    ax.set_ylabel("AIC Improvement\n(vs age-only model)")
    ax.set_title("(C) Model Fit: Geometric > Semantic")
    ax.legend(loc="upper left", fontsize=8)
    ax.axhline(0, color="gray", linestyle=":", alpha=0.3)


def panel_d_negative_controls(ax, results: dict):
    """Panel D: Negative controls — real keys vs random/preceding/permuted."""
    neg_ctrls = results.get("negative_controls", {})
    if not neg_ctrls:
        ax.text(0.5, 0.5, "No negative control data", transform=ax.transAxes,
                ha="center", va="center", fontsize=11)
        ax.set_title("(D) Negative Controls")
        return

    # Collect diffs across seeds for each control type
    control_labels = ["Real\n(subsequent)", "Preceding\nkeys", "Random\nkeys", "Permuted\n(null mean)"]
    seeds = sorted(neg_ctrls.keys())
    n_seeds = len(seeds)
    width = 0.35

    all_diffs = {label: [] for label in control_labels}
    for seed_str in seeds:
        ctrls = neg_ctrls[seed_str]
        rand = ctrls.get("random_keys", {})
        prec = ctrls.get("preceding_keys", {})
        perm = ctrls.get("permuted_keys", {})

        all_diffs["Real\n(subsequent)"].append(rand.get("real_diff", 0))
        all_diffs["Preceding\nkeys"].append(prec.get("preceding_diff", 0))
        all_diffs["Random\nkeys"].append(rand.get("random_diff", 0))
        all_diffs["Permuted\n(null mean)"].append(perm.get("perm_mean", 0))

    x = np.arange(len(control_labels))
    colors = ["#E91E63", "#9E9E9E", "#9E9E9E", "#9E9E9E"]

    for i, seed_str in enumerate(seeds):
        offset = (i - n_seeds / 2 + 0.5) * width
        vals = [all_diffs[label][i] for label in control_labels]
        ax.bar(x + offset, vals, width,
               color=colors, alpha=0.7 + 0.15 * i,
               edgecolor="black", linewidth=0.5,
               label=f"Seed {seed_str}" if i == 0 else None)

    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(control_labels, fontsize=8)
    ax.set_ylabel("Δ Max Cosine\n(forgotten − survived)")
    ax.set_title("(D) Negative Controls")

    # Annotate significance
    ax.text(0.98, 0.98, "Only subsequent real keys\npredict forgetting",
            transform=ax.transAxes, fontsize=8, ha="right", va="top",
            style="italic", color="gray")


# ─── Main ─────────────────────────────────────────────────────────────────────


def generate(output_dir: Path = PAPER_OUTPUT):
    """Generate Figure 5."""
    setup_style()

    results = load_results(output_dir)
    if not results:
        print("  ERROR: No interference_panel_results.json found.")
        print("  Run: uv run python -m analysis.interference_panel --keys-dir results/keys")
        return

    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    fig.suptitle(
        "Figure 5: Key Geometry Predicts Individual Edit Forgetting",
        fontsize=13, y=0.98,
    )

    panel_a_key_cosine_by_age(axes[0, 0], results)
    panel_b_forest_plot(axes[0, 1], results)
    panel_c_aic_ladder(axes[1, 0], results)
    panel_d_negative_controls(axes[1, 1], results)

    plt.tight_layout()
    save_figure(fig, "fig5_interference", output_dir)


def main():
    parser = argparse.ArgumentParser(description="Generate Figure 5: Per-edit interference")
    parser.add_argument("--output-dir", type=Path, default=PAPER_OUTPUT)
    args = parser.parse_args()
    generate(args.output_dir)


if __name__ == "__main__":
    main()

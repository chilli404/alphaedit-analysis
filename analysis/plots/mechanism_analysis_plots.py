#!/usr/bin/env python3
"""
Visualize mechanism analysis results: null-space geometry trajectory.

Produces a multi-panel figure showing how cache_c properties evolve
from 1K to 10K edits, revealing the mechanism behind AlphaEdit's collapse.

Usage:
    uv run python -m analysis.mechanism_analysis_plots
"""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

RESULTS_DIR = Path("results/mechanism_analysis")
OUTPUT_DIR = Path("results/figures/mechanism_analysis")


def load_mechanism_data(results_dir: Path) -> list[dict]:
    """Load all mechanism analysis JSONL files."""
    records = []
    for f in sorted(results_dir.glob("*.jsonl")):
        with open(f) as fh:
            for line in fh:
                line = line.strip()
                if line:
                    records.append(json.loads(line))
    return records


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    records = load_mechanism_data(RESULTS_DIR)
    if not records:
        print("No mechanism analysis data found.")
        return

    print(f"Loaded {len(records)} records")

    # Extract data by layer
    layers = sorted(set(r["layer_idx"] for r in records))
    edits = sorted(set(r["total_edits"] for r in records))
    hidden_dim = records[0]["cache"]["hidden_dim"]

    print(f"Layers: {layers}")
    print(f"Edit counts: {edits}")
    print(f"Hidden dim: {hidden_dim}")

    # Build arrays indexed by (layer, edit_count)
    data = {}
    for r in records:
        key = (r["layer_idx"], r["total_edits"])
        data[key] = r["cache"]

    # ─── Figure: 4-panel mechanism diagnostic ───
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    fig.suptitle("AlphaEdit Cache Geometry: Null-Space Exhaustion Trajectory (Seed 42)",
                 fontsize=13, fontweight="bold")

    colors = plt.cm.viridis(np.linspace(0.2, 0.9, len(layers)))

    # Panel A: Numerical rank vs edits (with ceiling line)
    ax = axes[0, 0]
    for i, layer in enumerate(layers):
        y = [data[(layer, e)]["cache_numerical_rank"] for e in edits]
        ax.plot(edits, y, "o-", color=colors[i], label=f"Layer {layer}", markersize=4)
    ax.axhline(hidden_dim, color="red", linestyle="--", alpha=0.7, label=f"Hidden dim ({hidden_dim})")
    ax.set_xlabel("Total Edits")
    ax.set_ylabel("Numerical Rank")
    ax.set_title("A. Cache Numerical Rank (dimensions consumed)")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # Annotate null-space remaining at 10K
    rank_10k = np.mean([data[(layer, 10000)]["cache_numerical_rank"] for layer in layers])
    remaining = hidden_dim - rank_10k
    ax.annotate(f"~{int(remaining)} dims\nremaining",
                xy=(10000, rank_10k), xytext=(8000, rank_10k + 1500),
                arrowprops=dict(arrowstyle="->", color="black"),
                fontsize=9)

    # Panel B: Effective rank vs edits (shows spectral concentration)
    ax = axes[0, 1]
    for i, layer in enumerate(layers):
        numerical = [data[(layer, e)]["cache_numerical_rank"] for e in edits]
        effective = [data[(layer, e)]["cache_effective_rank"] for e in edits]
        ax.plot(edits, effective, "o-", color=colors[i], label=f"Layer {layer}", markersize=4)
    # Add numerical rank reference (layer-average) as dashed
    avg_numerical = [np.mean([data[(layer, e)]["cache_numerical_rank"] for layer in layers]) for e in edits]
    ax.plot(edits, avg_numerical, "k--", alpha=0.5, label="Numerical rank (avg)")
    ax.set_xlabel("Total Edits")
    ax.set_ylabel("Effective Rank")
    ax.set_title("B. Effective Rank (spectral concentration)")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # Panel C: Condition number (log scale)
    ax = axes[1, 0]
    for i, layer in enumerate(layers):
        y = [data[(layer, e)]["cache_condition"] for e in edits]
        ax.semilogy(edits, y, "o-", color=colors[i], label=f"Layer {layer}", markersize=4)
    ax.set_xlabel("Total Edits")
    ax.set_ylabel("Condition Number")
    ax.set_title("C. Cache Condition Number (numerical stability)")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # Panel D: Consumption ratio = numerical_rank / hidden_dim and
    # "effective consumption" = effective_rank / hidden_dim
    ax = axes[1, 1]
    for i, layer in enumerate(layers):
        consumption = [data[(layer, e)]["cache_numerical_rank"] / hidden_dim for e in edits]
        eff_consumption = [data[(layer, e)]["cache_effective_rank"] / hidden_dim for e in edits]
        ax.plot(edits, consumption, "o-", color=colors[i], markersize=4,
                label=f"L{layer} numerical")
        ax.plot(edits, eff_consumption, "s--", color=colors[i], markersize=3, alpha=0.6)

    ax.axhline(1.0, color="red", linestyle=":", alpha=0.5, label="Full exhaustion")
    ax.axhline(0.7, color="orange", linestyle=":", alpha=0.5, label="70% threshold")
    ax.set_xlabel("Total Edits")
    ax.set_ylabel("Fraction of Hidden Dim")
    ax.set_title("D. Null-Space Consumption (solid=numerical, dashed=effective)")
    ax.legend(fontsize=7, ncol=2)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1.05)

    plt.tight_layout()
    out_path = OUTPUT_DIR / "mechanism_cache_geometry.pdf"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.savefig(out_path.with_suffix(".png"), dpi=150, bbox_inches="tight")
    print(f"\nSaved: {out_path}")

    # ─── Print key statistics ───
    print("\n" + "=" * 70)
    print("KEY FINDINGS")
    print("=" * 70)

    print(f"\nHidden dimension: {hidden_dim}")
    print(f"\nNull-space consumption at collapse boundary (5K-7K edits):")
    for e in [5000, 7000]:
        ranks = [data[(layer, e)]["cache_numerical_rank"] for layer in layers]
        eff_ranks = [data[(layer, e)]["cache_effective_rank"] for layer in layers]
        print(f"  {e} edits: numerical rank = {np.mean(ranks):.0f} "
              f"({np.mean(ranks)/hidden_dim*100:.1f}%), "
              f"effective rank = {np.mean(eff_ranks):.0f} "
              f"({np.mean(eff_ranks)/hidden_dim*100:.1f}%)")

    print(f"\nNull-space remaining at 10K edits:")
    for layer in layers:
        nr = data[(layer, 10000)]["cache_numerical_rank"]
        er = data[(layer, 10000)]["cache_effective_rank"]
        print(f"  Layer {layer}: {hidden_dim - nr} dims free (numerical), "
              f"eff_rank/hidden = {er/hidden_dim:.2%}")

    print(f"\nEffective/Numerical rank ratio (spectral concentration):")
    for e in edits:
        ratios = [data[(layer, e)]["cache_effective_rank"] /
                  max(1, data[(layer, e)]["cache_numerical_rank"]) for layer in layers]
        print(f"  {e:>5} edits: {np.mean(ratios):.3f} (1.0 = uniform, lower = concentrated)")

    print(f"\nCondition number trajectory (layer-averaged):")
    for e in edits:
        conds = [data[(layer, e)]["cache_condition"] for layer in layers]
        print(f"  {e:>5} edits: {np.mean(conds):.2e}")

    # ─── Supplementary: top singular value dominance ───
    print(f"\nTop singular value share (layer 7, deepest edited):")
    for e in edits:
        share = data[(7, e)]["cache_top_sv_share"]
        top_sv = data[(7, e)]["cache_top5_svs"][0]
        print(f"  {e:>5} edits: top_sv={top_sv:.1f}, share={share:.4f}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Visualize mechanism analysis results: null-space geometry trajectory.

Produces a multi-panel figure showing how cache_c properties evolve
from 1K to 10K edits, revealing the mechanism behind AlphaEdit's collapse.

Usage:
    uv run python -m analysis.plots.mechanism_analysis_plots
"""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

RESULTS_DIR = Path("results/mechanism_analysis")
OUTPUT_DIR = Path("results/figures/mechanism_analysis")


def load_mechanism_data(results_dir: Path) -> dict[int, list[dict]]:
    """Load mechanism analysis JSONL files, grouped by seed."""
    by_seed = {}
    # Scan seed subdirectories
    for seed_dir in sorted(results_dir.iterdir()):
        if not seed_dir.is_dir() or not seed_dir.name.startswith("seed"):
            continue
        seed = int(seed_dir.name.replace("seed", ""))
        records = []
        for f in sorted(seed_dir.glob("*.jsonl")):
            with open(f) as fh:
                for line in fh:
                    line = line.strip()
                    if line:
                        records.append(json.loads(line))
        if records:
            by_seed[seed] = records
    return by_seed


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    by_seed = load_mechanism_data(RESULTS_DIR)
    if not by_seed:
        print("No mechanism analysis data found.")
        return

    seeds = sorted(by_seed.keys())
    print(f"Loaded data for seeds: {seeds}")

    # Use first seed to determine structure
    all_records = []
    for records in by_seed.values():
        all_records.extend(records)

    layers = sorted(set(r["layer_idx"] for r in all_records))
    edits = sorted(set(r["total_edits"] for r in all_records))
    hidden_dim = all_records[0]["cache"]["hidden_dim"]

    print(f"Layers: {layers}")
    print(f"Edit counts: {edits}")
    print(f"Hidden dim: {hidden_dim}")

    # Build data indexed by (seed, layer, edit_count)
    data = {}
    for seed, records in by_seed.items():
        for r in records:
            key = (seed, r["layer_idx"], r["total_edits"])
            data[key] = r["cache"]

    # For plotting, average across seeds at each (layer, edits) point
    def get_metric(layer, edit_count, metric):
        values = []
        for seed in seeds:
            key = (seed, layer, edit_count)
            if key in data:
                values.append(data[key][metric])
        return np.mean(values) if values else np.nan

    # ─── Figure: 4-panel mechanism diagnostic ───
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    seed_str = ", ".join(str(s) for s in seeds)
    fig.suptitle(
        f"AlphaEdit Cache Geometry: Null-Space Exhaustion Trajectory\n"
        f"(Seeds: {seed_str}, mean across seeds)",
        fontsize=13, fontweight="bold",
    )

    colors = plt.cm.viridis(np.linspace(0.2, 0.9, len(layers)))

    # Panel A: Numerical rank vs edits
    ax = axes[0, 0]
    for i, layer in enumerate(layers):
        y = [get_metric(layer, e, "cache_numerical_rank") for e in edits]
        ax.plot(edits, y, "o-", color=colors[i], label=f"Layer {layer}", markersize=4)
        # Show per-seed as thin lines
        for seed in seeds:
            ys = [data.get((seed, layer, e), {}).get("cache_numerical_rank", np.nan) for e in edits]
            ax.plot(edits, ys, "-", color=colors[i], alpha=0.15, linewidth=1)
    ax.axhline(hidden_dim, color="red", linestyle="--", alpha=0.7, label=f"Hidden dim ({hidden_dim})")
    ax.set_xlabel("Total Edits")
    ax.set_ylabel("Numerical Rank")
    ax.set_title("A. Cache Numerical Rank (dimensions consumed)")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # Annotate null-space remaining at max edits
    max_edits = max(edits)
    rank_max = np.mean([get_metric(layer, max_edits, "cache_numerical_rank") for layer in layers])
    remaining = hidden_dim - rank_max
    ax.annotate(f"~{int(remaining)} dims\nremaining",
                xy=(max_edits, rank_max), xytext=(max_edits - 2000, rank_max + 1500),
                arrowprops=dict(arrowstyle="->", color="black"),
                fontsize=9)

    # Panel B: Effective rank vs edits
    ax = axes[0, 1]
    for i, layer in enumerate(layers):
        effective = [get_metric(layer, e, "cache_effective_rank") for e in edits]
        ax.plot(edits, effective, "o-", color=colors[i], label=f"Layer {layer}", markersize=4)
    avg_numerical = [np.mean([get_metric(layer, e, "cache_numerical_rank") for layer in layers]) for e in edits]
    ax.plot(edits, avg_numerical, "k--", alpha=0.5, label="Numerical rank (avg)")
    ax.set_xlabel("Total Edits")
    ax.set_ylabel("Effective Rank")
    ax.set_title("B. Effective Rank (spectral concentration)")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # Panel C: Condition number (log scale)
    ax = axes[1, 0]
    for i, layer in enumerate(layers):
        y = [get_metric(layer, e, "cache_condition") for e in edits]
        ax.semilogy(edits, y, "o-", color=colors[i], label=f"Layer {layer}", markersize=4)
    ax.set_xlabel("Total Edits")
    ax.set_ylabel("Condition Number")
    ax.set_title("C. Cache Condition Number (numerical stability)")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # Panel D: Consumption ratio
    ax = axes[1, 1]
    for i, layer in enumerate(layers):
        consumption = [get_metric(layer, e, "cache_numerical_rank") / hidden_dim for e in edits]
        eff_consumption = [get_metric(layer, e, "cache_effective_rank") / hidden_dim for e in edits]
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
    out_path = OUTPUT_DIR / "mechanism_cache_geometry.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"\nSaved: {out_path}")
    plt.close()

    # ─── Print key statistics ───
    print("\n" + "=" * 70)
    print("KEY FINDINGS")
    print("=" * 70)

    print(f"\nHidden dimension: {hidden_dim}")
    print(f"Seeds analyzed: {seeds}")

    print(f"\nNull-space consumption at collapse boundary (5K-7K edits):")
    for e in [5000, 7000]:
        if e not in edits:
            continue
        ranks = [get_metric(layer, e, "cache_numerical_rank") for layer in layers]
        eff_ranks = [get_metric(layer, e, "cache_effective_rank") for layer in layers]
        print(f"  {e} edits: numerical rank = {np.nanmean(ranks):.0f} "
              f"({np.nanmean(ranks)/hidden_dim*100:.1f}%), "
              f"effective rank = {np.nanmean(eff_ranks):.0f} "
              f"({np.nanmean(eff_ranks)/hidden_dim*100:.1f}%)")

    print(f"\nNull-space remaining at {max_edits} edits:")
    for layer in layers:
        nr = get_metric(layer, max_edits, "cache_numerical_rank")
        er = get_metric(layer, max_edits, "cache_effective_rank")
        print(f"  Layer {layer}: {hidden_dim - nr:.0f} dims free (numerical), "
              f"eff_rank/hidden = {er/hidden_dim:.2%}")

    print(f"\nEffective/Numerical rank ratio (spectral concentration):")
    for e in edits:
        ratios = [get_metric(layer, e, "cache_effective_rank") /
                  max(1, get_metric(layer, e, "cache_numerical_rank")) for layer in layers]
        print(f"  {e:>5} edits: {np.nanmean(ratios):.3f} (1.0 = uniform, lower = concentrated)")

    print(f"\nCondition number trajectory (layer-averaged):")
    for e in edits:
        conds = [get_metric(layer, e, "cache_condition") for layer in layers]
        print(f"  {e:>5} edits: {np.nanmean(conds):.2e}")


if __name__ == "__main__":
    main()

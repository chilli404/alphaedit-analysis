#!/usr/bin/env python3
"""
Controlled Coupling Experiment Plots.

Visualizes the comparison between low-coupling and high-coupling edit streams,
showing how semantic structure affects editing capacity.

4-panel figure:
  A: retention(t) low vs high coupling — the key result
  B: cache_effective_rank(t) per stream
  C: mean_removed_fraction(t) per stream
  D: mean_cache_condition(t) per stream

With vertical line at high-coupling collapse point and annotation of edit-count gap.

Usage:
    uv run python -m analysis.plots.controlled_coupling_plots
    uv run python -m analysis.plots.controlled_coupling_plots --results_dir results/controlled_coupling
"""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

# Paper-quality matplotlib settings
plt.rcParams.update({
    "font.size": 10,
    "axes.labelsize": 11,
    "axes.titlesize": 12,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "font.family": "serif",
})

STREAM_COLORS = {"low_coupling": "#2196F3", "high_coupling": "#E91E63"}
STREAM_LABELS = {"low_coupling": "Low Coupling", "high_coupling": "High Coupling"}


def load_stream_data(results_dir: Path, seed: int) -> dict[str, list[dict]]:
    """
    Load JSONL results for both streams of a given seed.

    Returns dict mapping stream name -> list of per-batch records.
    """
    streams = {}
    for stream_name in ["low_coupling", "high_coupling"]:
        pattern = f"{stream_name}_seed{seed}_*.jsonl"
        files = sorted(results_dir.glob(pattern))
        if not files:
            continue

        # Use most recent file
        jsonl_path = files[-1]
        records = []
        with open(jsonl_path) as f:
            for line in f:
                line = line.strip()
                if line:
                    records.append(json.loads(line))
        if records:
            streams[stream_name] = records

    return streams


def extract_timeseries(records: list[dict], metric_path: str) -> tuple[list, list]:
    """
    Extract (total_edits, metric_value) pairs from records.

    metric_path can be:
      - "mechanism.aggregate.mean_cache_effective_rank"
      - "evaluation.overall_efficacy"
      - etc.
    """
    x_vals = []
    y_vals = []

    parts = metric_path.split(".")
    for record in records:
        val = record
        for part in parts:
            if isinstance(val, dict):
                val = val.get(part)
            else:
                val = None
                break

        if val is not None and val != "inf":
            x_vals.append(record["total_edits"])
            y_vals.append(float(val))

    return x_vals, y_vals


def find_collapse_point(x_vals: list, y_vals: list, threshold: float = 0.5) -> int | None:
    """Find first edit count where metric drops below threshold."""
    for x, y in zip(x_vals, y_vals):
        if y < threshold:
            return x
    return None


def plot_controlled_coupling(
    streams: dict[str, list[dict]],
    seed: int,
    output_dir: Path,
):
    """Generate 4-panel comparison figure."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    fig.suptitle(
        f"Controlled Coupling Experiment: Semantic Structure → Editing Capacity\n"
        f"Seed {seed} | Low coupling (diverse) vs High coupling (clustered subjects)",
        fontsize=12, fontweight="bold",
    )

    # Panel A: Retention / Efficacy
    ax = axes[0, 0]
    collapse_points = {}
    for stream_name, records in streams.items():
        x, y = extract_timeseries(records, "evaluation.overall_efficacy")
        if x:
            ax.plot(x, y, "o-", color=STREAM_COLORS[stream_name],
                    label=STREAM_LABELS[stream_name], markersize=3, linewidth=1.5)
            collapse = find_collapse_point(x, y, threshold=0.5)
            if collapse:
                collapse_points[stream_name] = collapse

    if "high_coupling" in collapse_points:
        ax.axvline(collapse_points["high_coupling"], color=STREAM_COLORS["high_coupling"],
                   linestyle="--", alpha=0.5, linewidth=1)
    if "low_coupling" in collapse_points:
        ax.axvline(collapse_points["low_coupling"], color=STREAM_COLORS["low_coupling"],
                   linestyle="--", alpha=0.5, linewidth=1)

    # Annotate gap
    if "high_coupling" in collapse_points and "low_coupling" in collapse_points:
        gap = collapse_points["low_coupling"] - collapse_points["high_coupling"]
        mid = (collapse_points["high_coupling"] + collapse_points["low_coupling"]) / 2
        ax.annotate(f"Δ = {gap} edits",
                    xy=(mid, 0.5), fontsize=9, ha="center", color="black",
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow", alpha=0.8))

    ax.axhline(0.5, color="gray", linestyle=":", alpha=0.4)
    ax.set_xlabel("Total Edits")
    ax.set_ylabel("Overall Efficacy")
    ax.set_title("A. Retention: Efficacy Over Time")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1.05)

    # Panel B: Cache Effective Rank
    ax = axes[0, 1]
    for stream_name, records in streams.items():
        x, y = extract_timeseries(records, "mechanism.aggregate.mean_cache_effective_rank")
        if x:
            ax.plot(x, y, "o-", color=STREAM_COLORS[stream_name],
                    label=STREAM_LABELS[stream_name], markersize=3, linewidth=1.5)
    ax.set_xlabel("Total Edits")
    ax.set_ylabel("Effective Rank")
    ax.set_title("B. Cache Effective Rank (spectral concentration)")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Panel C: Removed Fraction (projection loss)
    ax = axes[1, 0]
    for stream_name, records in streams.items():
        x, y = extract_timeseries(records, "mechanism.aggregate.mean_removed_fraction")
        if x:
            ax.plot(x, y, "o-", color=STREAM_COLORS[stream_name],
                    label=STREAM_LABELS[stream_name], markersize=3, linewidth=1.5)
    ax.set_xlabel("Total Edits")
    ax.set_ylabel("Mean Removed Fraction")
    ax.set_title("C. Projection Loss (fraction removed by null-space)")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Panel D: Cache Condition Number
    ax = axes[1, 1]
    for stream_name, records in streams.items():
        x, y = extract_timeseries(records, "mechanism.aggregate.mean_cache_condition")
        if x:
            ax.semilogy(x, y, "o-", color=STREAM_COLORS[stream_name],
                        label=STREAM_LABELS[stream_name], markersize=3, linewidth=1.5)
    ax.set_xlabel("Total Edits")
    ax.set_ylabel("Condition Number")
    ax.set_title("D. Cache Condition Number (numerical stability)")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Add collapse line to mechanism panels if available
    if "high_coupling" in collapse_points:
        for ax in [axes[0, 1], axes[1, 0], axes[1, 1]]:
            ax.axvline(collapse_points["high_coupling"],
                       color=STREAM_COLORS["high_coupling"],
                       linestyle="--", alpha=0.3, linewidth=1)

    plt.tight_layout()
    out_path = output_dir / f"controlled_coupling_seed{seed}.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"  Saved: {out_path}")
    plt.close()

    return collapse_points


def main():
    parser = argparse.ArgumentParser(
        description="Plot controlled coupling experiment results"
    )
    parser.add_argument("--results_dir", type=str, default="results/controlled_coupling",
                        help="Directory with JSONL results")
    parser.add_argument("--output_dir", type=str, default="results/figures/controlled_coupling",
                        help="Output directory for plots")
    parser.add_argument("--seeds", type=int, nargs="+", default=[42],
                        help="Seeds to plot")
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("Controlled Coupling Plots")
    print(f"  Results dir: {results_dir}")
    print(f"  Output dir:  {output_dir}")
    print(f"  Seeds:       {args.seeds}")
    print("=" * 70)

    for seed in args.seeds:
        print(f"\n  Loading seed {seed}...")
        streams = load_stream_data(results_dir, seed)

        if not streams:
            print(f"  WARNING: No data found for seed {seed}")
            continue

        print(f"  Found streams: {list(streams.keys())}")
        for name, records in streams.items():
            print(f"    {name}: {len(records)} batches, "
                  f"{records[-1]['total_edits']} max edits")

        collapse_points = plot_controlled_coupling(streams, seed, output_dir)

        if collapse_points:
            print(f"\n  Collapse points (efficacy < 50%):")
            for name, edits in sorted(collapse_points.items()):
                print(f"    {STREAM_LABELS[name]}: {edits} edits")
            if "high_coupling" in collapse_points and "low_coupling" in collapse_points:
                gap = collapse_points["low_coupling"] - collapse_points["high_coupling"]
                print(f"    Capacity gap: {gap} edits "
                      f"({gap / collapse_points['low_coupling'] * 100:.0f}% earlier collapse)")

    print(f"\n{'=' * 70}")
    print("Done.")


if __name__ == "__main__":
    main()

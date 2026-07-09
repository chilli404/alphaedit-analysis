#!/usr/bin/env python3
"""
Null-Space Rank Consumption Analysis and Visualization.

Reads the JSONL output from src/nullspace_tracker.py and produces:
1. Rank consumption curve: how cache_c rank grows vs batch index
2. Consumption ratio timeline: fraction of null-space "used up"
3. Per-layer heatmap of rank consumption
4. Spectral decay plot: top singular values of cache_c over time
5. Summary statistics for the paper

This is the mechanistic analysis that explains WHY AlphaEdit degrades
at high edit counts — the null-space becomes saturated.

Usage:
    python analysis/nullspace_analysis.py --input results/nullspace_tracking/
    python analysis/nullspace_analysis.py --input results/nullspace_tracking/ --output_dir results/figures
"""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

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


def load_tracking_data(input_dir: Path) -> list[dict]:
    """Load all JSONL tracking files from the input directory."""
    records = []
    jsonl_files = sorted(input_dir.glob("rank_trace_*.jsonl"))

    if not jsonl_files:
        print(f"ERROR: No rank_trace_*.jsonl files found in {input_dir}")
        return []

    for jsonl_file in jsonl_files:
        with open(jsonl_file) as f:
            for line in f:
                line = line.strip()
                if line:
                    record = json.loads(line)
                    record["_source_file"] = jsonl_file.name
                    records.append(record)

    print(f"Loaded {len(records)} batch records from {len(jsonl_files)} files")
    return records


def records_to_dataframe(records: list[dict]) -> pd.DataFrame:
    """Flatten the nested records into a DataFrame for analysis."""
    rows = []
    for record in records:
        batch_idx = record["batch_idx"]
        num_requests = record["num_requests"]
        source = record.get("_source_file", "")

        for layer_str, layer_data in record.get("layers", {}).items():
            row = {
                "batch_idx": batch_idx,
                "total_edits": (batch_idx + 1) * num_requests,
                "num_requests": num_requests,
                "layer": int(layer_str),
                "source_file": source,
                **layer_data,
            }
            rows.append(row)

    return pd.DataFrame(rows)


def plot_rank_consumption_curve(df: pd.DataFrame, output_dir: Path) -> None:
    """
    KEY FIGURE: Shows how the covariance cache rank grows relative to
    the available null-space as edits accumulate.

    X-axis: Total edits applied
    Y-axis: Rank (numerical rank of cache_c)
    Lines: One per layer, colored by layer depth
    Horizontal dashed lines: null-space rank (capacity) for each layer
    """
    if df.empty:
        return

    layers = sorted(df["layer"].unique())
    fig, ax = plt.subplots(figsize=(8, 5))

    cmap = plt.cm.viridis(np.linspace(0.2, 0.9, len(layers)))

    for i, layer in enumerate(layers):
        layer_df = df[df["layer"] == layer].sort_values("batch_idx")

        if "cache_c_numerical_rank" in layer_df.columns:
            ax.plot(
                layer_df["total_edits"],
                layer_df["cache_c_numerical_rank"],
                "-o",
                color=cmap[i],
                markersize=3,
                label=f"Layer {layer}",
            )

        # Horizontal line showing null-space capacity
        if "nullspace_rank_initial" in layer_df.columns:
            ns_rank = layer_df["nullspace_rank_initial"].iloc[0]
            ax.axhline(
                y=ns_rank,
                color=cmap[i],
                linestyle="--",
                alpha=0.4,
                linewidth=0.8,
            )

    ax.set_xlabel("Total Sequential Edits Applied")
    ax.set_ylabel("Rank of Accumulated Covariance (cache_c)")
    ax.set_title("Null-Space Consumption: Cache Rank Growth vs Edit Count")
    ax.legend(loc="upper left", ncol=2, fontsize=8)

    # Add annotation
    ax.annotate(
        "Dashed lines = available\nnull-space dimensions",
        xy=(0.98, 0.02),
        xycoords="axes fraction",
        ha="right",
        va="bottom",
        fontsize=8,
        fontstyle="italic",
        color="gray",
    )

    output_path = output_dir / "nullspace_rank_consumption.pdf"
    fig.savefig(output_path)
    plt.close(fig)
    print(f"Saved: {output_path}")


def plot_consumption_ratio_timeline(df: pd.DataFrame, output_dir: Path) -> None:
    """
    Consumption ratio over time: what fraction of the null-space is "used".

    When this ratio approaches 1.0, the null-space is saturated and
    AlphaEdit can no longer avoid interfering with prior edits.
    """
    if df.empty or "consumption_ratio" not in df.columns:
        return

    layers = sorted(df["layer"].unique())
    fig, ax = plt.subplots(figsize=(8, 4.5))

    cmap = plt.cm.plasma(np.linspace(0.2, 0.85, len(layers)))

    for i, layer in enumerate(layers):
        layer_df = df[df["layer"] == layer].sort_values("batch_idx")
        ax.plot(
            layer_df["total_edits"],
            layer_df["consumption_ratio"],
            "-",
            color=cmap[i],
            linewidth=1.5,
            label=f"Layer {layer}",
        )

    ax.axhline(y=1.0, color="red", linestyle="--", alpha=0.6, linewidth=1.5)
    ax.axhline(y=0.5, color="orange", linestyle=":", alpha=0.4)

    ax.set_xlabel("Total Sequential Edits Applied")
    ax.set_ylabel("Consumption Ratio (cache rank / null-space rank)")
    ax.set_title("Null-Space Saturation Over Sequential Editing")
    ax.legend(loc="upper left", ncol=2, fontsize=8)
    ax.set_ylim(0, max(1.2, df["consumption_ratio"].max() * 1.1))

    # Annotate the danger zone
    ax.fill_between(
        ax.get_xlim(),
        0.8, 1.2,
        alpha=0.05,
        color="red",
        label="_nolegend_",
    )
    ax.text(
        0.98, 0.85,
        "Saturation zone",
        transform=ax.transAxes,
        ha="right",
        fontsize=8,
        color="red",
        alpha=0.7,
    )

    output_path = output_dir / "nullspace_consumption_ratio.pdf"
    fig.savefig(output_path)
    plt.close(fig)
    print(f"Saved: {output_path}")


def plot_layer_heatmap(df: pd.DataFrame, output_dir: Path) -> None:
    """
    Heatmap showing consumption ratio across (batch_idx, layer).
    Reveals which layers saturate first.
    """
    if df.empty or "consumption_ratio" not in df.columns:
        return

    pivot = df.pivot_table(
        values="consumption_ratio",
        index="layer",
        columns="batch_idx",
        aggfunc="mean",
    )

    if pivot.empty:
        return

    fig, ax = plt.subplots(figsize=(10, 4))
    sns.heatmap(
        pivot,
        ax=ax,
        cmap="YlOrRd",
        vmin=0,
        vmax=1.0,
        annot=False,
        cbar_kws={"label": "Consumption Ratio"},
    )

    ax.set_xlabel("Edit Batch Index")
    ax.set_ylabel("Layer")
    ax.set_title("Null-Space Consumption by Layer and Edit Batch")

    output_path = output_dir / "nullspace_layer_heatmap.pdf"
    fig.savefig(output_path)
    plt.close(fig)
    print(f"Saved: {output_path}")


def plot_spectral_decay(df: pd.DataFrame, output_dir: Path) -> None:
    """
    Shows how the top singular values of cache_c evolve over time.
    A spreading spectrum indicates diverse edit directions consuming
    more of the null-space.
    """
    if df.empty or "cache_c_top_svs" not in df.columns:
        return

    # Use the first layer's data for clarity
    layers = sorted(df["layer"].unique())
    target_layer = layers[len(layers) // 2]  # Middle layer

    layer_df = df[df["layer"] == target_layer].sort_values("batch_idx")

    # Extract top SVs
    svs_over_time = []
    for _, row in layer_df.iterrows():
        svs = row.get("cache_c_top_svs")
        if svs is not None and isinstance(svs, list):
            svs_over_time.append((row["total_edits"], svs))

    if not svs_over_time:
        return

    fig, ax = plt.subplots(figsize=(8, 4.5))

    # Plot top-k SVs as lines over time
    n_svs = min(5, min(len(s) for _, s in svs_over_time))
    colors = plt.cm.Blues(np.linspace(0.4, 1.0, n_svs))

    for k in range(n_svs):
        edits = [e for e, s in svs_over_time if len(s) > k]
        vals = [s[k] for _, s in svs_over_time if len(s) > k]
        ax.plot(edits, vals, "-o", color=colors[k], markersize=3,
                label=f"$\\sigma_{{{k+1}}}$")

    ax.set_xlabel("Total Sequential Edits Applied")
    ax.set_ylabel("Singular Value")
    ax.set_title(f"Spectral Evolution of cache_c (Layer {target_layer})")
    ax.legend(loc="upper left")
    ax.set_yscale("log")

    output_path = output_dir / "nullspace_spectral_decay.pdf"
    fig.savefig(output_path)
    plt.close(fig)
    print(f"Saved: {output_path}")


def plot_effective_rank_vs_batch(df: pd.DataFrame, output_dir: Path) -> None:
    """
    Effective rank (entropy-based) of cache_c vs batch index.
    This is a smoother measure than numerical rank, capturing how
    "spread out" the accumulated edit directions are.
    """
    if df.empty or "cache_c_effective_rank" not in df.columns:
        return

    layers = sorted(df["layer"].unique())
    fig, ax = plt.subplots(figsize=(8, 4.5))

    cmap = plt.cm.cool(np.linspace(0.2, 0.9, len(layers)))

    for i, layer in enumerate(layers):
        layer_df = df[df["layer"] == layer].sort_values("batch_idx")
        ax.plot(
            layer_df["total_edits"],
            layer_df["cache_c_effective_rank"],
            "-",
            color=cmap[i],
            linewidth=1.5,
            label=f"Layer {layer}",
        )

    ax.set_xlabel("Total Sequential Edits Applied")
    ax.set_ylabel("Effective Rank (entropy-based)")
    ax.set_title("Effective Dimensionality of Edit Directions Over Time")
    ax.legend(loc="upper left", ncol=2, fontsize=8)

    output_path = output_dir / "nullspace_effective_rank.pdf"
    fig.savefig(output_path)
    plt.close(fig)
    print(f"Saved: {output_path}")


def generate_summary_table(df: pd.DataFrame, output_dir: Path) -> None:
    """
    Summary table for paper: per-layer null-space capacity, final consumption,
    and estimated saturation point (extrapolated edit count at ratio=1.0).
    """
    if df.empty:
        return

    layers = sorted(df["layer"].unique())
    summary_rows = []

    for layer in layers:
        layer_df = df[df["layer"] == layer].sort_values("batch_idx")

        ns_rank = layer_df["nullspace_rank_initial"].iloc[0] if "nullspace_rank_initial" in layer_df.columns else None
        hidden_dim = layer_df["hidden_dim"].iloc[0] if "hidden_dim" in layer_df.columns else None

        final_cache_rank = None
        final_ratio = None
        if "cache_c_numerical_rank" in layer_df.columns and len(layer_df) > 0:
            final_cache_rank = layer_df["cache_c_numerical_rank"].iloc[-1]
        if "consumption_ratio" in layer_df.columns and len(layer_df) > 0:
            final_ratio = layer_df["consumption_ratio"].iloc[-1]

        # Estimate saturation point via linear extrapolation
        saturation_edits = None
        if "consumption_ratio" in layer_df.columns and len(layer_df) >= 2:
            ratios = layer_df["consumption_ratio"].values
            edits = layer_df["total_edits"].values
            # Use last two points for linear extrapolation
            if ratios[-1] < 1.0 and ratios[-1] > ratios[0]:
                slope = (ratios[-1] - ratios[0]) / (edits[-1] - edits[0])
                if slope > 0:
                    remaining = (1.0 - ratios[-1]) / slope
                    saturation_edits = int(edits[-1] + remaining)

        summary_rows.append({
            "layer": layer,
            "hidden_dim": hidden_dim,
            "nullspace_rank": ns_rank,
            "nullspace_fraction": round(ns_rank / hidden_dim, 3) if ns_rank and hidden_dim else None,
            "final_cache_rank": final_cache_rank,
            "final_consumption_ratio": round(final_ratio, 4) if final_ratio else None,
            "estimated_saturation_edits": saturation_edits,
        })

    summary_df = pd.DataFrame(summary_rows)
    output_path = output_dir / "nullspace_summary.csv"
    summary_df.to_csv(output_path, index=False)
    print(f"Saved: {output_path}")

    # Print for paper
    print("\n=== Null-Space Capacity Summary ===")
    print(summary_df.to_string(index=False))


def main():
    parser = argparse.ArgumentParser(
        description="Analyze null-space rank consumption data"
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("results/nullspace_tracking"),
        help="Directory containing rank_trace_*.jsonl files",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path("results/figures"),
        help="Output directory for figures",
    )
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    records = load_tracking_data(args.input)
    if not records:
        return

    df = records_to_dataframe(records)
    print(f"DataFrame: {len(df)} rows, layers: {sorted(df['layer'].unique())}")
    print(f"Edit range: {df['total_edits'].min()} to {df['total_edits'].max()}")
    print()

    print("=== Generating Null-Space Analysis Figures ===\n")
    plot_rank_consumption_curve(df, args.output_dir)
    plot_consumption_ratio_timeline(df, args.output_dir)
    plot_layer_heatmap(df, args.output_dir)
    plot_spectral_decay(df, args.output_dir)
    plot_effective_rank_vs_batch(df, args.output_dir)
    generate_summary_table(df, args.output_dir)
    print("\n=== Done ===")


if __name__ == "__main__":
    main()

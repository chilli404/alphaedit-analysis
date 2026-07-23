#!/usr/bin/env python3
"""
Matched Ordering Comparison: AlphaEdit vs MEMIT-Seq under key-clustered vs key-dispersed streams.

Shows how edit ordering (clustered vs dispersed key geometry) affects
degradation for each method. The central finding: AlphaEdit degrades much
faster under key-dispersed ordering (null-space exhaustion), while MEMIT-Seq
is more robust to ordering.

Outputs:
  - matched_ordering_comparison.png — 2x3 panel: efficacy/para/neigh × method
  - matched_ordering_efficacy.png — Single panel: all 4 curves (method × ordering)
  - matched_ordering_first1k.png — First-1K cohort retention (earliest edits forgotten?)
  - matched_ordering_summary.csv — Tabular data

Usage:
    python -m analysis.plots.matched_ordering_comparison
    python -m analysis.plots.matched_ordering_comparison --seeds 42 2024
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from analysis.loaders import load_matched_ordering_all_evals

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

COLORS = {
    ("AlphaEdit", "key_clustered"): "#2196F3",
    ("AlphaEdit", "key_dispersed"): "#90CAF9",
    ("MEMIT-Seq-lp1.0-ld0.0-cache0", "key_clustered"): "#4CAF50",
    ("MEMIT-Seq-lp1.0-ld0.0-cache0", "key_dispersed"): "#A5D6A7",
}

MARKERS = {
    ("AlphaEdit", "key_clustered"): "o",
    ("AlphaEdit", "key_dispersed"): "^",
    ("MEMIT-Seq-lp1.0-ld0.0-cache0", "key_clustered"): "s",
    ("MEMIT-Seq-lp1.0-ld0.0-cache0", "key_dispersed"): "D",
}

LABELS = {
    ("AlphaEdit", "key_clustered"): "AlphaEdit (clustered)",
    ("AlphaEdit", "key_dispersed"): "AlphaEdit (dispersed)",
    ("MEMIT-Seq-lp1.0-ld0.0-cache0", "key_clustered"): "MEMIT-Seq (clustered)",
    ("MEMIT-Seq-lp1.0-ld0.0-cache0", "key_dispersed"): "MEMIT-Seq (dispersed)",
}


def _key(method: str, ordering: str) -> tuple:
    return (method, ordering)


def plot_efficacy_comparison(df: pd.DataFrame, output_dir: Path) -> None:
    """Single-panel efficacy comparison: all method x ordering curves."""
    fig, ax = plt.subplots(figsize=(8, 5))

    for (method, ordering), group in df.groupby(["method", "ordering"]):
        key = _key(method, ordering)
        color = COLORS.get(key, "#9E9E9E")
        marker = MARKERS.get(key, "x")
        label = LABELS.get(key, f"{method} ({ordering})")

        # Per-seed thin lines
        for seed in group["seed"].unique():
            seed_data = group[group["seed"] == seed].sort_values("total_edits")
            ax.plot(seed_data["total_edits"], seed_data["efficacy"],
                    "-", color=color, alpha=0.25, linewidth=1)

        # Mean across seeds
        mean = group.groupby("total_edits")["efficacy"].mean().reset_index().sort_values("total_edits")
        ax.plot(mean["total_edits"], mean["efficacy"],
                f"-{marker}", color=color, linewidth=2.2, markersize=6, label=label)

        # Annotate final point
        if not mean.empty:
            final = mean.iloc[-1]
            ax.annotate(
                f"{final['efficacy']:.1%}",
                xy=(final["total_edits"], final["efficacy"]),
                xytext=(10, -5 if "dispersed" in ordering else 5),
                textcoords="offset points",
                fontsize=8, color=color, fontweight="bold",
            )

    ax.set_xlabel("Total Sequential Edits")
    ax.set_ylabel("Efficacy (all facts)")
    ax.set_title("Edit Retention Under Key-Geometry Orderings\n(same 5K facts, different insertion order)")
    ax.legend(loc="lower left", fontsize=9)
    ax.set_ylim(0.5, 1.02)
    ax.axhline(y=0.9, color="gray", linestyle=":", alpha=0.3, linewidth=0.8)

    all_edits = sorted(df["total_edits"].unique())
    ax.set_xticks(all_edits)
    ax.tick_params(axis="x", rotation=45)

    fig.tight_layout()
    output_path = output_dir / "matched_ordering_efficacy.png"
    fig.savefig(output_path)
    plt.close(fig)
    print(f"Saved: {output_path}")


def plot_first1k_retention(df: pd.DataFrame, output_dir: Path) -> None:
    """First-1K cohort efficacy over time — shows forgetting of earliest edits."""
    if "first_1k_efficacy" not in df.columns or df["first_1k_efficacy"].isna().all():
        print("  Skipping first_1k plot (no data)")
        return

    fig, ax = plt.subplots(figsize=(8, 5))

    for (method, ordering), group in df.groupby(["method", "ordering"]):
        key = _key(method, ordering)
        color = COLORS.get(key, "#9E9E9E")
        marker = MARKERS.get(key, "x")
        label = LABELS.get(key, f"{method} ({ordering})")

        mean = group.groupby("total_edits")["first_1k_efficacy"].mean().reset_index().sort_values("total_edits")
        if mean["first_1k_efficacy"].isna().all():
            continue
        ax.plot(mean["total_edits"], mean["first_1k_efficacy"],
                f"-{marker}", color=color, linewidth=2.2, markersize=6, label=label)

    ax.set_xlabel("Total Sequential Edits")
    ax.set_ylabel("First-1K Cohort Efficacy")
    ax.set_title("Retention of Earliest Edits\n(Do early facts get overwritten?)")
    ax.legend(loc="lower left", fontsize=9)
    ax.set_ylim(0.4, 1.02)

    all_edits = sorted(df["total_edits"].unique())
    ax.set_xticks(all_edits)
    ax.tick_params(axis="x", rotation=45)

    fig.tight_layout()
    output_path = output_dir / "matched_ordering_first1k.png"
    fig.savefig(output_path)
    plt.close(fig)
    print(f"Saved: {output_path}")


def plot_3panel(df: pd.DataFrame, output_dir: Path) -> None:
    """3-panel: efficacy, paraphrase, neighborhood."""
    metrics = ["efficacy", "paraphrase", "neighborhood"]
    available = [m for m in metrics if m in df.columns and df[m].notna().any()]

    fig, axes = plt.subplots(1, len(available), figsize=(5 * len(available), 4.5))
    if len(available) == 1:
        axes = [axes]

    for ax, metric in zip(axes, available):
        for (method, ordering), group in df.groupby(["method", "ordering"]):
            key = _key(method, ordering)
            color = COLORS.get(key, "#9E9E9E")
            marker = MARKERS.get(key, "x")
            label = LABELS.get(key, f"{method} ({ordering})")

            mean = group.groupby("total_edits")[metric].mean().reset_index().sort_values("total_edits")
            ax.plot(mean["total_edits"], mean[metric],
                    f"-{marker}", color=color, linewidth=2, markersize=5, label=label)

        ax.set_xlabel("Total Sequential Edits")
        ax.set_ylabel(metric.capitalize())
        ax.set_title(metric.capitalize())
        ax.legend(loc="lower left" if metric == "efficacy" else "best", fontsize=8)
        ax.set_ylim(0, 1.05)

        all_edits = sorted(df["total_edits"].unique())
        ax.set_xticks(all_edits)
        ax.tick_params(axis="x", rotation=45)

    fig.suptitle(
        "Matched Ordering: Method x Key-Geometry Comparison\n"
        "(Llama-3-8B-Instruct, 5K edits, 100-edit batches)",
        y=1.02, fontsize=12,
    )
    fig.tight_layout()

    output_path = output_dir / "matched_ordering_comparison.png"
    fig.savefig(output_path)
    plt.close(fig)
    print(f"Saved: {output_path}")


def print_summary(df: pd.DataFrame) -> None:
    """Print comparison table."""
    print("\n" + "=" * 90)
    print("MATCHED ORDERING COMPARISON")
    print("=" * 90)

    for (method, ordering), group in df.groupby(["method", "ordering"]):
        label = LABELS.get(_key(method, ordering), f"{method} ({ordering})")
        print(f"\n{'─' * 60}")
        print(f"  {label}")
        print(f"{'─' * 60}")

        grouped = group.groupby("total_edits").agg({
            "efficacy": "mean",
            "paraphrase": "mean",
            "neighborhood": "mean",
            "first_1k_efficacy": "mean",
            "latest_1k_efficacy": "mean",
            "retention_auc": "mean",
        })

        print(f"{'Edits':>7} | {'Effic':>7} | {'Paraph':>7} | {'Neighb':>7} | {'1st1K':>7} | {'Lat1K':>7} | {'AUC':>6}")
        print(f"{'-'*7}-+-{'-'*7}-+-{'-'*7}-+-{'-'*7}-+-{'-'*7}-+-{'-'*7}-+-{'-'*6}")

        for edits, row in grouped.iterrows():
            vals = [
                f"{row['efficacy']:.4f}",
                f"{row['paraphrase']:.4f}" if not np.isnan(row['paraphrase']) else "   N/A",
                f"{row['neighborhood']:.4f}" if not np.isnan(row['neighborhood']) else "   N/A",
                f"{row['first_1k_efficacy']:.4f}" if not np.isnan(row['first_1k_efficacy']) else "   N/A",
                f"{row['latest_1k_efficacy']:.4f}" if not np.isnan(row['latest_1k_efficacy']) else "   N/A",
                f"{row['retention_auc']:.4f}" if not np.isnan(row['retention_auc']) else "  N/A",
            ]
            print(f"{int(edits):>7} | {' | '.join(vals)}")

    # Ordering effect comparison
    print(f"\n{'=' * 90}")
    print("ORDERING EFFECT (dispersed - clustered efficacy)")
    print(f"{'=' * 90}")

    for method in df["method"].unique():
        clust = df[(df["method"] == method) & (df["ordering"] == "key_clustered")]
        disp = df[(df["method"] == method) & (df["ordering"] == "key_dispersed")]

        if clust.empty or disp.empty:
            continue

        clust_mean = clust.groupby("total_edits")["efficacy"].mean()
        disp_mean = disp.groupby("total_edits")["efficacy"].mean()

        print(f"\n  {method}:")
        print(f"  {'Edits':>7} | {'Clustered':>10} | {'Dispersed':>10} | {'Delta':>8}")
        print(f"  {'-'*7}-+-{'-'*10}-+-{'-'*10}-+-{'-'*8}")

        all_edits = sorted(set(clust_mean.index) | set(disp_mean.index))
        for edits in all_edits:
            c = clust_mean.get(edits, np.nan)
            d = disp_mean.get(edits, np.nan)
            delta = d - c if not (np.isnan(c) or np.isnan(d)) else np.nan
            c_s = f"{c:.4f}" if not np.isnan(c) else "     —"
            d_s = f"{d:.4f}" if not np.isnan(d) else "     —"
            delta_s = f"{delta:+.4f}" if not np.isnan(delta) else "    —"
            print(f"  {int(edits):>7} | {c_s:>10} | {d_s:>10} | {delta_s:>8}")

    print("\n" + "=" * 90)


def main():
    parser = argparse.ArgumentParser(
        description="Matched ordering comparison: AlphaEdit vs MEMIT-Seq under different key-geometry orderings"
    )
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 2024],
                        help="Seeds to include (default: 42 2024)")
    parser.add_argument("--output_dir", type=Path,
                        default=Path("results/figures/matched_ordering"),
                        help="Output directory for figures")
    parser.add_argument("--algs", nargs="+",
                        default=["AlphaEdit", "MEMIT-Seq-lp1.0-ld0.0-cache0"],
                        help="Algorithms to compare")
    parser.add_argument("--orderings", nargs="+",
                        default=["key_clustered", "key_dispersed"],
                        help="Orderings to compare")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    print("=== Matched Ordering Comparison ===\n")
    print(f"Seeds: {args.seeds}")
    print(f"Algorithms: {args.algs}")
    print(f"Orderings: {args.orderings}")
    print(f"Output: {args.output_dir}\n")

    # Load all data
    rows = load_matched_ordering_all_evals(
        seeds=args.seeds,
        orderings=args.orderings,
        algs=args.algs,
    )

    if not rows:
        print("\nERROR: No matched ordering full_eval data found.")
        print("Expected at: results/matched_ordering/{ALG}/{ORDERING}/seed{N}/full_eval_seed{N}.json")
        return

    df = pd.DataFrame(rows)
    print(f"Total data points: {len(df)}")
    print(f"Methods: {df['method'].unique().tolist()}")
    print(f"Orderings: {df['ordering'].unique().tolist()}")
    print(f"Seeds: {df['seed'].unique().tolist()}")
    print(f"Edit counts: {sorted(df['total_edits'].unique().tolist())}")

    # Save CSV
    csv_path = args.output_dir / "matched_ordering_summary.csv"
    df.to_csv(csv_path, index=False)
    print(f"\nSaved: {csv_path}")

    # Print summary
    print_summary(df)

    # Generate plots
    print("\nGenerating plots...")
    plot_efficacy_comparison(df, args.output_dir)
    plot_first1k_retention(df, args.output_dir)
    plot_3panel(df, args.output_dir)

    print("\n=== Done ===")


if __name__ == "__main__":
    main()

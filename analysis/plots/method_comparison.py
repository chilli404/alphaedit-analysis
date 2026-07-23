#!/usr/bin/env python3
"""
Method Comparison: AlphaEdit vs AlphaEdit-Poly2 vs MEMIT-Seq

Generates failure curve comparison showing how three methods degrade
as total sequential edits increase:
  - AlphaEdit (linear, null-space projection)
  - AlphaEdit-Poly2 (polynomial kernel, null-space projection)
  - MEMIT-Seq (no projection, sequential regularization)

Outputs:
  - method_comparison.png — Main figure: efficacy/paraphrase/neighborhood
  - method_comparison_summary.csv — Tabular data

Usage:
    python -m analysis.plots.method_comparison
    python -m analysis.plots.method_comparison --seeds 42
    python -m analysis.plots.method_comparison --output_dir results/figures/comparison
"""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from analysis.stats.aggregate import extract_metrics_from_case

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

RESULTS = Path("results")

import re

def _discover_memit_seq_variants(seeds: list[int]) -> list[tuple[float, float]]:
    """Auto-discover MEMIT-Seq variants from failure_curve_checkpointed results."""
    base = RESULTS / "failure_curve_checkpointed"
    pattern = re.compile(r"^MEMIT-Seq-lp([\d.]+)-ld([\d.e-]+)-cache\d+$")
    variants = set()

    for seed in seeds:
        seed_dir = base / f"seed{seed}"
        if not seed_dir.exists():
            continue
        for edits_dir in seed_dir.iterdir():
            if not edits_dir.is_dir() or not edits_dir.name.endswith("edits"):
                continue
            for subdir in edits_dir.iterdir():
                if not subdir.is_dir():
                    continue
                m = pattern.match(subdir.name)
                if m:
                    lp, ld = float(m.group(1)), float(m.group(2))
                    variants.add((lp, ld))

    return sorted(variants)


COLORS = {
    "AlphaEdit": "#2196F3",
    "AlphaEdit-Poly2": "#E91E63",
    "MEMIT-Seq": "#4CAF50",
}
MARKERS = {
    "AlphaEdit": "o",
    "AlphaEdit-Poly2": "D",
    "MEMIT-Seq": "s",
}


def _color_for(method: str) -> str:
    """Return color for method, falling back to MEMIT-Seq base for variants."""
    if method in COLORS:
        return COLORS[method]
    if method.startswith("MEMIT-Seq"):
        return COLORS["MEMIT-Seq"]
    return "#9E9E9E"


def _marker_for(method: str) -> str:
    """Return marker for method, falling back to MEMIT-Seq base for variants."""
    if method in MARKERS:
        return MARKERS[method]
    if method.startswith("MEMIT-Seq"):
        return MARKERS["MEMIT-Seq"]
    return "^"


def _load_failure_curve(seed: int, alg_name: str, method_label: str, base_dir: Path | None = None) -> list[dict]:
    """Load per-case results from failure_curve_checkpointed for any algorithm.

    Scans: {base_dir}/seed{N}/{edits}edits/{alg_name}/run_000/*_edits-case_*.json
    """
    if base_dir is None:
        base_dir = RESULTS / "failure_curve_checkpointed"
    base = base_dir / f"seed{seed}"
    if not base.exists():
        return []

    rows = []
    for edits_dir in sorted(base.iterdir()):
        if not edits_dir.is_dir() or not edits_dir.name.endswith("edits"):
            continue
        total_edits = int(edits_dir.name.replace("edits", ""))

        run_dir = edits_dir / alg_name / "run_000"
        if not run_dir.exists():
            continue

        case_files = list(run_dir.glob("*_edits-case_*.json"))
        if not case_files:
            continue

        metrics = _aggregate_cases(case_files)
        if metrics:
            metrics["total_edits"] = total_edits
            metrics["method"] = method_label
            metrics["seed"] = seed
            metrics["n_cases"] = len(case_files)
            rows.append(metrics)

    return rows


def load_alphaedit_failure_curve(seed: int) -> list[dict]:
    """Load AlphaEdit per-case results from failure_curve_checkpointed."""
    return _load_failure_curve(seed, "AlphaEdit", "AlphaEdit")


def load_poly2_results(seed: int) -> list[dict]:
    """Load AlphaEdit-Poly2 per-case results from polykernel_editor."""
    base = RESULTS / "polykernel_editor" / f"seed{seed}"
    if not base.exists():
        return []

    rows = []
    for edits_dir in sorted(base.iterdir()):
        if not edits_dir.is_dir() or not edits_dir.name.endswith("edits"):
            continue
        total_edits = int(edits_dir.name.replace("edits", ""))

        # Primary: AlphaEdit-poly2/run_000/
        run_dir = edits_dir / "AlphaEdit-poly2" / "run_000"
        if not run_dir.exists():
            continue

        case_files = list(run_dir.glob("*_edits-case_*.json"))
        if not case_files:
            continue

        metrics = _aggregate_cases(case_files)
        if metrics:
            metrics["total_edits"] = total_edits
            metrics["method"] = "AlphaEdit-Poly2"
            metrics["seed"] = seed
            metrics["n_cases"] = len(case_files)
            rows.append(metrics)

    return rows


def load_memit_seq_results(seed: int, lambda_prev: float = 1.0, lambda_delta: float = 0.0) -> list[dict]:
    """Load MEMIT-Seq results from the unified failure_curve_checkpointed structure.

    Source: failure_curve_checkpointed/seed{N}/{edits}edits/{variant}/run_000/ (per-case JSONs)

    The method label uses the full variant name (e.g. "MEMIT-Seq-lp1.0-ld0.0-cache0")
    so different configurations are distinguishable in plots.
    """
    variant = f"MEMIT-Seq-lp{lambda_prev}-ld{lambda_delta}-cache0"
    return _load_failure_curve(seed, variant, variant)



def _aggregate_cases(case_files: list[Path]) -> dict | None:
    """Aggregate per-case JSON files into summary metrics."""
    efficacy_vals = []
    paraphrase_vals = []
    neighborhood_vals = []

    for cf in case_files:
        with open(cf) as f:
            case = json.load(f)

        post = case.get("post", {})

        rc = post.get("rewrite_prompts_correct", [])
        if rc:
            efficacy_vals.append(sum(rc) / len(rc))

        pc = post.get("paraphrase_prompts_correct", [])
        if pc:
            paraphrase_vals.append(sum(pc) / len(pc))

        nc = post.get("neighborhood_prompts_correct", [])
        if nc:
            neighborhood_vals.append(sum(nc) / len(nc))

    if not efficacy_vals:
        return None

    return {
        "efficacy": np.mean(efficacy_vals),
        "paraphrase": np.mean(paraphrase_vals) if paraphrase_vals else np.nan,
        "neighborhood": np.mean(neighborhood_vals) if neighborhood_vals else np.nan,
    }


def plot_comparison(df: pd.DataFrame, output_dir: Path) -> None:
    """Main comparison figure: efficacy, paraphrase, neighborhood vs edit count."""
    metrics = ["efficacy", "paraphrase", "neighborhood"]
    available = [m for m in metrics if m in df.columns and df[m].notna().any()]

    fig, axes = plt.subplots(1, len(available), figsize=(5 * len(available), 4.5))
    if len(available) == 1:
        axes = [axes]

    # Stable ordering: AlphaEdit first, Poly2 second, then MEMIT-Seq variants sorted
    all_methods = df["method"].unique().tolist()
    ordered = []
    for fixed in ["AlphaEdit", "AlphaEdit-Poly2"]:
        if fixed in all_methods:
            ordered.append(fixed)
    for m in sorted(all_methods):
        if m not in ordered:
            ordered.append(m)
    methods = ordered

    for ax, metric in zip(axes, available):
        for method in methods:
            method_data = df[df["method"] == method].copy()
            if method_data.empty:
                continue

            color = _color_for(method)
            marker = _marker_for(method)

            # Per-seed lines (thin)
            for seed in method_data["seed"].unique():
                seed_data = method_data[method_data["seed"] == seed].sort_values("total_edits")
                ax.plot(
                    seed_data["total_edits"], seed_data[metric],
                    "-", color=color, alpha=0.25, linewidth=1,
                )

            # Mean across seeds (thick)
            mean_data = (
                method_data.groupby("total_edits")[metric]
                .mean().reset_index().sort_values("total_edits")
            )
            ax.plot(
                mean_data["total_edits"], mean_data[metric],
                f"-{marker}", color=color, linewidth=2.2, markersize=6, label=method,
            )

        ax.set_xlabel("Total Sequential Edits")
        ax.set_ylabel(metric.capitalize())
        ax.set_title(metric.capitalize())
        ax.legend(loc="lower left" if metric == "efficacy" else "best")
        ax.set_ylim(0, 1.05)
        ax.axhline(y=0.5, color="gray", linestyle=":", alpha=0.3)

        all_edits = sorted(df["total_edits"].unique())
        # Use reasonable tick spacing
        if len(all_edits) > 8:
            ax.set_xticks(all_edits[::2])
        else:
            ax.set_xticks(all_edits)
        ax.tick_params(axis="x", rotation=45)

    method_str = " vs ".join(methods)
    fig.suptitle(
        f"Method Comparison: {method_str}\n"
        "(Llama-3-8B-Instruct, MultiCounterFact, 100-edit batches)",
        y=1.02, fontsize=12,
    )
    fig.tight_layout()

    output_path = output_dir / "method_comparison.png"
    fig.savefig(output_path)
    plt.close(fig)
    print(f"Saved: {output_path}")


def plot_efficacy_focused(df: pd.DataFrame, output_dir: Path) -> None:
    """Single-panel efficacy comparison with annotations."""
    fig, ax = plt.subplots(figsize=(8, 5))

    # Stable ordering: AlphaEdit first, Poly2 second, then MEMIT-Seq variants sorted
    all_methods = df["method"].unique().tolist()
    ordered = []
    for fixed in ["AlphaEdit", "AlphaEdit-Poly2"]:
        if fixed in all_methods:
            ordered.append(fixed)
    for m in sorted(all_methods):
        if m not in ordered:
            ordered.append(m)
    methods = ordered

    for method in methods:
        method_data = df[df["method"] == method].copy()
        if method_data.empty:
            continue

        color = _color_for(method)
        marker = _marker_for(method)

        # Per-seed (thin)
        for seed in method_data["seed"].unique():
            seed_data = method_data[method_data["seed"] == seed].sort_values("total_edits")
            ax.plot(seed_data["total_edits"], seed_data["efficacy"],
                    "-", color=color, alpha=0.2, linewidth=1)

        # Mean
        mean_data = (
            method_data.groupby("total_edits")["efficacy"]
            .mean().reset_index().sort_values("total_edits")
        )
        ax.plot(mean_data["total_edits"], mean_data["efficacy"],
                f"-{marker}", color=color, linewidth=2.5, markersize=7, label=method)

        # Annotate final point
        if not mean_data.empty:
            final = mean_data.iloc[-1]
            ax.annotate(
                f"{final['efficacy']:.1%}",
                xy=(final["total_edits"], final["efficacy"]),
                xytext=(10, 5), textcoords="offset points",
                fontsize=8, color=color, fontweight="bold",
            )

    ax.set_xlabel("Total Sequential Edits")
    ax.set_ylabel("Efficacy (fraction of edits retained)")
    ax.set_title("Edit Retention: Linear vs Polynomial Kernel vs Sequential Regularization")
    ax.legend(loc="upper right", fontsize=10)
    ax.set_ylim(0, 1.05)
    ax.axhline(y=0.5, color="gray", linestyle=":", alpha=0.3, linewidth=0.8)

    # Shade degradation zone
    ax.axvspan(5000, 7000, alpha=0.04, color="red")

    all_edits = sorted(df["total_edits"].unique())
    ax.set_xticks(all_edits)
    ax.tick_params(axis="x", rotation=45)

    fig.tight_layout()
    output_path = output_dir / "method_comparison_efficacy.png"
    fig.savefig(output_path)
    plt.close(fig)
    print(f"Saved: {output_path}")


def print_summary(df: pd.DataFrame) -> None:
    """Print comparison table."""
    all_methods = df["method"].unique().tolist()
    ordered = []
    for fixed in ["AlphaEdit", "AlphaEdit-Poly2"]:
        if fixed in all_methods:
            ordered.append(fixed)
    for m in sorted(all_methods):
        if m not in ordered:
            ordered.append(m)
    methods = ordered

    print("\n" + "=" * 80)
    print("METHOD COMPARISON SUMMARY")
    print("=" * 80)

    for method in methods:
        method_data = df[df["method"] == method].sort_values("total_edits")
        if method_data.empty:
            continue

        print(f"\n{'─' * 50}")
        print(f"  {method}")
        print(f"{'─' * 50}")

        grouped = method_data.groupby("total_edits")[["efficacy", "paraphrase", "neighborhood", "n_cases"]].mean()
        print(f"{'Edits':>7} | {'N':>5} | {'Effic':>7} | {'Paraph':>7} | {'Neighb':>7}")
        print(f"{'-'*7}-+-{'-'*5}-+-{'-'*7}-+-{'-'*7}-+-{'-'*7}")

        for edits, row in grouped.iterrows():
            eff = f"{row['efficacy']:.4f}" if not np.isnan(row['efficacy']) else "   N/A"
            par = f"{row['paraphrase']:.4f}" if not np.isnan(row.get('paraphrase', np.nan)) else "   N/A"
            nei = f"{row['neighborhood']:.4f}" if not np.isnan(row.get('neighborhood', np.nan)) else "   N/A"
            print(f"{int(edits):>7} | {int(row['n_cases']):>5} | {eff} | {par} | {nei}")

    # Pairwise comparison at common checkpoints
    ae = df[df["method"] == "AlphaEdit"].groupby("total_edits")["efficacy"].mean()
    poly2 = df[df["method"] == "AlphaEdit-Poly2"].groupby("total_edits")["efficacy"].mean()
    # Find all MEMIT-Seq variants
    seq_methods = [m for m in methods if m.startswith("MEMIT-Seq")]

    if ae.empty:
        print("\n" + "=" * 80)
        return

    for seq_name in seq_methods:
        seq = df[df["method"] == seq_name].groupby("total_edits")["efficacy"].mean()

        print(f"\n{'─' * 60}")
        print(f"  Δ Efficacy (Poly2 − AlphaEdit) and ({seq_name} − AlphaEdit)")
        print(f"{'─' * 60}")

        all_edits = sorted(set(ae.index) | set(poly2.index) | set(seq.index))

        print(f"{'Edits':>7} | {'AE':>7} | {'Poly2':>7} | {'Δ P-AE':>7} | {'Seq':>7} | {'Δ S-AE':>7}")
        print(f"{'-'*7}-+-{'-'*7}-+-{'-'*7}-+-{'-'*7}-+-{'-'*7}-+-{'-'*7}")

        for edits in all_edits:
            ae_val = ae.get(edits, np.nan)
            p2_val = poly2.get(edits, np.nan)
            sq_val = seq.get(edits, np.nan)
            d_p = p2_val - ae_val if not (np.isnan(p2_val) or np.isnan(ae_val)) else np.nan
            d_s = sq_val - ae_val if not (np.isnan(sq_val) or np.isnan(ae_val)) else np.nan

            ae_s = f"{ae_val:.4f}" if not np.isnan(ae_val) else "    —"
            p2_s = f"{p2_val:.4f}" if not np.isnan(p2_val) else "    —"
            sq_s = f"{sq_val:.4f}" if not np.isnan(sq_val) else "    —"
            dp_s = f"{d_p:+.4f}" if not np.isnan(d_p) else "    —"
            ds_s = f"{d_s:+.4f}" if not np.isnan(d_s) else "    —"

            print(f"{int(edits):>7} | {ae_s} | {p2_s} | {dp_s} | {sq_s} | {ds_s}")

    print("\n" + "=" * 80)


def main():
    parser = argparse.ArgumentParser(description="Method comparison: AlphaEdit vs Poly2 vs MEMIT-Seq")
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 2024],
                        help="Seeds to include (default: 42 2024)")
    parser.add_argument("--output_dir", type=Path,
                        default=Path("results/figures/method_comparison"),
                        help="Output directory for figures")
    parser.add_argument("--lambda_prev", type=float, nargs="+", default=None,
                        help="MEMIT-Seq lambda_prev values (default: auto-discover)")
    parser.add_argument("--lambda_delta", type=float, nargs="+", default=None,
                        help="MEMIT-Seq lambda_delta values (default: auto-discover)")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Build all (lp, ld) pairs to load — auto-discover if not specified
    if args.lambda_prev is None and args.lambda_delta is None:
        seq_variants = _discover_memit_seq_variants(args.seeds)
    else:
        lp_vals = args.lambda_prev or [1.0]
        ld_vals = args.lambda_delta or [0.0]
        seq_variants = [(lp, ld) for lp in lp_vals for ld in ld_vals]

    print("=== Method Comparison ===\n")
    print(f"Seeds: {args.seeds}")
    print(f"MEMIT-Seq variants: {[f'lp{lp}-ld{ld}' for lp, ld in seq_variants]}")
    print(f"Output: {args.output_dir}\n")

    # Load all data
    all_rows = []

    for seed in args.seeds:
        print(f"Loading seed {seed}...")

        ae_rows = load_alphaedit_failure_curve(seed)
        print(f"  AlphaEdit failure curve: {len(ae_rows)} checkpoints")
        all_rows.extend(ae_rows)

        poly2_rows = load_poly2_results(seed)
        print(f"  AlphaEdit-Poly2: {len(poly2_rows)} checkpoints")
        all_rows.extend(poly2_rows)

        for lp, ld in seq_variants:
            variant_name = f"MEMIT-Seq-lp{lp}-ld{ld}-cache0"
            seq_rows = load_memit_seq_results(seed, lp, ld)
            print(f"  {variant_name}: {len(seq_rows)} checkpoints")
            all_rows.extend(seq_rows)

    if not all_rows:
        print("\nERROR: No data found.")
        return

    df = pd.DataFrame(all_rows)
    print(f"\nTotal data points: {len(df)}")
    print(f"Methods: {df['method'].unique().tolist()}")
    print(f"Edit counts: {sorted(df['total_edits'].unique().tolist())}")

    # Save CSV
    csv_path = args.output_dir / "method_comparison_summary.csv"
    df.to_csv(csv_path, index=False)
    print(f"\nSaved: {csv_path}")

    # Print summary
    print_summary(df)

    # Generate plots
    print("\nGenerating plots...")
    plot_comparison(df, args.output_dir)
    plot_efficacy_focused(df, args.output_dir)

    print("\n=== Done ===")


if __name__ == "__main__":
    main()

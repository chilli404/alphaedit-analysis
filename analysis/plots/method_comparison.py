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


def load_alphaedit_failure_curve(seed: int) -> list[dict]:
    """Load AlphaEdit per-case results from failure_curve_checkpointed."""
    base = RESULTS / "failure_curve_checkpointed" / f"seed{seed}"
    if not base.exists():
        return []

    rows = []
    for edits_dir in sorted(base.iterdir()):
        if not edits_dir.is_dir() or not edits_dir.name.endswith("edits"):
            continue
        total_edits = int(edits_dir.name.replace("edits", ""))

        # Find AlphaEdit run_000
        run_dir = edits_dir / "AlphaEdit" / "run_000"
        if not run_dir.exists():
            continue

        case_files = list(run_dir.glob("*_edits-case_*.json"))
        if not case_files:
            continue

        metrics = _aggregate_cases(case_files)
        if metrics:
            metrics["total_edits"] = total_edits
            metrics["method"] = "AlphaEdit"
            metrics["seed"] = seed
            metrics["n_cases"] = len(case_files)
            rows.append(metrics)

    return rows


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
    """Load MEMIT-Seq pre-aggregated eval results.

    Checks two sources:
      1. full_eval_seed{N}_lp{X}_ld{Y}.json (from eval_seqreg_checkpoints.py)
      2. matched_ordering/MEMIT-Seq-lp{X}-ld{Y}-cache0/*/seed{N}/full_eval_seed{N}.json
    """
    rows = []

    # Source 1: memit_seqreg/full_eval_seed{N}_lp{X}_ld{Y}.json
    eval_path = RESULTS / "memit_seqreg" / f"full_eval_seed{seed}_lp{lambda_prev}_ld{lambda_delta}.json"
    if eval_path.exists():
        with open(eval_path) as f:
            data = json.load(f)
        for key, entry in data.items():
            if not key.endswith("_edits"):
                continue
            total_edits = entry.get("total_edits", int(key.replace("_edits", "")))
            rows.append({
                "total_edits": total_edits,
                "method": "MEMIT-Seq",
                "seed": seed,
                "efficacy": entry["all_facts"]["efficacy"],
                "paraphrase": entry["all_facts"].get("paraphrase", np.nan),
                "neighborhood": entry["all_facts"].get("neighborhood", np.nan),
                "first_1k_efficacy": entry.get("first_1k", {}).get("efficacy", np.nan),
                "latest_1k_efficacy": entry.get("latest_1k", {}).get("efficacy", np.nan),
                "n_cases": entry.get("n_evaluated", 0),
            })

    # Source 2: matched_ordering full evals (5K edits, various orderings — use key_dispersed as "hardest")
    # Skip this if we already have data at 5000 from source 1
    existing_edits = {r["total_edits"] for r in rows}
    variant = f"MEMIT-Seq-lp{lambda_prev}-ld{lambda_delta}-cache0"
    for ordering in ["key_dispersed", "key_clustered"]:
        mo_path = RESULTS / "matched_ordering" / variant / ordering / f"seed{seed}" / f"full_eval_seed{seed}.json"
        if mo_path.exists() and 5000 not in existing_edits:
            with open(mo_path) as f:
                data = json.load(f)
            # matched_ordering full_eval has a different structure — check both formats
            if isinstance(data, dict) and "all_facts" in data:
                rows.append({
                    "total_edits": 5000,
                    "method": "MEMIT-Seq",
                    "seed": seed,
                    "efficacy": data["all_facts"]["efficacy"],
                    "paraphrase": data["all_facts"].get("paraphrase", np.nan),
                    "neighborhood": data["all_facts"].get("neighborhood", np.nan),
                    "n_cases": data.get("n_evaluated", 5000),
                    "source": f"matched_ordering/{ordering}",
                })
                existing_edits.add(5000)
                break

    return rows


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

    methods = [m for m in ["AlphaEdit", "AlphaEdit-Poly2", "MEMIT-Seq"] if m in df["method"].unique()]

    for ax, metric in zip(axes, available):
        for method in methods:
            method_data = df[df["method"] == method].copy()
            if method_data.empty:
                continue

            color = COLORS[method]
            marker = MARKERS[method]

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

    fig.suptitle(
        "Method Comparison: AlphaEdit vs Poly2 vs MEMIT-Seq\n"
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

    methods = [m for m in ["AlphaEdit", "AlphaEdit-Poly2", "MEMIT-Seq"] if m in df["method"].unique()]

    for method in methods:
        method_data = df[df["method"] == method].copy()
        if method_data.empty:
            continue

        color = COLORS[method]
        marker = MARKERS[method]

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
    methods = [m for m in ["AlphaEdit", "AlphaEdit-Poly2", "MEMIT-Seq"] if m in df["method"].unique()]

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
    print(f"\n{'─' * 50}")
    print("  Δ Efficacy (Poly2 − AlphaEdit) and (MEMIT-Seq − AlphaEdit)")
    print(f"{'─' * 50}")

    ae = df[df["method"] == "AlphaEdit"].groupby("total_edits")["efficacy"].mean()
    poly2 = df[df["method"] == "AlphaEdit-Poly2"].groupby("total_edits")["efficacy"].mean()
    seq = df[df["method"] == "MEMIT-Seq"].groupby("total_edits")["efficacy"].mean()

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
    parser.add_argument("--lambda_prev", type=float, default=1.0,
                        help="MEMIT-Seq lambda_prev (default: 1.0)")
    parser.add_argument("--lambda_delta", type=float, default=0.0,
                        help="MEMIT-Seq lambda_delta (default: 0.0)")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    print("=== Method Comparison ===\n")
    print(f"Seeds: {args.seeds}")
    print(f"MEMIT-Seq params: λ_prev={args.lambda_prev}, λ_delta={args.lambda_delta}")
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

        seq_rows = load_memit_seq_results(seed, args.lambda_prev, args.lambda_delta)
        print(f"  MEMIT-Seq: {len(seq_rows)} checkpoints")
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

#!/usr/bin/env python3
"""
LaTeX table generation for AlphaEdit reproducibility paper.

Generates tables matching the format from the original AlphaEdit paper,
with additional columns for 95% CIs and delta vs MEMIT.

Usage:
    python analysis/tables.py --results_dir results
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from analysis.confidence_intervals import bootstrap_ci, wilson_interval


def generate_main_results_table(
    per_case_csv: Path, output_dir: Path
) -> str:
    """
    Generate the main results table (Table 1 equivalent).

    Format:
    | Model | Dataset | Method | Efficacy | Generalization | Specificity | Fluency | Consistency | Δ vs MEMIT |
    """
    df = pd.read_csv(per_case_csv)

    metrics = ["efficacy", "generalization", "specificity", "fluency", "consistency"]
    available_metrics = [m for m in metrics if m in df.columns]

    rows = []
    for alg in ["MEMIT", "AlphaEdit"]:
        alg_data = df[df["algorithm"] == alg]
        if alg_data.empty:
            continue

        row = {"Method": alg}
        for metric in available_metrics:
            values = alg_data[metric].dropna().values
            if len(values) == 0:
                row[metric] = "—"
                continue

            mean = np.mean(values)
            # Use Wilson for binary-like metrics, bootstrap for continuous
            if metric in ["efficacy", "generalization", "specificity"]:
                n = len(values)
                successes = int(np.sum(values >= 0.5))
                ci_low, ci_high = wilson_interval(successes, n)
            else:
                _, ci_low, ci_high = bootstrap_ci(values)

            row[metric] = f"{mean:.3f} [{ci_low:.3f}, {ci_high:.3f}]"

        rows.append(row)

    # Compute deltas
    if len(rows) == 2:
        memit_data = df[df["algorithm"] == "MEMIT"]
        alpha_data = df[df["algorithm"] == "AlphaEdit"]
        delta_row = {"Method": "Δ (AlphaEdit − MEMIT)"}
        for metric in available_metrics:
            m_vals = memit_data[metric].dropna().values
            a_vals = alpha_data[metric].dropna().values
            if len(m_vals) > 0 and len(a_vals) > 0:
                delta = np.mean(a_vals) - np.mean(m_vals)
                delta_row[metric] = f"{delta:+.3f}"
            else:
                delta_row[metric] = "—"
        rows.append(delta_row)

    result_df = pd.DataFrame(rows)

    # Generate LaTeX
    latex = result_df.to_latex(index=False, escape=False, column_format="l" + "c" * len(available_metrics))

    # Save
    output_path = output_dir / "main_results_table.tex"
    with open(output_path, "w") as f:
        f.write(latex)
    print(f"LaTeX table saved: {output_path}")

    # Also save as readable text
    text_path = output_dir / "main_results_table.txt"
    with open(text_path, "w") as f:
        f.write(result_df.to_string(index=False))
    print(f"Text table saved: {text_path}")

    return latex


def generate_reproduction_vs_original_table(
    per_case_csv: Path, output_dir: Path
) -> str:
    """
    Table comparing our reproduced values to the original paper's reported values.

    Format from the research plan:
    | Claim | Original paper value | Our reproduced value | 95% CI / std | Match type | Notes |
    """
    df = pd.read_csv(per_case_csv)

    # Original paper values for Llama-3-8B-Instruct on MCF (from Table 1)
    # These should be filled in from the actual paper once confirmed
    original_values = {
        "efficacy": {"AlphaEdit": None, "MEMIT": None},
        "generalization": {"AlphaEdit": None, "MEMIT": None},
        "specificity": {"AlphaEdit": None, "MEMIT": None},
        "fluency": {"AlphaEdit": None, "MEMIT": None},
        "consistency": {"AlphaEdit": None, "MEMIT": None},
    }

    rows = []
    for metric, orig_vals in original_values.items():
        if metric not in df.columns:
            continue
        for alg, orig_val in orig_vals.items():
            alg_data = df[df["algorithm"] == alg]
            values = alg_data[metric].dropna().values
            if len(values) == 0:
                continue

            mean = np.mean(values)
            std = np.std(values, ddof=1) if len(values) > 1 else 0

            # Determine match type
            if orig_val is not None:
                diff = abs(mean - orig_val)
                if diff < 0.02:
                    match_type = "Full"
                elif diff < 0.05:
                    match_type = "Partial"
                else:
                    match_type = "No"
            else:
                match_type = "TBD"

            rows.append({
                "Metric": f"{metric} ({alg})",
                "Original": f"{orig_val:.3f}" if orig_val else "—",
                "Reproduced": f"{mean:.3f}",
                "Std": f"±{std:.3f}",
                "Match": match_type,
            })

    result_df = pd.DataFrame(rows)

    output_path = output_dir / "reproduction_vs_original.tex"
    latex = result_df.to_latex(index=False, escape=False)
    with open(output_path, "w") as f:
        f.write(latex)
    print(f"Reproduction comparison table: {output_path}")

    return latex


def main():
    parser = argparse.ArgumentParser(description="Generate LaTeX tables for paper")
    parser.add_argument("--results_dir", type=Path, default=Path("results"))
    parser.add_argument("--output_dir", type=Path, default=Path("results/tables"))
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    per_case_csv = args.results_dir / "per_case_results.csv"
    if not per_case_csv.exists():
        print(f"ERROR: {per_case_csv} not found. Run aggregate.py first.")
        return

    print("=== Generating LaTeX Tables ===\n")
    generate_main_results_table(per_case_csv, args.output_dir)
    generate_reproduction_vs_original_table(per_case_csv, args.output_dir)
    print("\n=== Done ===")


if __name__ == "__main__":
    main()

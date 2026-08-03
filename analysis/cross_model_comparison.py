#!/usr/bin/env python3
"""
Cross-model comparison analysis: Qwen2.5-7B-Instruct and GPT-J-6B.

Generates LaTeX macros and figures for the cross-architecture replication
section of the paper. Demonstrates that age-biased forgetting and the
MEMIT-Seq advantage replicate on non-Llama architectures.

Outputs:
  - cross_model_macros.tex  — LaTeX \newcommand definitions for inline numbers
  - cross_model_failure_curve.pdf — Failure curve per model
  - cross_model_retention.pdf — Age-binned retention at final checkpoint
  - cross_model_comparison_table.csv — Method comparison table

Usage:
    uv run python -m analysis.cross_model_comparison
    uv run python -m analysis.cross_model_comparison --output-dir results/figures/cross_model
"""

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from analysis.loaders import (
    load_checkpoint_metrics,
    load_checkpoint_cohorts,
)
from analysis.style import PAPER_OUTPUT


# ─── Configuration ────────────────────────────────────────────────────────────

MODELS = {
    "qwen": {
        "label": "Qwen2.5-7B",
        "macro_prefix": "qwen",
        "experiments": {
            "AlphaEdit": "mve1_qwen_mcf",
            "MEMIT": "mve2_qwen_memit_mcf",
        },
        "failure_curve_dir": "failure_curve_qwen",
    },
    "gptj": {
        "label": "GPT-J-6B",
        "macro_prefix": "gptj",
        "experiments": {
            "AlphaEdit": "mve1_gptj_mcf",
            "MEMIT": "mve2_gptj_memit_mcf",
        },
        "failure_curve_dir": "failure_curve_gptj",
    },
}

SEEDS = [42, 2024]
METRICS = ["efficacy", "paraphrase", "neighborhood"]
EDIT_POINTS = [1000, 2000, 3000, 4000, 5000]


# ─── LaTeX Macro Generation ──────────────────────────────────────────────────


def generate_macros(results: dict, output_dir: Path):
    """
    Generate LaTeX macros for all cross-model results.

    Macro naming convention:
      \\qwenAlphaEditEfficacy, \\qwenMEMITParaphrase, etc.
      \\gptjAlphaEditEfficacy, \\gptjMEMITParaphrase, etc.
    """
    lines = [
        "% Auto-generated cross-model macros",
        "% Run: uv run python -m analysis.cross_model_comparison",
        "",
    ]

    for model_key, model_cfg in MODELS.items():
        prefix = model_cfg["macro_prefix"]
        lines.append(f"% --- {model_cfg['label']} ---")

        for method, metrics in results.get(model_key, {}).items():
            method_clean = method.replace("-", "").replace("_", "")
            for metric, value in metrics.items():
                if value is not None:
                    macro_name = f"\\{prefix}{method_clean}{metric.capitalize()}"
                    lines.append(
                        f"\\newcommand{{{macro_name}}}{{{value:.3f}}}"
                    )
        lines.append("")

    output_path = output_dir / "cross_model_macros.tex"
    output_path.write_text("\n".join(lines))
    print(f"  Wrote: {output_path}")
    return output_path


# ─── Failure Curve ────────────────────────────────────────────────────────────


def plot_failure_curve(results: dict, output_dir: Path):
    """
    Plot failure curve for each model (efficacy vs edit count).

    Stub: requires matplotlib and actual run data.
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("  SKIP: matplotlib not available for failure curve plot")
        return

    # Stub: will be populated when run artifacts exist
    print("  STUB: cross_model_failure_curve.pdf (awaiting run data)")


# ─── Age-Binned Retention ─────────────────────────────────────────────────────


def compute_age_binned_retention(model_key: str, model_cfg: dict, seed: int):
    """
    Compute retention by cohort age at the final checkpoint.

    Returns dict: {cohort_idx: {efficacy, paraphrase, neighborhood}}
    """
    # Stub: loads from failure curve checkpoints
    return None


# ─── Comparison Table ─────────────────────────────────────────────────────────


def generate_comparison_table(results: dict, output_dir: Path):
    """
    Generate CSV comparison table: Model × Method × Metrics.

    Columns: model, method, n_edits, efficacy_mean, efficacy_std,
             paraphrase_mean, paraphrase_std, neighborhood_mean, neighborhood_std
    """
    rows = []
    for model_key, model_cfg in MODELS.items():
        for method, experiment in model_cfg["experiments"].items():
            row = {
                "model": model_cfg["label"],
                "method": method,
                "experiment": experiment,
            }
            # Stub: populate from actual run data
            model_results = results.get(model_key, {}).get(method, {})
            for metric in METRICS:
                row[f"{metric}_mean"] = model_results.get(metric)
                row[f"{metric}_std"] = model_results.get(f"{metric}_std")
            rows.append(row)

    output_path = output_dir / "cross_model_comparison_table.csv"
    if rows:
        with open(output_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)
        print(f"  Wrote: {output_path}")


# ─── Main ─────────────────────────────────────────────────────────────────────


def load_cross_model_results():
    """
    Load results for all cross-model experiments.

    Returns nested dict: {model_key: {method: {metric: value}}}
    """
    results = {}

    for model_key, model_cfg in MODELS.items():
        results[model_key] = {}
        for method, experiment in model_cfg["experiments"].items():
            metrics_by_seed = []
            for seed in SEEDS:
                m = load_checkpoint_metrics(
                    seed=seed,
                    edits=2000,
                    alg=method,
                )
                if m:
                    metrics_by_seed.append(m)

            if metrics_by_seed:
                aggregated = {}
                for metric in METRICS:
                    vals = [m[metric] for m in metrics_by_seed
                            if metric in m and m[metric] is not None]
                    if vals:
                        aggregated[metric] = np.mean(vals)
                        aggregated[f"{metric}_std"] = np.std(vals)
                results[model_key][method] = aggregated
            else:
                results[model_key][method] = {m: None for m in METRICS}

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Cross-model comparison analysis"
    )
    parser.add_argument(
        "--output-dir", type=Path,
        default=PAPER_OUTPUT / "cross_model",
        help="Output directory for figures and tables"
    )
    args = parser.parse_args()

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=== Cross-Model Comparison Analysis ===")
    print(f"  Output: {output_dir}")
    print(f"  Models: {[m['label'] for m in MODELS.values()]}")
    print()

    # Load results (will be empty stubs until runs complete)
    results = load_cross_model_results()

    # Generate outputs
    generate_macros(results, output_dir)
    generate_comparison_table(results, output_dir)
    plot_failure_curve(results, output_dir)

    print("\n=== Done ===")
    print("NOTE: Macro values will be populated after GPU runs complete.")


if __name__ == "__main__":
    main()

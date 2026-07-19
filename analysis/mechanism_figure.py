#!/usr/bin/env python3
"""
Mechanism Figure — aligns per-checkpoint measurements into a unified DataFrame.

Combines data from multiple runner outputs into a single table for the
paper's mechanism figure showing how internal diagnostics evolve with
editing progress.

Columns produced:
    checkpoint, total_edits, algorithm, seed,
    retention_by_age, probability_locality,
    cache_eff_rank, cache_stable_rank, condition_number,
    projection_signal_retention, weight_drift, spectral_distortion,
    capability_perplexity, capability_mmlu

Data sources:
    - retention_by_age:              checkpoint_runner retention probes (edit_ordering.json + re-eval)
    - probability_locality:          specificity_prob from checkpoint results
    - cache_eff_rank, stable_rank:   mechanism_analyzer JSONL
    - condition_number:              plasticity_tracker JSONL
    - projection_signal_retention:   1 - projection_loss (complement of plasticity_tracker)
    - weight_drift:                  mechanism_analyzer (weight_drift_frobenius)
    - spectral_distortion:           mechanism_analyzer (SVD comparison with base)
    - capability:                    capability_probe JSONL (perplexity + MMLU)

Usage:
    python analysis/mechanism_figure.py \\
        --results_dir results/ \\
        --seed 42 \\
        --algorithm AlphaEdit \\
        --output_dir results/mechanism_figure/

    python analysis/mechanism_figure.py --help
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def load_jsonl(path: Path) -> list[dict]:
    """Load a JSONL file into a list of dicts."""
    rows = []
    if not path.exists():
        return rows
    with open(path) as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def find_mechanism_data(results_dir: Path, seed: int, algorithm: str) -> pd.DataFrame:
    """Find and load mechanism_analyzer JSONL data."""
    patterns = [
        f"mechanism_analysis/seed{seed}/**/*{algorithm}*.jsonl",
        f"mechanism/seed{seed}/*.jsonl",
        f"**/mechanism_*seed{seed}*{algorithm}*.jsonl",
    ]
    rows = []
    for pattern in patterns:
        for f in results_dir.rglob(pattern.split("/")[-1]):
            if str(seed) in str(f) and (algorithm.lower() in str(f).lower() or "mechanism" in str(f).lower()):
                rows.extend(load_jsonl(f))
                break
        if rows:
            break
    return pd.DataFrame(rows) if rows else pd.DataFrame()


def find_plasticity_data(results_dir: Path, seed: int, algorithm: str) -> pd.DataFrame:
    """Find and load plasticity_tracker JSONL data."""
    rows = []
    for f in sorted(results_dir.rglob("plasticity_*.jsonl")):
        if str(seed) in str(f):
            rows.extend(load_jsonl(f))
    return pd.DataFrame(rows) if rows else pd.DataFrame()


def find_capability_data(results_dir: Path, seed: int) -> pd.DataFrame:
    """Find and load capability_probe JSONL data."""
    rows = []
    for f in sorted(results_dir.rglob("capability_*.jsonl")):
        if str(seed) in str(f):
            rows.extend(load_jsonl(f))
    return pd.DataFrame(rows) if rows else pd.DataFrame()


def find_checkpoint_results(results_dir: Path, seed: int, algorithm: str) -> pd.DataFrame:
    """
    Load per-case results from failure_curve_checkpointed at each checkpoint.

    Groups by checkpoint and computes aggregate metrics.
    """
    from stats.aggregate import extract_metrics_from_case

    base_paths = [
        results_dir / "failure_curve_checkpointed" / f"seed{seed}",
        results_dir / f"seed{seed}",
    ]

    rows = []
    for base in base_paths:
        if not base.exists():
            continue
        for ckpt_dir in sorted(base.glob("*edits")):
            try:
                total_edits = int(ckpt_dir.name.replace("edits", ""))
            except ValueError:
                continue

            # Look for algorithm results
            alg_dir = ckpt_dir / "alphaedit_results" / algorithm / "run_000"
            if not alg_dir.exists():
                alg_dir = ckpt_dir / algorithm / "run_000"
            if not alg_dir.exists():
                continue

            case_metrics = []
            for jf in sorted(alg_dir.glob("*_edits-case_*.json")):
                try:
                    with open(jf) as f:
                        case_data = json.load(f)
                    case_metrics.append(extract_metrics_from_case(case_data))
                except (json.JSONDecodeError, KeyError):
                    continue

            if case_metrics:
                df_ckpt = pd.DataFrame(case_metrics)
                rows.append({
                    "total_edits": total_edits,
                    "checkpoint": total_edits // 100,
                    "efficacy": df_ckpt["efficacy"].mean() if "efficacy" in df_ckpt else np.nan,
                    "generalization": df_ckpt["generalization"].mean() if "generalization" in df_ckpt else np.nan,
                    "specificity": df_ckpt["specificity"].mean() if "specificity" in df_ckpt else np.nan,
                    "specificity_prob": df_ckpt["specificity_prob"].mean() if "specificity_prob" in df_ckpt else np.nan,
                    "n_cases": len(df_ckpt),
                })

    return pd.DataFrame(rows) if rows else pd.DataFrame()


def align_mechanism_data(
    results_dir: Path,
    seed: int,
    algorithm: str,
) -> pd.DataFrame:
    """
    Align all per-checkpoint measurements into a single DataFrame.

    Returns one row per checkpoint with all available mechanism metrics.
    """
    # Load individual data sources
    mechanism_df = find_mechanism_data(results_dir, seed, algorithm)
    plasticity_df = find_plasticity_data(results_dir, seed, algorithm)
    capability_df = find_capability_data(results_dir, seed)
    checkpoint_df = find_checkpoint_results(results_dir, seed, algorithm)

    # Start with checkpoint behavioral metrics as the spine
    if checkpoint_df.empty:
        print(f"WARNING: No checkpoint results found for {algorithm} seed{seed}")
        return pd.DataFrame()

    aligned = checkpoint_df.copy()
    aligned["algorithm"] = algorithm
    aligned["seed"] = seed

    # Probability locality (from specificity_prob in checkpoint results)
    if "specificity_prob" in aligned.columns:
        aligned["probability_locality"] = aligned["specificity_prob"]

    # Merge mechanism metrics (by batch/checkpoint)
    if not mechanism_df.empty:
        batch_col = "batch" if "batch" in mechanism_df.columns else "batch_idx"
        if batch_col in mechanism_df.columns:
            mech_agg = mechanism_df.groupby(batch_col).agg({
                col: "mean" for col in mechanism_df.columns
                if col != batch_col and mechanism_df[col].dtype in [np.float64, np.int64]
            }).reset_index()
            mech_agg = mech_agg.rename(columns={batch_col: "checkpoint"})

            # Map to aligned columns
            col_map = {
                "effective_rank": "cache_eff_rank",
                "stable_rank": "cache_stable_rank",
                "weight_drift_frobenius": "weight_drift",
                "spectral_distortion": "spectral_distortion",
            }
            for src, dst in col_map.items():
                if src in mech_agg.columns:
                    aligned = aligned.merge(
                        mech_agg[["checkpoint", src]].rename(columns={src: dst}),
                        on="checkpoint", how="left",
                    )

    # Merge plasticity metrics
    if not plasticity_df.empty:
        batch_col = "batch" if "batch" in plasticity_df.columns else "batch_idx"
        if batch_col in plasticity_df.columns:
            plast_agg = plasticity_df.groupby(batch_col).agg({
                col: "mean" for col in plasticity_df.columns
                if col != batch_col and plasticity_df[col].dtype in [np.float64, np.int64]
            }).reset_index()
            plast_agg = plast_agg.rename(columns={batch_col: "checkpoint"})

            if "projection_removed_fraction" in plast_agg.columns:
                aligned = aligned.merge(
                    plast_agg[["checkpoint", "projection_removed_fraction"]],
                    on="checkpoint", how="left",
                )
                aligned["projection_signal_retention"] = 1.0 - aligned["projection_removed_fraction"]

            if "condition_number" in plast_agg.columns:
                aligned = aligned.merge(
                    plast_agg[["checkpoint", "condition_number"]],
                    on="checkpoint", how="left",
                )

    # Merge capability metrics
    if not capability_df.empty:
        batch_col = "batch" if "batch" in capability_df.columns else "batch_idx"
        if batch_col in capability_df.columns:
            cap_agg = capability_df.groupby(batch_col).first().reset_index()
            cap_agg = cap_agg.rename(columns={batch_col: "checkpoint"})
            if "perplexity" in cap_agg.columns:
                aligned = aligned.merge(
                    cap_agg[["checkpoint", "perplexity"]].rename(
                        columns={"perplexity": "capability_perplexity"}
                    ),
                    on="checkpoint", how="left",
                )
            if "mmlu_accuracy" in cap_agg.columns:
                aligned = aligned.merge(
                    cap_agg[["checkpoint", "mmlu_accuracy"]].rename(
                        columns={"mmlu_accuracy": "capability_mmlu"}
                    ),
                    on="checkpoint", how="left",
                )

    return aligned


def generate_figure_data(
    results_dir: Path,
    seed: int,
    algorithm: str,
    output_dir: Path,
) -> pd.DataFrame:
    """Generate aligned mechanism figure data and save to CSV."""
    output_dir.mkdir(parents=True, exist_ok=True)

    aligned = align_mechanism_data(results_dir, seed, algorithm)

    if aligned.empty:
        print(f"No data to align for {algorithm} seed{seed}")
        return aligned

    # Save
    output_path = output_dir / f"mechanism_aligned_{algorithm}_seed{seed}.csv"
    aligned.to_csv(output_path, index=False)

    print(f"Mechanism figure data: {output_path}")
    print(f"  Checkpoints: {len(aligned)}")
    print(f"  Columns: {list(aligned.columns)}")

    # Print summary of available data
    data_cols = [c for c in aligned.columns if aligned[c].notna().any()
                 and c not in ("algorithm", "seed", "total_edits", "checkpoint", "n_cases")]
    print(f"  Available metrics: {data_cols}")

    return aligned


def main():
    parser = argparse.ArgumentParser(
        description="Align per-checkpoint mechanism measurements for paper figure"
    )
    parser.add_argument("--results_dir", type=Path, default=Path("results"),
                        help="Root results directory")
    parser.add_argument("--seed", type=int, default=42,
                        help="Seed to analyze")
    parser.add_argument("--algorithm", default="AlphaEdit",
                        choices=["AlphaEdit", "MEMIT"],
                        help="Algorithm to analyze")
    parser.add_argument("--output_dir", type=Path, default=None,
                        help="Output directory (default: results/mechanism_figure/)")
    args = parser.parse_args()

    output_dir = args.output_dir or (args.results_dir / "mechanism_figure")
    generate_figure_data(args.results_dir, args.seed, args.algorithm, output_dir)


if __name__ == "__main__":
    main()

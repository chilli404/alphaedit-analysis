#!/usr/bin/env python3
"""Scheduling Forensics — JSONL Mechanism Log Parser.

Parses per-batch JSONL mechanism logs produced by alphaedit_stream_runner.py
during the matched ordering GPU runs. Extracts ||dW||_F, removed_fraction,
cache effective rank, and q_t trajectories.

Prerequisite — pull JSONL logs from S3 (excludes large checkpoint files):
    aws s3 cp s3://grainger-mlops-pimmachinelearning-dev/continual-learning/alphaedit/checkpoints/matched_ordering/AlphaEdit/greedy_minmax/seed42/ results/matched_ordering/diagnostics/greedy_minmax/seed42/ --recursive --exclude "batch_*"
    aws s3 cp s3://grainger-mlops-pimmachinelearning-dev/continual-learning/alphaedit/checkpoints/matched_ordering/AlphaEdit/cluster_topo/seed42/ results/matched_ordering/diagnostics/cluster_topo/seed42/ --recursive --exclude "batch_*"
    aws s3 cp s3://grainger-mlops-pimmachinelearning-dev/continual-learning/alphaedit/checkpoints/matched_ordering/AlphaEdit/random/seed42/ results/matched_ordering/diagnostics/random/seed42/ --recursive --exclude "batch_*"
    aws s3 cp s3://grainger-mlops-pimmachinelearning-dev/continual-learning/alphaedit/checkpoints/matched_ordering/AlphaEdit/key_clustered/seed42/ results/matched_ordering/diagnostics/key_clustered/seed42/ --recursive --exclude "batch_*"
    aws s3 cp s3://grainger-mlops-pimmachinelearning-dev/continual-learning/alphaedit/checkpoints/matched_ordering/AlphaEdit/key_dispersed/seed42/ results/matched_ordering/diagnostics/key_dispersed/seed42/ --recursive --exclude "batch_*"

Usage:
    uv run python scheduling/forensics_mechanism.py --seed 42
    uv run python scheduling/forensics_mechanism.py --seed 42 --orderings greedy_minmax key_clustered
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RESULTS = PROJECT_ROOT / "results"
REPORTS_DIR = PROJECT_ROOT / "scheduling" / "reports"

ALL_ORDERINGS = ["key_clustered", "key_dispersed", "greedy_minmax", "cluster_topo", "random"]


def find_jsonl_files(seed: int, ordering: str) -> list[Path]:
    """Find JSONL mechanism log files for a given ordering."""
    # Check diagnostics dir (pulled from S3)
    diag_dir = RESULTS / "matched_ordering" / "diagnostics" / ordering / f"seed{seed}"
    if diag_dir.exists():
        files = sorted(diag_dir.glob("*.jsonl"))
        if files:
            return files

    # Check standard results dir
    results_dir = RESULTS / "matched_ordering" / "AlphaEdit" / ordering / f"seed{seed}"
    if results_dir.exists():
        files = sorted(results_dir.glob("*.jsonl"))
        if files:
            return files

    return []


def parse_jsonl_records(jsonl_path: Path) -> list[dict]:
    """Parse a JSONL file into list of records."""
    records = []
    with open(jsonl_path) as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    return records


def extract_mechanism_trajectories(records: list[dict]) -> dict:
    """Extract per-batch mechanism metrics from JSONL records."""
    batches = []
    update_norms = []
    removed_fractions = []
    effective_ranks = []
    q_t_values = []
    cache_conditions = []
    fit_quality_proj = []
    fit_quality_raw = []

    for record in sorted(records, key=lambda r: r.get("batch_idx", 0)):
        batch_idx = record.get("batch_idx")
        if batch_idx is None:
            continue

        mechanism = record.get("mechanism", {})
        agg = mechanism.get("aggregate", {})

        batches.append(batch_idx)
        update_norms.append(agg.get("mean_update_norm"))
        removed_fractions.append(agg.get("mean_removed_fraction"))
        effective_ranks.append(agg.get("mean_cache_effective_rank"))
        q_t_values.append(agg.get("mean_q_t"))
        cache_conditions.append(agg.get("mean_cache_condition"))
        fit_quality_proj.append(agg.get("mean_fit_quality_projected"))
        fit_quality_raw.append(agg.get("mean_fit_quality_raw"))

    return {
        "batches": batches,
        "update_norm": update_norms,
        "removed_fraction": removed_fractions,
        "effective_rank": effective_ranks,
        "q_t": q_t_values,
        "cache_condition": cache_conditions,
        "fit_quality_projected": fit_quality_proj,
        "fit_quality_raw": fit_quality_raw,
    }


def detect_spike_onset(values: list, threshold_multiplier: float = 3.0) -> int | None:
    """Detect where a metric spikes above threshold_multiplier × early median."""
    clean = [v for v in values if v is not None]
    if len(clean) < 20:
        return None

    # Early baseline: first 20 values
    early_median = float(np.median(clean[:20]))
    if early_median == 0:
        return None

    threshold = early_median * threshold_multiplier
    for i, v in enumerate(clean):
        if v > threshold:
            return i
    return None


def format_mechanism_report(all_trajectories: dict, seed: int) -> str:
    """Format mechanism trajectory report."""
    lines = [
        "## 6. Mechanism Trajectories (from JSONL logs)",
        "",
        f"Seed: {seed}",
        "",
    ]

    # Summary table
    lines.extend([
        "### ||dW|| Update Norm Trajectory",
        "| Ordering | Batches | Early (0-10) | Mid (30-50) | Late (70-99) | Spike Onset |",
        "|----------|:-------:|:------------:|:-----------:|:------------:|:-----------:|",
    ])

    for ordering, traj in sorted(all_trajectories.items()):
        norms = traj["update_norm"]
        clean_norms = [v for v in norms if v is not None]
        if not clean_norms:
            lines.append(f"| {ordering} | 0 | — | — | — | — |")
            continue

        n = len(clean_norms)
        early = clean_norms[:min(10, n)]
        mid = clean_norms[30:min(50, n)] if n > 30 else []
        late = clean_norms[70:] if n > 70 else []

        early_str = f"{np.mean(early):.4f}" if early else "—"
        mid_str = f"{np.mean(mid):.4f}" if mid else "—"
        late_str = f"{np.mean(late):.4f}" if late else "—"

        spike = detect_spike_onset(clean_norms)
        spike_str = f"batch {spike}" if spike is not None else "None"

        lines.append(f"| {ordering} | {n} | {early_str} | {mid_str} | {late_str} | {spike_str} |")

    # Removed fraction table
    lines.extend([
        "",
        "### Removed Fraction (signal lost to projection)",
        "| Ordering | Early (0-10) | Mid (30-50) | Late (70-99) | Spike Onset |",
        "|----------|:------------:|:-----------:|:------------:|:-----------:|",
    ])

    for ordering, traj in sorted(all_trajectories.items()):
        fracs = traj["removed_fraction"]
        clean = [v for v in fracs if v is not None]
        if not clean:
            lines.append(f"| {ordering} | — | — | — | — |")
            continue

        n = len(clean)
        early = clean[:min(10, n)]
        mid = clean[30:min(50, n)] if n > 30 else []
        late = clean[70:] if n > 70 else []

        early_str = f"{np.mean(early):.4f}" if early else "—"
        mid_str = f"{np.mean(mid):.4f}" if mid else "—"
        late_str = f"{np.mean(late):.4f}" if late else "—"

        spike = detect_spike_onset(clean, threshold_multiplier=2.0)
        spike_str = f"batch {spike}" if spike is not None else "None"

        lines.append(f"| {ordering} | {early_str} | {mid_str} | {late_str} | {spike_str} |")

    # q_t table
    lines.extend([
        "",
        "### q_t (Functional Signal Preservation Ratio)",
        "| Ordering | Early (0-10) | Mid (30-50) | Late (70-99) |",
        "|----------|:------------:|:-----------:|:------------:|",
    ])

    for ordering, traj in sorted(all_trajectories.items()):
        qts = traj["q_t"]
        clean = [v for v in qts if v is not None]
        if not clean:
            lines.append(f"| {ordering} | — | — | — |")
            continue

        n = len(clean)
        early = clean[:min(10, n)]
        mid = clean[30:min(50, n)] if n > 30 else []
        late = clean[70:] if n > 70 else []

        early_str = f"{np.mean(early):.4f}" if early else "—"
        mid_str = f"{np.mean(mid):.4f}" if mid else "—"
        late_str = f"{np.mean(late):.4f}" if late else "—"

        lines.append(f"| {ordering} | {early_str} | {mid_str} | {late_str} |")

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Parse JSONL mechanism logs for forensics")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--orderings", nargs="+", default=ALL_ORDERINGS)
    args = parser.parse_args()

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*70}")
    print("Mechanism Log Forensics")
    print(f"  Seed: {args.seed}")
    print(f"  Orderings: {args.orderings}")
    print(f"{'='*70}")

    all_trajectories = {}
    for ordering in args.orderings:
        jsonl_files = find_jsonl_files(args.seed, ordering)
        if not jsonl_files:
            print(f"  {ordering}: No JSONL files found")
            continue

        # Use the most recent (largest) JSONL file
        jsonl_path = max(jsonl_files, key=lambda p: p.stat().st_size)
        print(f"  {ordering}: Loading {jsonl_path.name} ({jsonl_path.stat().st_size / 1024:.0f} KB)")

        records = parse_jsonl_records(jsonl_path)
        if not records:
            print(f"    WARNING: No valid records parsed")
            continue

        traj = extract_mechanism_trajectories(records)
        all_trajectories[ordering] = traj
        n_valid = sum(1 for v in traj["update_norm"] if v is not None)
        print(f"    Parsed {len(records)} records, {n_valid} with mechanism data")

    if not all_trajectories:
        print("\n  No mechanism logs found. Pull from S3 first:")
        print("    aws s3 cp s3://grainger-mlops-pimmachinelearning-dev/continual-learning/alphaedit/"
              "checkpoints/matched_ordering/AlphaEdit/{ordering}/seed42/ "
              "results/matched_ordering/diagnostics/{ordering}/seed42/ --recursive --exclude 'batch_*'")
        sys.exit(0)

    # Generate report section
    report = format_mechanism_report(all_trajectories, args.seed)
    print(f"\n{report}")

    # Save
    report_path = REPORTS_DIR / f"mechanism_trajectories_seed{args.seed}.md"
    with open(report_path, "w") as f:
        f.write(report)
    print(f"\n  Saved: {report_path}")

    # Save raw data
    data_path = REPORTS_DIR / f"mechanism_data_seed{args.seed}.json"
    with open(data_path, "w") as f:
        json.dump(all_trajectories, f, indent=2)
    print(f"  Data: {data_path}")

    print(f"\n{'='*70}\n")


if __name__ == "__main__":
    main()

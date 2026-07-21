"""Paper tables and paper_numbers.json generator.

Produces:
  - table1_reproduction.csv (method × dataset × metrics, mean ± std across seeds)
  - table2_controlled_coupling.csv (seed × stream × metrics)
  - table3_matched_comparison.csv (method × edits × cohort metrics)
  - table4_stream_audit.csv (matched vs manipulated properties)
  - paper_numbers.json (all numbers cited in prose)

Usage:
    uv run python -m analysis.paper_tables
    uv run python -m analysis.paper_tables --output-dir results/figures/paper
"""

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from analysis.style import PAPER_OUTPUT
from analysis.loaders import (
    load_checkpoint_metrics,
    load_checkpoint_cohorts,
    load_comparison_ordered,
    load_controlled_coupling_behavioral,
    load_controlled_coupling_jsonl,
    load_mve_metrics,
    load_seqreg_eval,
    load_stream_audit,
)

# ─── Configuration ────────────────────────────────────────────────────────────

SEEDS = [42, 2024, 137, 7, 99]
EDIT_POINTS = [2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000, 10000]
BATCH_SIZE = 100


# ─── Table 1: Reproduction Results ───────────────────────────────────────────


def table1_reproduction(output_dir: Path):
    """Table 1: Multi-seed reproduction results (mean ± std).

    Uses MVE experiment results (all 5 seeds) as the authoritative source.
    Falls back to failure_curve_checkpointed at 2K if MVE data unavailable.
    """
    rows = []

    configs = [
        ("AlphaEdit", "mcf", "mve1_alphaedit_mcf", "AlphaEdit", SEEDS),
        ("MEMIT", "mcf", "mve2_memit_mcf", "MEMIT", SEEDS),
        ("AlphaEdit", "zsre", "mve3_alphaedit_zsre", "AlphaEdit", [42, 137, 2024, 7, 99]),
    ]

    for method_name, dataset, experiment, alg, seeds in configs:
        metrics_by_seed = []
        for seed in seeds:
            # Primary: MVE experiment results
            m = load_mve_metrics(experiment, seed, alg)
            # Fallback: failure curve at 2K
            if m is None:
                m = load_checkpoint_metrics(seed, 2000, alg)
            if m:
                metrics_by_seed.append(m)

        if not metrics_by_seed:
            continue

        row = {
            "method": method_name,
            "dataset": dataset,
            "n_edits": 2000,
            "n_seeds": len(metrics_by_seed),
        }

        for metric in ("efficacy", "paraphrase", "neighborhood",
                       "neighborhood_prob"):
            vals = [m[metric] for m in metrics_by_seed if metric in m and m[metric] is not None]
            if vals:
                row[f"{metric}_mean"] = np.mean(vals)
                row[f"{metric}_std"] = np.std(vals)

        rows.append(row)

    # Write CSV
    csv_path = output_dir / "table1_reproduction.csv"
    if rows:
        fieldnames = list(rows[0].keys())
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        print(f"  {csv_path.name}: {len(rows)} rows")
    return rows


# ─── Table 2: Controlled Coupling Summary ────────────────────────────────────


def table2_controlled_coupling(output_dir: Path):
    """Table 2: Controlled coupling summary across seeds."""
    rows = []

    for seed in [42, 137]:
        behav = load_controlled_coupling_behavioral(seed)
        if behav is None:
            continue

        for stream in ("low_coupling", "high_coupling"):
            data = behav.get(stream, {})
            if not data:
                continue

            row = {
                "seed": seed,
                "stream": stream,
                "overall_efficacy": data.get("overall_efficacy"),
                "first_1k_efficacy": data.get("first_1k_mean_efficacy"),
                "latest_1k_efficacy": data.get("last_1k_mean_efficacy"),
                "retention_auc": data.get("retention_auc"),
            }

            # Get mechanism metrics from JSONL (last record)
            jsonl_records = load_controlled_coupling_jsonl(stream, seed)
            if jsonl_records:
                last = jsonl_records[-1]
                agg = last.get("mechanism", {}).get("aggregate", {})
                row["effective_rank"] = agg.get("mean_cache_effective_rank")
                row["removed_fraction"] = agg.get("mean_removed_fraction")
                row["condition"] = agg.get("mean_cache_condition")

            rows.append(row)

    csv_path = output_dir / "table2_controlled_coupling.csv"
    if rows:
        fieldnames = list(rows[0].keys())
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        print(f"  {csv_path.name}: {len(rows)} rows")
    return rows


# ─── Table 3: Matched Baseline Comparison ────────────────────────────────────


def table3_matched_comparison(output_dir: Path):
    """Table 3: Decisive comparison (MEMIT vs AlphaEdit vs SeqReg at 3K/5K)."""
    rows = []
    seed = 42

    for edits in (3000, 5000):
        # AlphaEdit
        ae = load_checkpoint_metrics(seed, edits, "AlphaEdit")
        if ae:
            # Get cohort data
            cohorts = load_checkpoint_cohorts(seed, edits, "AlphaEdit", BATCH_SIZE)
            first_1k = latest_1k = None
            if cohorts:
                f1k = [cohorts[i]["efficacy"] for i in range(10) if i in cohorts]
                first_1k = np.mean(f1k) if f1k else None
                n_batches = edits // BATCH_SIZE
                l1k = [cohorts[i]["efficacy"] for i in range(n_batches - 10, n_batches) if i in cohorts]
                latest_1k = np.mean(l1k) if l1k else None

            rows.append({
                "method": "AlphaEdit",
                "edits": edits,
                "efficacy": ae.get("efficacy"),
                "paraphrase": ae.get("paraphrase"),
                "neighborhood": ae.get("neighborhood"),
                "first_1k": first_1k,
                "latest_1k": latest_1k,
            })

        # MEMIT
        memit = load_checkpoint_metrics(seed, edits, "MEMIT")
        if memit:
            rows.append({
                "method": "MEMIT",
                "edits": edits,
                "efficacy": memit.get("efficacy"),
                "paraphrase": memit.get("paraphrase"),
                "neighborhood": memit.get("neighborhood"),
                "first_1k": None,
                "latest_1k": None,
            })

        # MEMIT+SeqReg
        seqreg = load_seqreg_eval(seed)
        key = f"{edits}_edits"
        if seqreg and key in seqreg:
            sr = seqreg[key]
            rows.append({
                "method": "MEMIT+SeqReg",
                "edits": edits,
                "efficacy": sr.get("all_facts", {}).get("efficacy"),
                "paraphrase": sr.get("all_facts", {}).get("paraphrase"),
                "neighborhood": sr.get("all_facts", {}).get("neighborhood"),
                "first_1k": sr.get("first_1k", {}).get("efficacy"),
                "latest_1k": sr.get("latest_1k", {}).get("efficacy"),
            })

    csv_path = output_dir / "table3_matched_comparison.csv"
    if rows:
        fieldnames = list(rows[0].keys())
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        print(f"  {csv_path.name}: {len(rows)} rows")
    return rows


# ─── Table 4: Stream-Matching Audit ──────────────────────────────────────────


def table4_stream_audit(output_dir: Path):
    """Table 4: Stream-matching audit (matched vs manipulated properties)."""
    audit = load_stream_audit(42)
    if audit is None:
        print("  table4_stream_audit.csv: SKIP (no audit data)")
        return []

    low_key = "low_structure" if "low_structure" in audit else "low_coupling"
    high_key = "high_structure" if "high_structure" in audit else "high_coupling"
    low = audit.get(low_key, {})
    high = audit.get(high_key, {})

    properties = [
        ("n_records", "N records"),
        ("prompt_len_mean", "Prompt length (mean)"),
        ("prompt_len_std", "Prompt length (std)"),
        ("target_new_len_mean", "Target length (mean)"),
        ("n_unique_relations", "Unique relations"),
        ("relation_entropy", "Relation entropy"),
        ("n_unique_subjects", "Unique subjects"),
        ("subject_reuse_rate", "Subject reuse rate"),
        ("max_subject_repeats", "Max subject repeats"),
        ("mean_intra_batch_overlap", "Mean intra-batch overlap"),
        ("max_intra_batch_overlap", "Max intra-batch overlap"),
    ]

    rows = []
    for key, label in properties:
        rows.append({
            "property": label,
            "low_coupling": low.get(key),
            "high_coupling": high.get(key),
            "status": "MANIPULATED" if key in ("n_unique_subjects", "subject_reuse_rate",
                                                "max_subject_repeats",
                                                "mean_intra_batch_overlap",
                                                "max_intra_batch_overlap") else "MATCHED",
        })

    csv_path = output_dir / "table4_stream_audit.csv"
    if rows:
        fieldnames = list(rows[0].keys())
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        print(f"  {csv_path.name}: {len(rows)} rows")
    return rows


# ─── Paper Numbers JSON ───────────────────────────────────────────────────────


def generate_paper_numbers(output_dir: Path):
    """Generate paper_numbers.json with all cited values."""
    numbers = {}

    # Failure curve trajectory
    for seed in [42, 2024]:
        for edits in EDIT_POINTS:
            ae = load_checkpoint_metrics(seed, edits, "AlphaEdit")
            memit = load_checkpoint_metrics(seed, edits, "MEMIT")
            if ae:
                numbers[f"ae_efficacy_seed{seed}_{edits}"] = ae["efficacy"]
            if memit:
                numbers[f"memit_efficacy_seed{seed}_{edits}"] = memit["efficacy"]

    # Controlled coupling
    for seed in [42, 137]:
        behav = load_controlled_coupling_behavioral(seed)
        if behav:
            for stream in ("low_coupling", "high_coupling"):
                data = behav.get(stream, {})
                prefix = f"coupling_{stream}_seed{seed}"
                numbers[f"{prefix}_overall_eff"] = data.get("overall_efficacy")
                numbers[f"{prefix}_first_1k"] = data.get("first_1k_mean_efficacy")
                numbers[f"{prefix}_last_1k"] = data.get("last_1k_mean_efficacy")
                numbers[f"{prefix}_auc"] = data.get("retention_auc")

        # First-1K gap
        if behav:
            low_f1k = behav.get("low_coupling", {}).get("first_1k_mean_efficacy")
            high_f1k = behav.get("high_coupling", {}).get("first_1k_mean_efficacy")
            if low_f1k is not None and high_f1k is not None:
                numbers[f"coupling_first_1k_gap_seed{seed}"] = low_f1k - high_f1k

    # Order sensitivity
    for edits in [3000, 7000]:
        orders = load_comparison_ordered(42, edits)
        ae_orders = [o for o in orders if o["algorithm"] == "AlphaEdit"]
        if ae_orders:
            effs = [o["efficacy"] for o in ae_orders]
            numbers[f"order_cv_{edits}"] = np.std(effs) / np.mean(effs) * 100
            numbers[f"order_spread_{edits}"] = max(effs) - min(effs)

    # SeqReg comparison
    seqreg = load_seqreg_eval(42)
    if seqreg:
        for key, edits in [("2000_edits", 2000), ("3000_edits", 3000),
                           ("4000_edits", 4000), ("5000_edits", 5000)]:
            if key in seqreg:
                sr = seqreg[key]
                numbers[f"seqreg_efficacy_{edits}"] = sr.get("all_facts", {}).get("efficacy")
                numbers[f"seqreg_paraphrase_{edits}"] = sr.get("all_facts", {}).get("paraphrase")
                numbers[f"seqreg_neighborhood_{edits}"] = sr.get("all_facts", {}).get("neighborhood")
                numbers[f"seqreg_auc_{edits}"] = sr.get("retention_auc")
                numbers[f"seqreg_first_1k_{edits}"] = sr.get("first_1k", {}).get("efficacy")
                numbers[f"seqreg_latest_1k_{edits}"] = sr.get("latest_1k", {}).get("efficacy")
                numbers[f"seqreg_latest_100_{edits}"] = sr.get("latest_100", {}).get("efficacy")

    # Write JSON
    json_path = output_dir / "paper_numbers.json"
    with open(json_path, "w") as f:
        json.dump(numbers, f, indent=2, default=str)
    n_values = sum(1 for v in numbers.values() if v is not None)
    print(f"  paper_numbers.json: {n_values} values")
    return numbers


# ─── Main ─────────────────────────────────────────────────────────────────────


def generate(output_dir: Path = PAPER_OUTPUT):
    """Generate all tables and paper_numbers.json."""
    output_dir.mkdir(parents=True, exist_ok=True)
    print("Generating tables...")
    table1_reproduction(output_dir)
    table2_controlled_coupling(output_dir)
    table3_matched_comparison(output_dir)
    table4_stream_audit(output_dir)
    print("\nGenerating paper numbers...")
    generate_paper_numbers(output_dir)


def main():
    parser = argparse.ArgumentParser(description="Generate paper tables and numbers")
    parser.add_argument("--output-dir", type=Path, default=PAPER_OUTPUT)
    args = parser.parse_args()
    generate(args.output_dir)


if __name__ == "__main__":
    main()

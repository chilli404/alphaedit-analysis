#!/usr/bin/env python3
"""
Phase 0.2-0.3: Validate checkpoint continuity and produce standardized metrics.

1. Parses all metadata JSONL files to verify:
   - No batch gaps (every 1K boundary covered by at least one run segment)
   - Consistent parameters across continuation segments
   - Cache_c was restored (inferred from resumed_from_batch matching prior ended_at_batch)

2. Produces a standardized metrics CSV with BOTH argmax and probability-based metrics:
   - efficacy_argmax: strict (per-token argmax match)
   - efficacy_prob: P(target_new) based on NLL
   - specificity_argmax: neighborhood argmax correctness
   - specificity_prob: P(target_true) > P(target_new) for neighborhood prompts
   - generalization_argmax: paraphrase argmax
   - generalization_prob: paraphrase NLL-based

Output:
   results/validation/
     checkpoint_continuity_report.json   # Continuity validation
     standardized_metrics.csv            # Per-case metrics with both metric types

Usage:
    uv run python -m analysis.validate_and_standardize
    uv run python -m analysis.validate_and_standardize --results_dir results/failure_curve_checkpointed
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


RESULTS_BASE = Path("results/failure_curve_checkpointed")
OUTPUT_DIR = Path("results/validation")


# ─── Phase 0.3: Checkpoint Continuity Validation ─────────────────────────────

def validate_checkpoint_continuity(results_dir: Path) -> dict:
    """
    Parse all metadata JSONL files and verify batch coverage continuity.

    For each (seed, algorithm, target_edits) combination, check that:
    1. Run segments cover the full batch range [0, target_edits/100)
    2. Parameters are consistent across segments
    3. resumed_from_batch of each segment matches the checkpoint boundary
    """
    metadata_files = sorted(results_dir.rglob("metadata/*.jsonl"))

    # Parse all metadata entries
    entries = []
    for mf in metadata_files:
        with open(mf) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                entry = json.loads(line)
                entry["_source_file"] = str(mf)
                entries.append(entry)

    if not entries:
        return {"status": "ERROR", "message": "No metadata files found", "entries": 0}

    # Group by (seed, algorithm, dataset_size_limit)
    groups = {}
    for e in entries:
        key = (e["seed"], e["algorithm"], e["dataset_size_limit"])
        groups.setdefault(key, []).append(e)

    report = {
        "status": "OK",
        "total_metadata_entries": len(entries),
        "groups": {},
        "issues": [],
    }

    for (seed, alg, ds_limit), group_entries in sorted(groups.items()):
        group_key = f"seed{seed}_{alg}_{ds_limit}edits"
        total_batches = ds_limit // 100

        # Extract batch ranges from entries
        segments = []
        for e in group_entries:
            segments.append({
                "resumed_from_batch": e.get("resumed_from_batch"),
                "ended_at_batch": e.get("ended_at_batch"),
                "timestamp": e.get("timestamp_utc"),
                "hostname": e.get("hostname"),
                "run_id": e.get("run_id"),
                "checkpoint_dir": e.get("checkpoint_dir"),
                "eval_at_checkpoints_only": e.get("params", {}).get("eval_at_checkpoints_only", False),
                "save_interval": e.get("params", {}).get("save_interval", 10),
            })

        # Check parameter consistency
        param_keys = ["save_interval", "eval_at_checkpoints_only"]
        param_values = {}
        for seg in segments:
            for pk in param_keys:
                val = seg.get(pk)
                param_values.setdefault(pk, set()).add(str(val))

        param_consistent = all(len(v) == 1 for v in param_values.values())
        if not param_consistent:
            report["issues"].append(
                f"{group_key}: Inconsistent parameters across segments: {param_values}"
            )

        # Check batch coverage
        # With eval_at_checkpoints_only, evaluations happen at save_interval boundaries.
        # The metadata only captures the evaluation segment — prior editing segments
        # (with smaller dataset_size_limit) are captured in separate metadata files.
        # For this validation, we check that resumed_from_batch aligns with the
        # expected checkpoint boundary (dataset_size_limit / num_edits).
        save_interval = segments[0]["save_interval"] if segments else 10
        expected_start_batch = ds_limit // 100  # Expected resumed_from_batch

        # Verify that segments properly resume from the expected point
        # (which indicates cache_c was loaded from checkpoint)
        resumed_batches = [seg.get("resumed_from_batch") for seg in segments
                          if seg.get("resumed_from_batch") is not None]
        expected_checkpoints = list(range(save_interval - 1, total_batches, save_interval))

        # For eval_at_checkpoints_only, the run resumes FROM the target batch
        # and evaluates all facts at that point. Coverage = data exists.
        covered_checkpoints = set()
        for seg in segments:
            start = seg.get("resumed_from_batch")
            end = seg.get("ended_at_batch")
            if start is None or end is None:
                continue
            # The segment covers batches [start, end]
            for cp in expected_checkpoints:
                if start <= cp <= end:
                    covered_checkpoints.add(cp)

        # Also: if resumed_from_batch == total_batches, it means the full edit
        # sequence was loaded from checkpoint, which is the expected pattern
        cache_restored = any(
            seg.get("resumed_from_batch") == expected_start_batch
            for seg in segments
        )

        missing_checkpoints = [cp for cp in expected_checkpoints if cp not in covered_checkpoints]

        group_report = {
            "total_batches": total_batches,
            "n_segments": len(segments),
            "expected_checkpoints": expected_checkpoints,
            "covered_checkpoints": sorted(covered_checkpoints),
            "missing_checkpoints": missing_checkpoints,
            "cache_restored": cache_restored,
            "resumed_batches": resumed_batches,
            "param_consistent": param_consistent,
            "segments": segments,
        }

        if missing_checkpoints:
            report["issues"].append(
                f"{group_key}: Missing checkpoints at batches {missing_checkpoints}"
            )

        report["groups"][group_key] = group_report

    if report["issues"]:
        report["status"] = "WARNINGS"

    return report


# ─── Phase 0.2: Standardized Metric Extraction ───────────────────────────────

def extract_standardized_metrics(case_json: dict) -> dict:
    """
    Extract BOTH argmax and probability-based metrics from a case JSON.

    This resolves the metric ambiguity identified in Phase 0:
    - `correct` fields use strict per-token argmax matching
    - `probs` fields use NLL-based comparison (target_true vs target_new)
    """
    post = case_json.get("post", {})

    row = {
        "case_id": case_json["case_id"],
        "num_edits": case_json.get("num_edits"),
    }

    # Extract requested_rewrite info
    rw = case_json.get("requested_rewrite", {})
    if isinstance(rw, list):
        rw = rw[0] if rw else {}
    row["subject"] = rw.get("subject", "")
    row["target_new"] = rw.get("target_new", {}).get("str", "")
    row["target_true"] = rw.get("target_true", {}).get("str", "")

    # ─── Efficacy (rewrite prompts) ───
    rc = post.get("rewrite_prompts_correct", [])
    if rc:
        row["efficacy_argmax"] = sum(rc) / len(rc)
    else:
        row["efficacy_argmax"] = np.nan

    rp = post.get("rewrite_prompts_probs", [])
    if rp:
        # target_new has LOWER NLL = higher probability = edit successful
        row["efficacy_prob"] = sum(
            1 for p in rp if p["target_new"] < p["target_true"]
        ) / len(rp)
        row["efficacy_nll_new"] = np.mean([p["target_new"] for p in rp])
        row["efficacy_nll_true"] = np.mean([p["target_true"] for p in rp])
    else:
        row["efficacy_prob"] = np.nan
        row["efficacy_nll_new"] = np.nan
        row["efficacy_nll_true"] = np.nan

    # ─── Generalization (paraphrase prompts) ───
    pc = post.get("paraphrase_prompts_correct", [])
    if pc:
        row["generalization_argmax"] = sum(pc) / len(pc)
    else:
        row["generalization_argmax"] = np.nan

    pp = post.get("paraphrase_prompts_probs", [])
    if pp:
        row["generalization_prob"] = sum(
            1 for p in pp if p["target_new"] < p["target_true"]
        ) / len(pp)
    else:
        row["generalization_prob"] = np.nan

    # ─── Specificity (neighborhood prompts) ───
    nc = post.get("neighborhood_prompts_correct", [])
    if nc:
        row["specificity_argmax"] = sum(nc) / len(nc)
    else:
        row["specificity_argmax"] = np.nan

    np_probs = post.get("neighborhood_prompts_probs", [])
    if np_probs:
        # For specificity: target_true should have LOWER NLL (= higher prob) than target_new
        # i.e., the edit did NOT bleed into unrelated facts
        row["specificity_prob"] = sum(
            1 for p in np_probs if p["target_true"] < p["target_new"]
        ) / len(np_probs)
        row["specificity_nll_true"] = np.mean([p["target_true"] for p in np_probs])
        row["specificity_nll_new"] = np.mean([p["target_new"] for p in np_probs])
    else:
        row["specificity_prob"] = np.nan
        row["specificity_nll_true"] = np.nan
        row["specificity_nll_new"] = np.nan

    return row


def find_run_dirs(base_dir: Path) -> list[dict]:
    """
    Scan checkpoint results directory structure.

    Handles multiple extraction patterns:
    - alphaedit_results/AlphaEdit/run_000/  (combined tarball)
    - alphaedit_results/MEMIT/run_000/      (combined tarball)
    - alphaedit_results_MEMIT/MEMIT/run_000/ (separate MEMIT tarball)
    - results/AlphaEdit/run_000/            (raw results dir)
    """
    entries = []
    for seed_dir in sorted(base_dir.iterdir()):
        if not seed_dir.is_dir() or not seed_dir.name.startswith("seed"):
            continue
        seed = int(seed_dir.name.replace("seed", ""))

        for edits_dir in sorted(seed_dir.iterdir()):
            if not edits_dir.is_dir() or not edits_dir.name.endswith("edits"):
                continue
            total_edits = int(edits_dir.name.replace("edits", ""))

            # Search all subdirs that might contain algorithm results
            for subdir in edits_dir.iterdir():
                if not subdir.is_dir():
                    continue
                # Skip tarballs and non-result dirs
                if subdir.suffix in (".gz", ".tar"):
                    continue

                # Look for algorithm dirs (AlphaEdit, MEMIT) at various depths
                _scan_for_alg_dirs(subdir, seed, total_edits, entries)

    # Deduplicate: for same (seed, algorithm, total_edits), keep the run with most cases
    by_key = {}
    for e in entries:
        key = (e["seed"], e["algorithm"], e["total_edits"])
        if key not in by_key or e["n_cases"] > by_key[key]["n_cases"]:
            by_key[key] = e

    return list(by_key.values())


def _scan_for_alg_dirs(parent: Path, seed: int, total_edits: int, entries: list):
    """Recursively scan for AlphaEdit/MEMIT run directories."""
    if parent.name in ("AlphaEdit", "MEMIT"):
        # This IS an algorithm directory — look for run dirs
        for run_dir in sorted(parent.iterdir()):
            if not run_dir.is_dir() or not run_dir.name.startswith("run_"):
                continue
            case_files = list(run_dir.glob("*_edits-case_*.json"))
            if len(case_files) < 10:
                continue
            entries.append({
                "seed": seed,
                "algorithm": parent.name,
                "total_edits": total_edits,
                "run_dir": run_dir,
                "n_cases": len(case_files),
            })
        return

    # Otherwise, recurse one level (but not infinitely)
    for child in parent.iterdir():
        if child.is_dir() and child.name in ("AlphaEdit", "MEMIT", "results",
                                              "alphaedit_results", "alphaedit_results_MEMIT",
                                              "results_MEMIT"):
            _scan_for_alg_dirs(child, seed, total_edits, entries)


def build_standardized_metrics(results_dir: Path) -> pd.DataFrame:
    """Build standardized metrics DataFrame from all results."""
    entries = find_run_dirs(results_dir)
    print(f"  Found {len(entries)} run directories")

    all_rows = []
    for entry in entries:
        run_dir = entry["run_dir"]
        case_files = sorted(run_dir.glob("*_edits-case_*.json"))

        for cf in case_files:
            with open(cf) as f:
                case_data = json.load(f)
            row = extract_standardized_metrics(case_data)
            row["seed"] = entry["seed"]
            row["algorithm"] = entry["algorithm"]
            row["total_edits"] = entry["total_edits"]
            row["batch_idx"] = row["case_id"] // 100  # Derive cohort from case_id
            all_rows.append(row)

    if not all_rows:
        return pd.DataFrame()

    df = pd.DataFrame(all_rows)
    return df


# ─── Summary Statistics ───────────────────────────────────────────────────────

def compute_summary_table(df: pd.DataFrame) -> pd.DataFrame:
    """Compute summary table showing both metric types side-by-side."""
    metric_cols = [
        "efficacy_argmax", "efficacy_prob",
        "generalization_argmax", "generalization_prob",
        "specificity_argmax", "specificity_prob",
    ]

    summary = (
        df.groupby(["algorithm", "total_edits", "seed"])[metric_cols]
        .mean()
        .reset_index()
    )

    # Cross-seed mean
    cross_seed = (
        summary.groupby(["algorithm", "total_edits"])[metric_cols]
        .agg(["mean", "std"])
        .reset_index()
    )

    return summary, cross_seed


# ─── Main ────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Phase 0.2-0.3: Validate and standardize")
    parser.add_argument("--results_dir", type=Path, default=RESULTS_BASE)
    parser.add_argument("--output_dir", type=Path, default=OUTPUT_DIR)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("Phase 0.2-0.3: Checkpoint Validation + Metric Standardization")
    print("=" * 70)

    # ─── Phase 0.3: Validate continuity ───
    print("\n--- Phase 0.3: Checkpoint Continuity Validation ---\n")
    report = validate_checkpoint_continuity(args.results_dir)
    print(f"  Status: {report['status']}")
    print(f"  Total metadata entries: {report['total_metadata_entries']}")
    print(f"  Groups found: {len(report['groups'])}")

    if report["issues"]:
        print("\n  ISSUES:")
        for issue in report["issues"]:
            print(f"    - {issue}")
    else:
        print("\n  No issues found. All checkpoint boundaries covered.")

    # Print coverage summary
    print("\n  Coverage summary:")
    for group_key, gr in sorted(report["groups"].items()):
        n_covered = len(gr["covered_checkpoints"])
        n_expected = len(gr["expected_checkpoints"])
        status = "OK" if not gr["missing_checkpoints"] else "GAPS"
        print(f"    {group_key}: {n_covered}/{n_expected} checkpoints [{status}]")

    # Save report
    report_path = args.output_dir / "checkpoint_continuity_report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2, default=str)
    print(f"\n  Report saved: {report_path}")

    # ─── Phase 0.2: Standardized metrics ───
    print("\n--- Phase 0.2: Standardized Metric Extraction ---\n")
    print("  Extracting all cases with both argmax and probability metrics...")
    df = build_standardized_metrics(args.results_dir)

    if df.empty:
        print("  ERROR: No case data found.")
        return

    print(f"  Total cases extracted: {len(df)}")
    print(f"  Algorithms: {sorted(df['algorithm'].unique())}")
    print(f"  Seeds: {sorted(df['seed'].unique())}")
    print(f"  Edit counts: {sorted(df['total_edits'].unique())}")

    # Save standardized CSV
    csv_path = args.output_dir / "standardized_metrics.csv"
    df.to_csv(csv_path, index=False)
    print(f"\n  Saved: {csv_path} ({len(df)} rows)")

    # Compute and print summary
    print("\n  --- Metric Summary (cross-seed mean) ---\n")
    per_seed, cross_seed = compute_summary_table(df)

    # Print condensed summary
    for alg in sorted(df["algorithm"].unique()):
        print(f"  {alg}:")
        alg_data = per_seed[per_seed["algorithm"] == alg]
        for edits in sorted(alg_data["total_edits"].unique()):
            ed = alg_data[alg_data["total_edits"] == edits]
            eff_a = ed["efficacy_argmax"].mean()
            eff_p = ed["efficacy_prob"].mean()
            spec_a = ed["specificity_argmax"].mean()
            spec_p = ed["specificity_prob"].mean()
            print(f"    {edits:>5} edits: "
                  f"eff_argmax={eff_a:.3f} eff_prob={eff_p:.3f} | "
                  f"spec_argmax={spec_a:.3f} spec_prob={spec_p:.3f}")
        print()

    # Save per-seed summary
    summary_path = args.output_dir / "per_seed_summary.csv"
    per_seed.to_csv(summary_path, index=False)
    print(f"  Saved: {summary_path}")

    # Highlight key discrepancies
    print("\n  --- Key Discrepancies (argmax vs prob) ---\n")
    ae_data = per_seed[per_seed["algorithm"] == "AlphaEdit"]
    if not ae_data.empty:
        for edits in sorted(ae_data["total_edits"].unique()):
            ed = ae_data[ae_data["total_edits"] == edits]
            eff_gap = (ed["efficacy_prob"].mean() - ed["efficacy_argmax"].mean())
            spec_gap = (ed["specificity_prob"].mean() - ed["specificity_argmax"].mean())
            if abs(spec_gap) > 0.1:
                print(f"    AlphaEdit @ {edits}: spec_prob - spec_argmax = {spec_gap:+.3f} "
                      f"(prob={ed['specificity_prob'].mean():.3f}, "
                      f"argmax={ed['specificity_argmax'].mean():.3f})")

    print("\n" + "=" * 70)
    print("Validation complete.")
    print("=" * 70)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Aggregate per-case evaluation JSONs into cohort summaries (full_eval_seed{N}.json).

Downloads per-case files from S3, merges across run dirs (latest takes precedence),
computes per-cohort metrics, and writes the same format as eval_matched_ordering.py.

Usage:
    uv run python scripts/aggregate_percase_evals.py
    uv run python scripts/aggregate_percase_evals.py --ordering fb_high_exposure --seed 42
    uv run python scripts/aggregate_percase_evals.py --dry-run
"""

import argparse
import json
import os
import subprocess
import sys
import tempfile
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np

S3_BASE = "s3://grainger-mlops-pimmachinelearning-dev/continual-learning/alphaedit/results"
LOCAL_BASE = Path(__file__).resolve().parent.parent / "results"
BATCH_SIZE = 100


CONDITIONS = [
    ("fb_high_exposure", 42),
    ("fb_high_exposure", 2024),
    ("fb_high_exposure", 137),
    ("fb_low_exposure", 42),
    ("fb_low_exposure", 2024),
    ("fb_low_exposure", 137),
    ("key_clustered", 137),
    ("key_dispersed", 137),
]


def s3_list_case_files(ordering, seed):
    """List all per-case JSON files across all run dirs on S3."""
    prefix = f"{S3_BASE}/matched_ordering/AlphaEdit/{ordering}/seed{seed}/AlphaEdit/"
    result = subprocess.run(
        ["aws", "s3", "ls", "--recursive", prefix],
        capture_output=True, text=True, timeout=120,
    )
    if result.returncode != 0:
        print(f"  WARNING: aws s3 ls failed for {ordering}/seed{seed}")
        return {}

    # Parse: collect case_id -> (s3_path, run_dir) using latest run
    case_files = {}
    for line in result.stdout.strip().split("\n"):
        if not line or "case_" not in line:
            continue
        parts = line.split()
        if len(parts) < 4:
            continue
        s3_path = parts[-1]
        fname = s3_path.split("/")[-1]
        if not fname.startswith("100_edits-case_") or not fname.endswith(".json"):
            continue

        case_id_str = fname.replace("100_edits-case_", "").replace(".json", "")
        try:
            case_id = int(case_id_str)
        except ValueError:
            continue

        # Extract run dir number for precedence
        run_dir = "run_000"
        for part in s3_path.split("/"):
            if part.startswith("run_"):
                run_dir = part
                break

        # Keep the latest run's file
        if case_id not in case_files or run_dir > case_files[case_id][1]:
            full_s3 = f"s3://grainger-mlops-pimmachinelearning-dev/{s3_path}"
            case_files[case_id] = (full_s3, run_dir)

    return case_files


def download_case_files(case_files, tmp_dir, ordering, seed):
    """Bulk-download per-case JSONs via aws s3 sync. Returns dict: case_id -> local_path."""
    # Group files by run_dir for efficient sync
    run_dirs = defaultdict(list)
    for cid, (s3_path, run_dir) in case_files.items():
        run_dirs[run_dir].append(cid)

    # Sync each run_dir that has files we need
    s3_base_dir = f"{S3_BASE}/matched_ordering/AlphaEdit/{ordering}/seed{seed}/AlphaEdit"
    local_base = os.path.join(tmp_dir, "runs")
    os.makedirs(local_base, exist_ok=True)

    for run_dir, cids in run_dirs.items():
        s3_run = f"{s3_base_dir}/{run_dir}/"
        local_run = os.path.join(local_base, run_dir)
        os.makedirs(local_run, exist_ok=True)
        print(f"    Syncing {run_dir} ({len(cids)} cases)...")
        subprocess.run(
            ["aws", "s3", "sync", s3_run, local_run,
             "--exclude", "*", "--include", "100_edits-case_*.json",
             "--quiet"],
            capture_output=True, timeout=600,
        )

    # Build case_id -> local_path from synced files (latest run wins)
    local_paths = {}
    for run_dir in sorted(run_dirs.keys()):
        local_run = os.path.join(local_base, run_dir)
        if not os.path.isdir(local_run):
            continue
        for fname in os.listdir(local_run):
            if not fname.startswith("100_edits-case_"):
                continue
            cid_str = fname.replace("100_edits-case_", "").replace(".json", "")
            try:
                cid = int(cid_str)
            except ValueError:
                continue
            local_paths[cid] = os.path.join(local_run, fname)

    print(f"    Synced {len(local_paths)} case files")
    return local_paths


def load_ordering_stream(ordering, seed):
    """Load the ordering stream to get case_id -> position mapping."""
    local_path = LOCAL_BASE / "matched_ordering" / "orderings" / f"{ordering}_seed{seed}.json"
    if not local_path.exists():
        # Try S3
        s3_path = f"{S3_BASE}/matched_ordering/orderings/{ordering}_seed{seed}.json"
        local_path.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(
            ["aws", "s3", "cp", s3_path, str(local_path), "--quiet"],
            capture_output=True, timeout=60,
        )
    if not local_path.exists():
        return None
    with open(local_path) as f:
        stream = json.load(f)
    return {r["case_id"]: i for i, r in enumerate(stream)}


def _prob_pref_rewrite(probs_list):
    """Prob-pref for rewrite/paraphrase: success = target_new more probable (lower NLL)."""
    if not probs_list or not isinstance(probs_list[0], dict):
        return None
    return float(np.mean([x["target_true"] > x["target_new"] for x in probs_list]))


def _prob_pref_neighborhood(probs_list):
    """Prob-pref for neighborhood: success = target_true more probable (preserved)."""
    if not probs_list or not isinstance(probs_list[0], dict):
        return None
    return float(np.mean([x["target_true"] < x["target_new"] for x in probs_list]))


def parse_case_json(path):
    """Extract metrics from a per-case JSON.

    Uses probability-preference metric (official AlphaEdit metric) as primary.
    Falls back to argmax if _probs data is unavailable.
    """
    with open(path) as f:
        d = json.load(f)
    post = d.get("post", {})

    # Probability-preference (primary, official metric)
    eff = _prob_pref_rewrite(post.get("rewrite_prompts_probs", []))
    para = _prob_pref_rewrite(post.get("paraphrase_prompts_probs", []))
    neigh = _prob_pref_neighborhood(post.get("neighborhood_prompts_probs", []))

    # Fallback to argmax if _probs unavailable
    if eff is None:
        eff = float(np.mean(post.get("rewrite_prompts_correct", [False])))
    if para is None:
        para = float(np.mean(post.get("paraphrase_prompts_correct", [False])))
    if neigh is None:
        neigh = float(np.mean(post.get("neighborhood_prompts_correct", [False])))

    return {
        "case_id": d["case_id"],
        "efficacy": float(eff),
        "paraphrase": float(para),
        "neighborhood": float(neigh),
    }


def compute_cohort_summary(metrics_by_position, total_edits):
    """Compute cohort-level summary matching full_eval format."""
    n_batches = total_edits // BATCH_SIZE

    cohort_metrics = {}
    all_eff, all_para, all_neigh = [], [], []

    for batch_idx in range(n_batches):
        start_pos = batch_idx * BATCH_SIZE
        end_pos = start_pos + BATCH_SIZE
        batch_eff, batch_para, batch_neigh = [], [], []

        for pos in range(start_pos, end_pos):
            if pos in metrics_by_position:
                m = metrics_by_position[pos]
                batch_eff.append(m["efficacy"])
                batch_para.append(m["paraphrase"])
                batch_neigh.append(m["neighborhood"])
                all_eff.append(m["efficacy"])
                all_para.append(m["paraphrase"])
                all_neigh.append(m["neighborhood"])

        n_facts = len(batch_eff)
        cohort_metrics[str(batch_idx)] = {
            "edits_range": f"{start_pos}-{end_pos}",
            "efficacy": float(np.mean(batch_eff)) if batch_eff else None,
            "paraphrase": float(np.mean(batch_para)) if batch_para else None,
            "neighborhood": float(np.mean(batch_neigh)) if batch_neigh else None,
            "n_facts": n_facts,
        }

    n_evaluated = len(all_eff)

    # Cohort slices
    def _slice_mean(values, start, end):
        s = values[start:end]
        return float(np.mean(s)) if s else None

    first_1k_eff = [metrics_by_position[p]["efficacy"] for p in range(min(1000, total_edits)) if p in metrics_by_position]
    first_1k_para = [metrics_by_position[p]["paraphrase"] for p in range(min(1000, total_edits)) if p in metrics_by_position]
    first_1k_neigh = [metrics_by_position[p]["neighborhood"] for p in range(min(1000, total_edits)) if p in metrics_by_position]

    latest_1k_start = max(0, total_edits - 1000)
    latest_1k_eff = [metrics_by_position[p]["efficacy"] for p in range(latest_1k_start, total_edits) if p in metrics_by_position]
    latest_1k_para = [metrics_by_position[p]["paraphrase"] for p in range(latest_1k_start, total_edits) if p in metrics_by_position]
    latest_1k_neigh = [metrics_by_position[p]["neighborhood"] for p in range(latest_1k_start, total_edits) if p in metrics_by_position]

    mid_start, mid_end = 1000, max(1000, total_edits - 1000)
    mid_eff = [metrics_by_position[p]["efficacy"] for p in range(mid_start, mid_end) if p in metrics_by_position]
    mid_para = [metrics_by_position[p]["paraphrase"] for p in range(mid_start, mid_end) if p in metrics_by_position]
    mid_neigh = [metrics_by_position[p]["neighborhood"] for p in range(mid_start, mid_end) if p in metrics_by_position]

    latest_100_start = max(0, total_edits - 100)
    l100_eff = [metrics_by_position[p]["efficacy"] for p in range(latest_100_start, total_edits) if p in metrics_by_position]
    l100_para = [metrics_by_position[p]["paraphrase"] for p in range(latest_100_start, total_edits) if p in metrics_by_position]
    l100_neigh = [metrics_by_position[p]["neighborhood"] for p in range(latest_100_start, total_edits) if p in metrics_by_position]

    # Retention AUC: mean of per-cohort efficacy values
    cohort_effs = [v["efficacy"] for v in cohort_metrics.values() if v["efficacy"] is not None]
    retention_auc = float(np.mean(cohort_effs)) if cohort_effs else None

    def _safe_mean(vals):
        return float(np.mean(vals)) if vals else None

    return {
        "total_edits": total_edits,
        "n_evaluated": n_evaluated,
        "all_facts": {
            "efficacy": _safe_mean(all_eff),
            "paraphrase": _safe_mean(all_para),
            "neighborhood": _safe_mean(all_neigh),
        },
        "first_1k": {
            "efficacy": _safe_mean(first_1k_eff),
            "paraphrase": _safe_mean(first_1k_para),
            "neighborhood": _safe_mean(first_1k_neigh),
        },
        "middle_cohort": {
            "efficacy": _safe_mean(mid_eff),
            "paraphrase": _safe_mean(mid_para),
            "neighborhood": _safe_mean(mid_neigh),
        },
        "latest_1k": {
            "efficacy": _safe_mean(latest_1k_eff),
            "paraphrase": _safe_mean(latest_1k_para),
            "neighborhood": _safe_mean(latest_1k_neigh),
        },
        "latest_100": {
            "efficacy": _safe_mean(l100_eff),
            "paraphrase": _safe_mean(l100_para),
            "neighborhood": _safe_mean(l100_neigh),
        },
        "retention_auc": retention_auc,
        "cohort_metrics": cohort_metrics,
    }


def process_condition(ordering, seed, dry_run=False):
    """Process one (ordering, seed) condition end-to-end."""
    print(f"\n{'='*60}")
    print(f"Processing: {ordering} / seed {seed}")

    # Load ordering to map case_id -> position
    cid_to_pos = load_ordering_stream(ordering, seed)
    if cid_to_pos is None:
        print(f"  ERROR: Could not load ordering stream")
        return False
    total_stream = len(cid_to_pos)
    print(f"  Stream: {total_stream} records")

    # List case files on S3
    print(f"  Listing S3 case files...")
    case_files = s3_list_case_files(ordering, seed)
    print(f"  Found {len(case_files)} unique case files (across all runs)")

    if dry_run:
        print(f"  [DRY RUN] Would download and aggregate {len(case_files)} files")
        return True

    # Download
    with tempfile.TemporaryDirectory() as tmp_dir:
        print(f"  Downloading to {tmp_dir}...")
        local_paths = download_case_files(case_files, tmp_dir, ordering, seed)
        print(f"  Downloaded {len(local_paths)} files")

        # Parse all case files
        print(f"  Parsing metrics...")
        metrics_by_position = {}
        missing_pos = 0
        for case_id, path in local_paths.items():
            try:
                m = parse_case_json(path)
                pos = cid_to_pos.get(case_id)
                if pos is not None:
                    metrics_by_position[pos] = m
                else:
                    missing_pos += 1
            except Exception as e:
                print(f"    WARNING: Failed to parse case {case_id}: {e}")

        if missing_pos > 0:
            print(f"  WARNING: {missing_pos} cases not in ordering stream (orphaned)")
        print(f"  Parsed {len(metrics_by_position)} cases with positions")

    # Build full_eval at checkpoint boundaries
    # Checkpoints: batch_9 (1K), batch_19 (2K), ... batch_99 (10K) — or batch_49 (5K)
    n_batches = total_stream // BATCH_SIZE
    checkpoint_batches = list(range(9, n_batches, 10))  # 9, 19, 29, ..., 99
    max_evaluated_pos = max(metrics_by_position.keys()) if metrics_by_position else 0

    full_eval = {}
    for ckpt_batch in checkpoint_batches:
        total_edits = (ckpt_batch + 1) * BATCH_SIZE
        if total_edits > max_evaluated_pos + BATCH_SIZE:
            # No data beyond this point
            continue

        summary = compute_cohort_summary(metrics_by_position, total_edits)
        summary["checkpoint"] = f"batch_{ckpt_batch}"

        # Only include if we have reasonable coverage
        coverage = summary["n_evaluated"] / total_edits if total_edits > 0 else 0
        if coverage < 0.5:
            print(f"  Skipping {total_edits}_edits: only {summary['n_evaluated']}/{total_edits} evaluated ({coverage:.0%})")
            continue

        full_eval[f"{total_edits}_edits"] = summary
        print(f"  {total_edits}_edits: {summary['n_evaluated']}/{total_edits} evaluated, "
              f"eff={summary['all_facts']['efficacy']:.3f}, "
              f"first_1k_eff={summary['first_1k']['efficacy']:.3f}")

    if not full_eval:
        print(f"  ERROR: No checkpoints with sufficient coverage")
        return False

    # Save
    out_dir = LOCAL_BASE / "matched_ordering" / "AlphaEdit" / ordering / f"seed{seed}"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"full_eval_seed{seed}.json"
    with open(out_path, "w") as f:
        json.dump(full_eval, f, indent=2)
    print(f"  Saved: {out_path}")

    # Also upload to S3
    s3_dest = f"{S3_BASE}/matched_ordering/AlphaEdit/{ordering}/seed{seed}/full_eval_seed{seed}.json"
    subprocess.run(
        ["aws", "s3", "cp", str(out_path), s3_dest, "--quiet"],
        capture_output=True, timeout=30,
    )
    print(f"  Uploaded to S3")

    return True


def main():
    parser = argparse.ArgumentParser(description="Aggregate per-case evals into full_eval")
    parser.add_argument("--ordering", type=str, default=None, help="Single ordering to process")
    parser.add_argument("--seed", type=int, default=None, help="Single seed to process")
    parser.add_argument("--dry-run", action="store_true", help="List files without downloading")
    args = parser.parse_args()

    if args.ordering and args.seed:
        conditions = [(args.ordering, args.seed)]
    else:
        conditions = CONDITIONS

    results = {}
    for ordering, seed in conditions:
        ok = process_condition(ordering, seed, dry_run=args.dry_run)
        results[(ordering, seed)] = ok

    print(f"\n{'='*60}")
    print("Summary:")
    for (ordering, seed), ok in results.items():
        status = "OK" if ok else "FAILED"
        print(f"  {ordering}/seed{seed}: {status}")


if __name__ == "__main__":
    main()

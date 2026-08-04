#!/usr/bin/env python3
"""Scheduling Result Forensics — CPU-only analysis.

Produces a comprehensive forensics report to distinguish global model collapse
from orderly interference in the scheduling experiment results.

Forensic items covered:
  2. Per-batch conditioning profile (geometry-only, from key vectors)
  3. Retention before collapse (from existing full_eval JSONs)
  4. Per-edit survival model (pre-collapse window)
  5. Fill the exposure table (all orderings)
  6. Installation quality (from full_eval latest_100)

Items NOT covered (require GPU or S3 data):
  1. Capability trajectories (WikiText/MMLU) — needs GPU
  2b. ||dW||_F per batch — needs JSONL mechanism logs from S3

Usage:
    uv run python scheduling/forensics.py --seeds 42
    uv run python scheduling/forensics.py --seeds 42 2024
"""

import argparse
import json
import sys
import warnings
from pathlib import Path

import numpy as np
from scipy import stats as scipy_stats

# Suppress residual RuntimeWarnings from numpy linalg internals
warnings.filterwarnings("ignore", category=RuntimeWarning, message=".*overflow.*")
warnings.filterwarnings("ignore", category=RuntimeWarning, message=".*invalid value.*")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "analysis"))

from analysis.loaders import (
    load_matched_ordering_full_eval,
    load_matched_ordering_keys,
    load_matched_ordering_stream,
)
from analysis.matched_ordering_key_geometry import (
    batch_effective_rank,
    prefix_cache_spectrum,
    within_batch_cosine,
)

REPORTS_DIR = PROJECT_ROOT / "scheduling" / "reports"
LATEX_DIR = PROJECT_ROOT / "scheduling" / "latex"

ALL_ORDERINGS = ["key_clustered", "key_dispersed", "greedy_minmax", "cluster_topo", "random"]
ALL_ALGS = ["AlphaEdit", "MEMIT-Seq-lp1.0-ld0.0-cache0"]
CHECKPOINTS = [1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000, 10000]
BATCH_SIZE = 100


# ─── Numeric Safety ──────────────────────────────────────────────────────────


def safe_load_keys(seed: int) -> dict | None:
    """Load key vectors with float64 casting and zero-norm filtering.

    Returns dict with:
      - keys: (N, D) float64 array with zero-norm rows zeroed out
      - case_ids: list of case IDs (same length as keys)
      - zero_norm_mask: boolean array (True where original norm was < 1e-10)
    """
    key_data = load_matched_ordering_keys(seed)
    if key_data is None:
        # Fallback to seed 42 keys (model-intrinsic, same across seeds)
        key_data = load_matched_ordering_keys(42)
    if key_data is None:
        return None

    keys = key_data["keys"].astype(np.float64)
    case_ids = key_data["case_ids"]

    # Identify zero-norm keys (extraction failures stored as zero vectors)
    norms = np.linalg.norm(keys, axis=1)
    zero_mask = norms < 1e-10
    n_zero = int(zero_mask.sum())
    if n_zero > 0:
        print(f"    WARNING: {n_zero}/{len(keys)} keys have near-zero norm (excluded from geometry)")

    return {"keys": keys, "case_ids": case_ids, "zero_norm_mask": zero_mask}


def assert_finite(arr: np.ndarray, context: str = ""):
    """Assert that array contains no NaN or Inf values."""
    if not np.all(np.isfinite(arr)):
        n_nan = int(np.isnan(arr).sum())
        n_inf = int(np.isinf(arr).sum())
        raise ValueError(f"Non-finite values in {context}: {n_nan} NaN, {n_inf} Inf")


# ─── Item 3: Collapse Timeline ───────────────────────────────────────────────


def extract_retention_trajectories(seeds: list[int], algs: list[str] = None) -> dict:
    """Extract first_1k retention and latest_100 efficacy at each checkpoint."""
    if algs is None:
        algs = ALL_ALGS
    trajectories = {}

    for seed in seeds:
        for alg in algs:
            for ordering in ALL_ORDERINGS:
                data = load_matched_ordering_full_eval(seed, ordering, alg)
                if data is None:
                    continue

                # Use (alg, ordering, seed) as key for multi-alg support
                key = (alg, ordering, seed)
                trajectories[key] = {
                    "first_1k_retention": {},
                    "latest_1k_efficacy": {},
                    "latest_100_efficacy": {},
                    "all_facts_efficacy": {},
                }

                for edits in CHECKPOINTS:
                    ckpt = data.get(f"{edits}_edits")
                    if ckpt is None:
                        continue
                    trajectories[key]["first_1k_retention"][edits] = ckpt.get("first_1k", {}).get("efficacy")
                    trajectories[key]["latest_1k_efficacy"][edits] = ckpt.get("latest_1k", {}).get("efficacy")
                    trajectories[key]["latest_100_efficacy"][edits] = ckpt.get("latest_100", {}).get("efficacy")
                    trajectories[key]["all_facts_efficacy"][edits] = ckpt.get("all_facts", {}).get("efficacy")

    return trajectories


def detect_collapse_onset(trajectory: dict, threshold: float = 0.7) -> int | None:
    """Find the first checkpoint where first_1k retention drops below threshold."""
    retention = trajectory.get("first_1k_retention", {})
    for edits in sorted(retention.keys()):
        val = retention[edits]
        if val is not None and val < threshold:
            return edits
    return None


# ─── Item 5: Geometric Exposure ──────────────────────────────────────────────


def compute_exposure_table(seeds: list[int]) -> dict:
    """Compute geometric exposure metrics for ALL orderings."""
    results = {}

    for seed in seeds:
        key_data = safe_load_keys(seed)
        if key_data is None:
            print(f"  WARNING: No key vectors found for seed {seed}")
            continue

        keys = key_data["keys"]  # float64
        case_ids = key_data["case_ids"]
        zero_mask = key_data["zero_norm_mask"]

        # L2 normalize (float64 throughout)
        norms = np.linalg.norm(keys, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-8)
        keys_normed = keys / norms

        # Build index excluding zero-norm keys from geometry
        case_id_to_idx = {cid: i for i, cid in enumerate(case_ids)
                         if not zero_mask[i]}

        for ordering_name in ALL_ORDERINGS:
            stream = load_matched_ordering_stream(seed, ordering_name)
            if stream is None:
                continue

            # Extract case_id sequence
            ordering_cids = [r["case_id"] for r in stream[:10000]]

            # Filter to keys that exist in our key set
            valid_positions = [i for i, cid in enumerate(ordering_cids) if cid in case_id_to_idx]
            valid_cids = [ordering_cids[i] for i in valid_positions]

            # First 1K exposure
            first_1k_n = min(1000, len(valid_cids))
            first_1k_cids = valid_cids[:first_1k_n]
            subsequent_cids = valid_cids[first_1k_n:]

            if not first_1k_cids or not subsequent_cids:
                continue

            first_1k_keys = keys_normed[[case_id_to_idx[c] for c in first_1k_cids]]
            subsequent_keys = keys_normed[[case_id_to_idx[c] for c in subsequent_cids]]

            # Compute max cosine from each first-1K key to all subsequent keys
            # Do in chunks to manage memory (1K × 9K = 9M entries)
            chunk_size = 500
            max_cos_values = np.zeros(first_1k_n)

            for i in range(0, first_1k_n, chunk_size):
                end = min(i + chunk_size, first_1k_n)
                cos_chunk = first_1k_keys[i:end] @ subsequent_keys.T  # (chunk, n_subsequent)
                assert_finite(cos_chunk, f"exposure cos_chunk [{ordering_name}/s{seed}]")
                max_cos_values[i:end] = cos_chunk.max(axis=1)

            # Compute within-batch mean cosine (first 10 batches)
            n_batches_sample = min(10, len(stream) // BATCH_SIZE)
            within_cos_values = []
            for b in range(n_batches_sample):
                batch_cids = [stream[b * BATCH_SIZE + j]["case_id"]
                              for j in range(BATCH_SIZE)
                              if stream[b * BATCH_SIZE + j]["case_id"] in case_id_to_idx]
                if len(batch_cids) < 2:
                    continue
                batch_keys = keys_normed[[case_id_to_idx[c] for c in batch_cids]]
                cos_mat = batch_keys @ batch_keys.T
                assert_finite(cos_mat, f"within-batch cos [{ordering_name}/s{seed} batch {b}]")
                n = len(batch_cids)
                mask = np.triu(np.ones((n, n), dtype=bool), k=1)
                within_cos_values.append(float(cos_mat[mask].mean()))

            results[(ordering_name, seed)] = {
                "frac_above_0.3": float(np.mean(max_cos_values > 0.3)),
                "frac_above_0.4": float(np.mean(max_cos_values > 0.4)),
                "mean_max_cos": float(np.mean(max_cos_values)),
                "median_max_cos": float(np.median(max_cos_values)),
                "max_max_cos": float(np.max(max_cos_values)),
                "within_batch_cos_mean": float(np.mean(within_cos_values)) if within_cos_values else None,
            }
            print(f"    {ordering_name}/s{seed}: frac>0.3={results[(ordering_name, seed)]['frac_above_0.3']:.4f}")

    return results


# ─── Item 2 (partial): Per-Batch Conditioning Profile ────────────────────────


def compute_batch_conditioning(seeds: list[int]) -> dict:
    """Compute per-batch K@K^T condition number and within-batch cosine.

    Reports both cosine-normalized κ (unit-norm keys) and unnormalized Gram κ
    (raw keys), since key-norm disparities affect the actual solve.
    """
    results = {}

    for seed in seeds:
        key_data = safe_load_keys(seed)
        if key_data is None:
            continue

        keys = key_data["keys"]  # float64
        case_ids = key_data["case_ids"]
        zero_mask = key_data["zero_norm_mask"]

        # Index excludes zero-norm keys
        case_id_to_idx = {cid: i for i, cid in enumerate(case_ids)
                         if not zero_mask[i]}

        for ordering_name in ALL_ORDERINGS:
            stream = load_matched_ordering_stream(seed, ordering_name)
            if stream is None:
                continue

            stream_trimmed = stream[:10000]
            n_batches = len(stream_trimmed) // BATCH_SIZE

            # Within-batch cosine per batch (passes float64 keys to geometry func)
            wb_cos = within_batch_cosine(keys, stream_trimmed, BATCH_SIZE, case_id_to_idx)

            # Batch effective rank
            b_rank = batch_effective_rank(keys, stream_trimmed, BATCH_SIZE, case_id_to_idx)

            # Per-batch K@K^T condition numbers (both normalized and unnormalized)
            batch_conditions = []
            batch_conditions_unnorm = []
            for b in range(n_batches):
                batch_records = stream_trimmed[b * BATCH_SIZE: (b + 1) * BATCH_SIZE]
                indices = [case_id_to_idx[r["case_id"]] for r in batch_records
                           if r["case_id"] in case_id_to_idx]
                if len(indices) < 2:
                    batch_conditions.append(float("inf"))
                    batch_conditions_unnorm.append(float("inf"))
                    continue

                batch_keys = keys[indices]  # float64

                # === Cosine-normalized condition (unit-norm Gram) ===
                bn = np.linalg.norm(batch_keys, axis=1, keepdims=True)
                bn = np.maximum(bn, 1e-8)
                batch_keys_n = batch_keys / bn

                gram_norm = batch_keys_n @ batch_keys_n.T
                assert_finite(gram_norm, f"gram_norm [{ordering_name}/s{seed} batch {b}]")
                eigvals = np.linalg.eigvalsh(gram_norm)
                eigvals_pos = eigvals[eigvals > 1e-10]
                if len(eigvals_pos) >= 2:
                    cond = float(eigvals_pos[-1] / eigvals_pos[0])
                else:
                    cond = float("inf")
                batch_conditions.append(cond)

                # === Unnormalized condition (raw Gram — reflects actual solve) ===
                gram_raw = batch_keys @ batch_keys.T
                assert_finite(gram_raw, f"gram_raw [{ordering_name}/s{seed} batch {b}]")
                eigvals_raw = np.linalg.eigvalsh(gram_raw)
                eigvals_raw_pos = eigvals_raw[eigvals_raw > 1e-10]
                if len(eigvals_raw_pos) >= 2:
                    cond_raw = float(eigvals_raw_pos[-1] / eigvals_raw_pos[0])
                else:
                    cond_raw = float("inf")
                batch_conditions_unnorm.append(cond_raw)

            # Prefix cache spectrum at key checkpoints (float64 keys propagate)
            prefix_checkpoints = [1000, 2000, 3000, 4000, 5000, 7000, 10000]
            prefix_spec = prefix_cache_spectrum(keys, stream_trimmed, prefix_checkpoints, case_id_to_idx)

            # Summarize by regions
            early = slice(0, 10)
            mid = slice(40, 50)
            late = slice(90, min(100, n_batches))

            results[(ordering_name, seed)] = {
                "within_batch_cos": {
                    "early_mean": float(np.mean(wb_cos[early])),
                    "mid_mean": float(np.mean(wb_cos[mid])),
                    "late_mean": float(np.mean(wb_cos[late])),
                    "all": wb_cos,
                },
                "batch_condition": {
                    "early_mean": float(np.mean(batch_conditions[early])),
                    "mid_mean": float(np.mean(batch_conditions[mid])),
                    "late_mean": float(np.mean(batch_conditions[late])),
                    "all": batch_conditions,
                },
                "batch_condition_unnorm": {
                    "early_mean": float(np.mean(batch_conditions_unnorm[early])),
                    "mid_mean": float(np.mean(batch_conditions_unnorm[mid])),
                    "late_mean": float(np.mean(batch_conditions_unnorm[late])),
                    "all": batch_conditions_unnorm,
                },
                "effective_rank": {
                    "early_mean": float(np.mean(b_rank[early])),
                    "mid_mean": float(np.mean(b_rank[mid])),
                    "late_mean": float(np.mean(b_rank[late])),
                    "all": b_rank,
                },
                "prefix_cache_spectrum": prefix_spec,
            }
            print(f"    {ordering_name}/s{seed}: κ_cos early={results[(ordering_name, seed)]['batch_condition']['early_mean']:.1f}"
                  f" mid={results[(ordering_name, seed)]['batch_condition']['mid_mean']:.1f}"
                  f" late={results[(ordering_name, seed)]['batch_condition']['late_mean']:.1f}"
                  f"  |  κ_raw early={results[(ordering_name, seed)]['batch_condition_unnorm']['early_mean']:.1f}"
                  f" mid={results[(ordering_name, seed)]['batch_condition_unnorm']['mid_mean']:.1f}"
                  f" late={results[(ordering_name, seed)]['batch_condition_unnorm']['late_mean']:.1f}")

    return results


# ─── Item 4: Pre-Collapse Survival Model ─────────────────────────────────────


def fit_survival_model(seeds: list[int], exposure_data: dict, trajectories: dict) -> dict:
    """Fit exposure→retention within pre-collapse window for greedy_minmax.

    Uses cohort_metrics from full_eval to correlate position-based geometric
    exposure with actual retention at the first healthy checkpoint where per-cohort
    data is available.
    """
    results = {}

    for seed in seeds:
        key_data = safe_load_keys(seed)
        if key_data is None:
            continue

        keys = key_data["keys"]  # float64
        case_ids = key_data["case_ids"]
        zero_mask = key_data["zero_norm_mask"]

        case_id_to_idx = {cid: i for i, cid in enumerate(case_ids)
                         if not zero_mask[i]}

        # L2 normalize (float64)
        norms = np.linalg.norm(keys, axis=1, keepdims=True)
        keys_normed = keys / np.maximum(norms, 1e-8)

        for ordering_name in ["greedy_minmax", "key_clustered"]:
            stream = load_matched_ordering_stream(seed, ordering_name)
            if stream is None:
                continue

            data = load_matched_ordering_full_eval(seed, ordering_name, "AlphaEdit")
            if data is None:
                continue

            # Find pre-collapse checkpoint (first_1k > 0.7)
            # Use 4000_edits (batch 39) as the test point for greedy
            test_edits = 4000 if ordering_name == "greedy_minmax" else 5000
            ckpt = data.get(f"{test_edits}_edits")
            if ckpt is None or "cohort_metrics" not in ckpt:
                continue

            cohort_metrics = ckpt["cohort_metrics"]

            # Compute per-cohort mean max-cosine to subsequent keys
            stream_cids = [r["case_id"] for r in stream[:test_edits]]
            n_cohorts = test_edits // BATCH_SIZE

            cohort_exposures = []
            cohort_retentions = []

            for c in range(n_cohorts):
                # Cohort exposure: mean max-cos to ALL subsequent keys
                start = c * BATCH_SIZE
                end = (c + 1) * BATCH_SIZE
                cohort_cids = [cid for cid in stream_cids[start:end] if cid in case_id_to_idx]
                subseq_cids = [cid for cid in stream_cids[end:] if cid in case_id_to_idx]

                if not cohort_cids or not subseq_cids:
                    continue

                cohort_keys = keys_normed[[case_id_to_idx[c] for c in cohort_cids]]
                subseq_keys = keys_normed[[case_id_to_idx[c] for c in subseq_cids]]

                cos_mat = cohort_keys @ subseq_keys.T
                assert_finite(cos_mat, f"survival cos [{ordering_name}/s{seed} cohort {c}]")
                mean_max_cos = float(cos_mat.max(axis=1).mean())
                cohort_exposures.append(mean_max_cos)

                # Cohort retention from cohort_metrics
                cm = cohort_metrics.get(str(c))
                if cm:
                    cohort_retentions.append(cm["efficacy"])
                else:
                    cohort_retentions.append(np.nan)

            # Filter out NaN
            valid = [(e, r) for e, r in zip(cohort_exposures, cohort_retentions)
                     if not np.isnan(r)]
            if len(valid) < 5:
                continue

            exposures_arr = np.array([v[0] for v in valid])
            retentions_arr = np.array([v[1] for v in valid])

            # Pearson correlation
            r, p = scipy_stats.pearsonr(exposures_arr, retentions_arr)

            # Also correlation with position (age)
            positions = np.arange(len(valid))
            r_pos, p_pos = scipy_stats.pearsonr(positions, retentions_arr)

            results[(ordering_name, seed)] = {
                "test_edits": test_edits,
                "n_cohorts": len(valid),
                "exposure_retention_r": float(r),
                "exposure_retention_p": float(p),
                "position_retention_r": float(r_pos),
                "position_retention_p": float(p_pos),
                "mean_exposure": float(np.mean(exposures_arr)),
                "std_exposure": float(np.std(exposures_arr)),
                "mean_retention": float(np.mean(retentions_arr)),
                "std_retention": float(np.std(retentions_arr)),
            }
            print(f"    {ordering_name}/s{seed} @{test_edits}: "
                  f"exposure→retention r={r:.3f} (p={p:.4f}), "
                  f"position→retention r={r_pos:.3f} (p={p_pos:.4f})")

    return results


# ─── Report Generation ────────────────────────────────────────────────────────


def format_collapse_table(trajectories: dict, seeds: list[int]) -> str:
    """Format the collapse timeline table."""
    lines = [
        "## 1. Collapse Timeline (first_1k retention)",
        "",
        "| Algorithm | Ordering | Seed | " + " | ".join(f"{e//1000}K" for e in CHECKPOINTS) + " | Collapse Onset |",
        "|-----------|----------|------|" + "|".join([":---:"] * len(CHECKPOINTS)) + "|:---:|",
    ]

    for (alg, ordering_name, seed), traj in sorted(trajectories.items()):
        if seed not in seeds:
            continue
        onset = detect_collapse_onset(traj)
        onset_str = f"**{onset//1000}K**" if onset else "None"

        # Short alg name for display
        alg_short = "AE" if alg == "AlphaEdit" else "M-Seq"

        vals = []
        for edits in CHECKPOINTS:
            v = traj["first_1k_retention"].get(edits)
            if v is not None:
                s = f"{v:.3f}"
                if v < 0.3:
                    s = f"**{s}**"
                vals.append(s)
            else:
                vals.append("—")

        lines.append(f"| {alg_short} | {ordering_name} | {seed} | {' | '.join(vals)} | {onset_str} |")

    return "\n".join(lines)


def format_installation_table(trajectories: dict, seeds: list[int]) -> str:
    """Format installation quality table (latest_100 efficacy at each checkpoint)."""
    lines = [
        "## 2. Installation Quality (latest_100 efficacy)",
        "",
        "| Algorithm | Ordering | Seed | " + " | ".join(f"{e//1000}K" for e in CHECKPOINTS) + " |",
        "|-----------|----------|------|" + "|".join([":---:"] * len(CHECKPOINTS)) + "|",
    ]

    for (alg, ordering_name, seed), traj in sorted(trajectories.items()):
        if seed not in seeds:
            continue
        alg_short = "AE" if alg == "AlphaEdit" else "M-Seq"

        vals = []
        for edits in CHECKPOINTS:
            v = traj["latest_100_efficacy"].get(edits)
            if v is not None:
                s = f"{v:.3f}"
                if v < 0.8:
                    s = f"**{s}**"
                vals.append(s)
            else:
                vals.append("—")

        lines.append(f"| {alg_short} | {ordering_name} | {seed} | {' | '.join(vals)} |")

    return "\n".join(lines)


def format_exposure_table(exposure_data: dict, seeds: list[int]) -> str:
    """Format the complete geometric exposure table."""
    lines = [
        "## 3. Geometric Exposure (Complete)",
        "",
        "| Ordering | Seed | frac>0.3 | frac>0.4 | mean_max_cos | within_batch_cos |",
        "|----------|------|:--------:|:--------:|:------------:|:----------------:|",
    ]

    for ordering_name in ALL_ORDERINGS:
        for seed in seeds:
            key = (ordering_name, seed)
            if key not in exposure_data:
                continue

            d = exposure_data[key]
            wb = d.get("within_batch_cos_mean")
            wb_str = f"{wb:.4f}" if wb is not None else "—"

            lines.append(
                f"| {ordering_name} | {seed} | "
                f"{d['frac_above_0.3']:.4f} | "
                f"{d['frac_above_0.4']:.4f} | "
                f"{d['mean_max_cos']:.4f} | "
                f"{wb_str} |"
            )

    return "\n".join(lines)


def format_conditioning_table(conditioning: dict, seeds: list[int]) -> str:
    """Format the per-batch conditioning profile table."""
    lines = [
        "## 4. Per-Batch Conditioning Profile",
        "",
        "### 4a. Within-Batch Cosine",
        "| Ordering | Seed | Early (0-10) | Mid (40-50) | Late (90-99) |",
        "|----------|------|:------------:|:-----------:|:------------:|",
    ]

    for ordering_name in ALL_ORDERINGS:
        for seed in seeds:
            key = (ordering_name, seed)
            if key not in conditioning:
                continue
            d = conditioning[key]["within_batch_cos"]
            lines.append(f"| {ordering_name} | {seed} | {d['early_mean']:.4f} | {d['mid_mean']:.4f} | {d['late_mean']:.4f} |")

    lines.extend([
        "",
        "### 4b. Batch K@K^T Condition Number (cosine-normalized)",
        "| Ordering | Seed | Early (0-10) | Mid (40-50) | Late (90-99) |",
        "|----------|------|:------------:|:-----------:|:------------:|",
    ])

    for ordering_name in ALL_ORDERINGS:
        for seed in seeds:
            key = (ordering_name, seed)
            if key not in conditioning:
                continue
            d = conditioning[key]["batch_condition"]
            lines.append(f"| {ordering_name} | {seed} | {d['early_mean']:.1f} | {d['mid_mean']:.1f} | {d['late_mean']:.1f} |")

    lines.extend([
        "",
        "### 4b'. Batch K@K^T Condition Number (unnormalized — reflects actual solve)",
        "| Ordering | Seed | Early (0-10) | Mid (40-50) | Late (90-99) |",
        "|----------|------|:------------:|:-----------:|:------------:|",
    ])

    for ordering_name in ALL_ORDERINGS:
        for seed in seeds:
            key = (ordering_name, seed)
            if key not in conditioning:
                continue
            d = conditioning[key].get("batch_condition_unnorm")
            if d is None:
                continue
            lines.append(f"| {ordering_name} | {seed} | {d['early_mean']:.1f} | {d['mid_mean']:.1f} | {d['late_mean']:.1f} |")

    lines.extend([
        "",
        "### 4c. Batch Effective Rank",
        "| Ordering | Seed | Early (0-10) | Mid (40-50) | Late (90-99) |",
        "|----------|------|:------------:|:-----------:|:------------:|",
    ])

    for ordering_name in ALL_ORDERINGS:
        for seed in seeds:
            key = (ordering_name, seed)
            if key not in conditioning:
                continue
            d = conditioning[key]["effective_rank"]
            lines.append(f"| {ordering_name} | {seed} | {d['early_mean']:.1f} | {d['mid_mean']:.1f} | {d['late_mean']:.1f} |")

    lines.extend([
        "",
        "### 4d. Prefix Cache Spectrum (Cumulative Condition Number)",
        "| Ordering | Seed | @1K | @2K | @3K | @4K | @5K | @7K | @10K |",
        "|----------|------|:---:|:---:|:---:|:---:|:---:|:---:|:----:|",
    ])

    for ordering_name in ALL_ORDERINGS:
        for seed in seeds:
            key = (ordering_name, seed)
            if key not in conditioning:
                continue
            spec = conditioning[key]["prefix_cache_spectrum"]
            vals = []
            for t in [1000, 2000, 3000, 4000, 5000, 7000, 10000]:
                if t in spec:
                    c = spec[t].get("condition", "inf")
                    vals.append(f"{c}" if isinstance(c, str) else f"{c:.0f}")
                else:
                    vals.append("—")
            lines.append(f"| {ordering_name} | {seed} | {' | '.join(vals)} |")

    return "\n".join(lines)


def format_survival_model(survival: dict) -> str:
    """Format the pre-collapse survival model results."""
    lines = [
        "## 5. Pre-Collapse Survival Model",
        "",
        "Correlation between per-cohort geometric exposure (mean max-cos to subsequent keys)",
        "and cohort retention, measured at the last healthy checkpoint.",
        "",
        "| Ordering | Seed | Test Edits | N cohorts | Exposure→Retention r | p-value | Position→Retention r | p-value |",
        "|----------|------|:----------:|:---------:|:--------------------:|:-------:|:--------------------:|:-------:|",
    ]

    for (ordering_name, seed), data in sorted(survival.items()):
        sig = "**" if data["exposure_retention_p"] < 0.05 else ""
        lines.append(
            f"| {ordering_name} | {seed} | {data['test_edits']} | {data['n_cohorts']} | "
            f"{sig}{data['exposure_retention_r']:.3f}{sig} | {data['exposure_retention_p']:.4f} | "
            f"{data['position_retention_r']:.3f} | {data['position_retention_p']:.4f} |"
        )

    lines.extend([
        "",
        "Interpretation:",
        "- Negative exposure→retention r: higher geometric exposure predicts more forgetting (hypothesis supported)",
        "- Negative position→retention r: older cohorts degrade more (expected, age effect)",
        "- If exposure→retention is significant AFTER controlling for position: geometry matters beyond age",
    ])

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Scheduling Result Forensics (CPU-only)")
    parser.add_argument("--seeds", type=int, nargs="+", default=[42],
                        help="Seeds to analyze (default: 42)")
    parser.add_argument("--algs", nargs="+", default=ALL_ALGS,
                        help="Algorithms to include (default: AlphaEdit + MEMIT-Seq)")
    args = parser.parse_args()

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    LATEX_DIR.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*70}")
    print("Scheduling Result Forensics")
    print(f"  Seeds: {args.seeds}")
    print(f"  Algs:  {args.algs}")
    print(f"{'='*70}")

    # ─── Item 3: Retention Trajectories ───────────────────────────────────
    print("\n─── Item 3: Extracting retention trajectories ───")
    trajectories = extract_retention_trajectories(args.seeds, args.algs)
    print(f"  Loaded {len(trajectories)} alg×ordering×seed combinations")

    # ─── Item 5: Geometric Exposure ───────────────────────────────────────
    print("\n─── Item 5: Computing geometric exposure for all orderings ───")
    exposure_data = compute_exposure_table(args.seeds)

    # ─── Item 2: Per-Batch Conditioning ───────────────────────────────────
    print("\n─── Item 2: Computing per-batch conditioning profiles ───")
    conditioning = compute_batch_conditioning(args.seeds)

    # ─── Item 6: Installation Quality (from trajectories) ─────────────────
    print("\n─── Item 6: Installation quality extracted from trajectories")

    # ─── Item 4: Pre-Collapse Survival Model ──────────────────────────────
    print("\n─── Item 4: Fitting pre-collapse survival model ───")
    survival = fit_survival_model(args.seeds, exposure_data, trajectories)

    # ─── Assemble Report ──────────────────────────────────────────────────
    print(f"\n{'─'*70}")
    print("Assembling forensics report...")

    collapse_table = format_collapse_table(trajectories, args.seeds)
    install_table = format_installation_table(trajectories, args.seeds)
    exposure_table = format_exposure_table(exposure_data, args.seeds)
    conditioning_table = format_conditioning_table(conditioning, args.seeds)
    survival_section = format_survival_model(survival)

    report = "\n\n".join([
        "# Scheduling Forensics Report",
        f"Seeds: {args.seeds}",
        collapse_table,
        install_table,
        exposure_table,
        conditioning_table,
        survival_section,
        "## 6. Mechanism Trajectories\n\nRequires JSONL logs from S3. Run `scheduling/forensics_mechanism.py` after pulling logs.",
        "## 7. Capability Probes\n\nRequires GPU. Run `scripts/run_capability_probe_ordering.sh` on cluster.",
    ])

    # Print key tables to console
    print(f"\n{collapse_table}")
    print(f"\n{install_table}")
    print(f"\n{exposure_table}")

    # Save
    report_path = REPORTS_DIR / "forensics_report.md"
    with open(report_path, "w") as f:
        f.write(report)
    print(f"\n  Report: {report_path}")

    # Save machine-readable data
    data_path = REPORTS_DIR / "forensics_data.json"
    serializable = {
        "trajectories": {f"{k[0]}_{k[1]}_seed{k[2]}": v for k, v in trajectories.items()},
        "exposure": {f"{k[0]}_seed{k[1]}": v for k, v in exposure_data.items()},
        "conditioning": {
            f"{k[0]}_seed{k[1]}": {
                "within_batch_cos": {kk: vv for kk, vv in v["within_batch_cos"].items() if kk != "all"},
                "batch_condition": {kk: vv for kk, vv in v["batch_condition"].items() if kk != "all"},
                "batch_condition_unnorm": {kk: vv for kk, vv in v["batch_condition_unnorm"].items() if kk != "all"},
                "effective_rank": {kk: vv for kk, vv in v["effective_rank"].items() if kk != "all"},
                "prefix_cache_spectrum": {str(kk): vv for kk, vv in v["prefix_cache_spectrum"].items()},
            }
            for k, v in conditioning.items()
        },
        "survival_model": {f"{k[0]}_seed{k[1]}": v for k, v in survival.items()},
    }
    with open(data_path, "w") as f:
        json.dump(serializable, f, indent=2)
    print(f"  Data: {data_path}")

    # Emit LaTeX macros
    macros = ["% Auto-generated by scheduling/forensics.py", ""]
    for (alg, ordering_name, seed), traj in trajectories.items():
        alg_prefix = "AE" if alg == "AlphaEdit" else "MSeq"
        ord_prefix = ordering_name.replace("_", "").title().replace(" ", "")
        prefix = f"{alg_prefix}{ord_prefix}"
        onset = detect_collapse_onset(traj)
        if onset:
            macros.append(f"\\newcommand{{\\forensics{prefix}CollapseOnset}}{{{onset//1000}K}}")
        for edits in [3000, 5000, 7000, 10000]:
            v = traj["first_1k_retention"].get(edits)
            if v is not None:
                macros.append(f"\\newcommand{{\\forensics{prefix}Ret{edits//1000}K}}{{{v*100:.1f}\\%}}")

    macros_path = LATEX_DIR / "forensics_macros.tex"
    with open(macros_path, "w") as f:
        f.write("\n".join(macros))
    print(f"  LaTeX: {macros_path}")

    print(f"\n{'='*70}")
    print("Done. Next steps:")
    print("  1. Pull JSONL mechanism logs: see scheduling/forensics_mechanism.py")
    print("  2. Run capability probes: bash scripts/run_capability_probe_ordering.sh 42 AlphaEdit greedy_minmax")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()

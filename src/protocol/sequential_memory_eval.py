"""Sequential-Memory Evaluation Protocol — core metric computation.

Computes 8 metrics per method from existing checkpoint data:

1. current_batch_efficacy  — Can the model recall what it just learned?
2. latest_1k_efficacy      — Short-term memory (most recent 1000 edits)
3. first_1k_retention      — Long-term memory (oldest 1000 edits still held?)
4. age_binned_retention    — Retention curve by edit age (oldest → newest)
5. retention_auc           — Scalar summary of the retention curve
6. order_variance          — Coefficient of variation across orderings
7. concentration_sensitivity — Retention gap between dispersed vs concentrated edits
8. cost                    — Runtime and memory per batch

Usage:
    from src.protocol.sequential_memory_eval import evaluate_method

    report = evaluate_method("AlphaEdit", seed=42, edits=5000)
    print(report.current_batch_efficacy)
    print(report.retention_auc)
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

# ─── Path Configuration ──────────────────────────────────────────────────────

PROJECT = Path(__file__).resolve().parent.parent.parent
RESULTS = PROJECT / "results"

# ─── Data Structures ─────────────────────────────────────────────────────────


@dataclass
class RetentionCurve:
    """Age-binned retention: maps age (in edits) to efficacy."""

    ages: List[int]  # ascending age (oldest first)
    efficacies: List[float]

    def as_dict(self) -> Dict[int, float]:
        return dict(zip(self.ages, self.efficacies))


@dataclass
class MethodReport:
    """Complete protocol report for one method at one (seed, edit_count) point."""

    method: str
    seed: int
    edits: int
    batch_size: int

    # Core metrics
    current_batch_efficacy: Optional[float] = None
    latest_1k_efficacy: Optional[float] = None
    first_1k_retention: Optional[float] = None
    retention_curve: Optional[RetentionCurve] = None
    retention_auc: Optional[float] = None
    order_variance: Optional[float] = None
    concentration_sensitivity: Optional[float] = None
    cost_seconds_per_batch: Optional[float] = None
    cost_memory_mb: Optional[float] = None

    # Supporting data
    n_facts_evaluated: Optional[int] = None
    n_cohorts: Optional[int] = None
    order_variance_n_orderings: Optional[int] = None
    cohort_efficacies: Optional[Dict[int, float]] = None

    def summary_dict(self) -> Dict[str, object]:
        """Flat dict of protocol metrics for table generation."""
        return {
            "method": self.method,
            "seed": self.seed,
            "edits": self.edits,
            "current_batch_efficacy": self.current_batch_efficacy,
            "latest_1k_efficacy": self.latest_1k_efficacy,
            "first_1k_retention": self.first_1k_retention,
            "retention_auc": self.retention_auc,
            "order_variance": self.order_variance,
            "concentration_sensitivity": self.concentration_sensitivity,
            "cost_seconds_per_batch": self.cost_seconds_per_batch,
            "cost_memory_mb": self.cost_memory_mb,
        }


# ─── Low-Level Data Loading ──────────────────────────────────────────────────


def _extract_efficacy(case_json: dict) -> Optional[float]:
    """Extract binary efficacy from a case JSON."""
    post = case_json.get("post", {})
    vals = post.get("rewrite_prompts_correct")
    if isinstance(vals, list) and vals:
        return sum(vals) / len(vals)
    return None


def _load_cases_raw(seed: int, edits: int, alg: str) -> List[dict]:
    """Load raw per-case JSON files for a checkpoint."""
    run_dir = (
        RESULTS / "failure_curve_checkpointed" / f"seed{seed}"
        / f"{edits}edits" / alg / "run_000"
    )
    if not run_dir.exists():
        return []
    cases = []
    for f_path in run_dir.glob("*_edits-case_*.json"):
        with open(f_path) as f:
            cases.append(json.load(f))
    return cases


def _load_cohort_efficacies(
    seed: int, edits: int, alg: str, batch_size: int
) -> Optional[Dict[int, float]]:
    """Load per-cohort mean efficacy at a checkpoint.

    Returns dict mapping cohort_index → mean efficacy.
    Cohort index = case_id // batch_size (insertion batch number).
    """
    cases = _load_cases_raw(seed, edits, alg)
    if not cases:
        return None

    from collections import defaultdict
    cohorts = defaultdict(list)
    for case in cases:
        case_id = case.get("case_id")
        if case_id is None:
            continue
        eff = _extract_efficacy(case)
        if eff is not None:
            cohorts[case_id // batch_size].append(eff)

    if not cohorts:
        return None
    return {idx: float(np.mean(vals)) for idx, vals in cohorts.items()}


# ─── Metric Computations ─────────────────────────────────────────────────────


def compute_current_batch_efficacy(
    cohort_efficacies: Dict[int, float], edits: int, batch_size: int
) -> Optional[float]:
    """Efficacy of the most recently edited cohort."""
    latest_cohort = (edits // batch_size) - 1
    return cohort_efficacies.get(latest_cohort)


def compute_latest_1k_efficacy(
    cohort_efficacies: Dict[int, float], edits: int, batch_size: int
) -> Optional[float]:
    """Mean efficacy of the 10 most recent cohorts (1000 facts)."""
    n_cohorts = edits // batch_size
    start = max(0, n_cohorts - 10)
    vals = [cohort_efficacies[i] for i in range(start, n_cohorts)
            if i in cohort_efficacies]
    return float(np.mean(vals)) if vals else None


def compute_first_1k_retention(
    cohort_efficacies: Dict[int, float],
) -> Optional[float]:
    """Mean efficacy of the first 10 cohorts (facts 0-999)."""
    vals = [cohort_efficacies[i] for i in range(10) if i in cohort_efficacies]
    return float(np.mean(vals)) if vals else None


def compute_age_binned_retention(
    cohort_efficacies: Dict[int, float],
    edits: int,
    batch_size: int,
    n_bins: int = 5,
) -> Optional[RetentionCurve]:
    """Retention curve binned by edit age (oldest → newest).

    Returns ages in units of edits (how long ago the cohort was inserted).
    """
    n_cohorts = edits // batch_size
    if n_cohorts < n_bins:
        return None

    bin_size = n_cohorts // n_bins
    ages = []
    efficacies = []

    for b in range(n_bins):
        cohort_start = b * bin_size
        cohort_end = (b + 1) * bin_size if b < n_bins - 1 else n_cohorts
        bin_vals = [cohort_efficacies[i] for i in range(cohort_start, cohort_end)
                    if i in cohort_efficacies]
        if not bin_vals:
            continue
        # Age = edits since this bin's median cohort was inserted
        median_cohort = (cohort_start + cohort_end) // 2
        age = edits - (median_cohort * batch_size)
        ages.append(age)
        efficacies.append(float(np.mean(bin_vals)))

    if not ages:
        return None
    return RetentionCurve(ages=ages, efficacies=efficacies)


def compute_retention_auc(
    cohort_efficacies: Dict[int, float], edits: int, batch_size: int
) -> Optional[float]:
    """Area under cohort retention curve, normalized to [0, 1].

    Computes trapezoidal AUC over all cohort efficacies ordered oldest→newest.
    Perfect retention (all cohorts at 1.0) yields AUC = 1.0.
    """
    n_cohorts = edits // batch_size
    # Collect efficacies in insertion order (oldest first)
    vals = [cohort_efficacies.get(i) for i in range(n_cohorts)]
    vals = [v for v in vals if v is not None]
    if len(vals) < 2:
        return None
    # Normalized AUC: trapz over [0, 1] x-axis
    trapezoid = getattr(np, "trapezoid", np.trapz)
    return float(trapezoid(vals, dx=1.0 / (len(vals) - 1)))


def compute_order_variance(
    seed: int, edits: int, alg: str
) -> Tuple[Optional[float], int]:
    """Coefficient of variation of efficacy across dataset orderings.

    Returns (cv_percent, n_orderings).
    """
    base = RESULTS / "comparison_ordered" / f"seed{seed}" / f"{edits}edits"
    if not base.exists():
        return None, 0

    # Collect per-ordering aggregate efficacy
    order_efficacies = []

    # Detect directory layout
    if (base / "order0").exists():
        order_dirs = [(str(i), base / f"order{i}") for i in range(10)
                      if (base / f"order{i}").exists()]
    else:
        order_dirs = [("0", base)]
        order_dirs += [(str(i), base / f"order{i}") for i in range(1, 10)
                       if (base / f"order{i}").exists()]

    for order_id, d in order_dirs:
        run_dir = d / alg / "run_000"
        if not run_dir.exists():
            continue
        case_files = list(run_dir.glob("*_edits-case_*.json"))
        if not case_files:
            continue

        effs = []
        for f_path in case_files:
            with open(f_path) as f:
                data = json.load(f)
            eff = _extract_efficacy(data)
            if eff is not None:
                effs.append(eff)
        if effs:
            order_efficacies.append(float(np.mean(effs)))

    if len(order_efficacies) < 2:
        return None, len(order_efficacies)

    mean_eff = np.mean(order_efficacies)
    if mean_eff == 0:
        return None, len(order_efficacies)
    cv = float(np.std(order_efficacies) / mean_eff * 100)
    return cv, len(order_efficacies)


def compute_concentration_sensitivity(seed: int) -> Optional[float]:
    """Retention AUC gap between low-coupling and high-coupling streams.

    Positive value = low coupling retains better (expected).
    """
    path = RESULTS / "controlled_coupling" / f"behavioral_eval_seed{seed}.json"
    if not path.exists():
        return None
    with open(path) as f:
        data = json.load(f)

    low_auc = data.get("low_coupling", {}).get("retention_auc")
    high_auc = data.get("high_coupling", {}).get("retention_auc")
    if low_auc is None or high_auc is None:
        return None
    return float(low_auc - high_auc)


def compute_cost(seed: int, alg: str) -> Tuple[Optional[float], Optional[float]]:
    """Estimate per-batch runtime (seconds) and memory cost (MB).

    Runtime: from controlled coupling JSONL exec_time_s (AlphaEdit only),
             or from SeqReg logs.
    Memory: from checkpoint file sizes on disk.
    """
    runtime = None
    memory = None

    if alg == "AlphaEdit":
        # Runtime from controlled coupling logs
        jsonl_path = RESULTS / "controlled_coupling"
        for jsonl in sorted(jsonl_path.glob(f"low_coupling_seed{seed}*.jsonl")):
            times = []
            with open(jsonl) as f:
                for line in f:
                    line = line.strip()
                    if line:
                        record = json.loads(line)
                        t = record.get("exec_time_s")
                        if t is not None:
                            times.append(t)
            if times:
                runtime = float(np.mean(times))
                break

        # Memory: cache_c stores one covariance matrix per edited layer.
        # Llama-3-8B with AlphaEdit edits layers [4,5,6,7,8] (5 layers).
        # Rewrite module is mlp.down_proj (shape 4096×14336), so keys are
        # 14336-dim. cache_c shape = (14336, 14336) per layer, float32.
        # Storage: 5 layers × 14336 × 14336 × 4 bytes ≈ 4.1 GB total
        memory = 5 * 14336 * 14336 * 4 / 1e6  # ~4110 MB

    elif alg in ("MEMIT+SeqReg", "MEMIT_SeqReg"):
        # Runtime from SeqReg logs (approximate from case file timestamps)
        # Memory: per-cache key storage
        metadata_path = RESULTS / "memit_seqreg" / f"metadata_seed{seed}_lp1.0_ld1.0.json"
        if metadata_path.exists():
            with open(metadata_path) as f:
                meta = json.load(f)
            cache_max = meta.get("cache_max", 20)
            # Each cached batch: 100 keys × 14336 dims × 8 bytes × 5 layers
            memory = float(cache_max * 100 * 14336 * 8 * 5 / 1e6)

    elif alg == "MEMIT":
        # MEMIT carries no state between batches
        memory = 0.0

    return runtime, memory


# ─── Cross-Checkpoint Retention Trajectories ─────────────────────────────────


def compute_cohort_trajectory(
    seed: int, alg: str, cohort_idx: int, batch_size: int = 100,
    edit_points: Optional[List[int]] = None,
) -> Dict[int, float]:
    """Track a single cohort's efficacy across multiple checkpoints.

    Returns dict mapping edit_count → cohort efficacy at that checkpoint.
    Only includes checkpoints where the cohort has already been inserted.
    """
    if edit_points is None:
        edit_points = [2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000, 10000]

    insertion_edits = (cohort_idx + 1) * batch_size
    trajectory = {}

    for edits in edit_points:
        if edits < insertion_edits:
            continue
        cohorts = _load_cohort_efficacies(seed, edits, alg, batch_size)
        if cohorts and cohort_idx in cohorts:
            trajectory[edits] = cohorts[cohort_idx]

    return trajectory


def compute_cross_checkpoint_auc(
    seed: int, alg: str, batch_size: int = 100,
    edit_points: Optional[List[int]] = None,
) -> Optional[float]:
    """Retention AUC computed across multiple checkpoints.

    For each checkpoint, computes retention AUC over all cohorts that exist
    at that point. Returns the mean AUC across checkpoints.
    """
    if edit_points is None:
        edit_points = [2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000, 10000]

    aucs = []
    for edits in edit_points:
        cohorts = _load_cohort_efficacies(seed, edits, alg, batch_size)
        if cohorts is None:
            continue
        auc = compute_retention_auc(cohorts, edits, batch_size)
        if auc is not None:
            aucs.append(auc)

    return float(np.mean(aucs)) if aucs else None


# ─── Main Entry Point ─────────────────────────────────────────────────────────


def evaluate_method(
    alg: str,
    seed: int,
    edits: int,
    batch_size: int = 100,
    order_edits: Optional[int] = None,
) -> MethodReport:
    """Evaluate a method under the sequential-memory protocol.

    Args:
        alg: Algorithm name ("AlphaEdit", "MEMIT", "MEMIT+SeqReg")
        seed: Random seed
        edits: Total edit count to evaluate at
        batch_size: Edits per batch (default 100)
        order_edits: Edit count for order sensitivity data (defaults to
                     closest available: 3000 or 7000)

    Returns:
        MethodReport with all 8 protocol metrics filled where data exists.
    """
    report = MethodReport(
        method=alg,
        seed=seed,
        edits=edits,
        batch_size=batch_size,
    )

    # --- Load cohort data (foundation for metrics 1-5) ---
    cohort_effs = _load_cohort_efficacies(seed, edits, alg, batch_size)

    if cohort_effs is None and alg in ("MEMIT+SeqReg", "MEMIT_SeqReg"):
        # Try SeqReg pre-computed eval
        cohort_effs = _load_seqreg_cohorts(seed, edits)

    if cohort_effs is not None:
        report.cohort_efficacies = cohort_effs
        report.n_cohorts = len(cohort_effs)
        report.n_facts_evaluated = sum(1 for _ in cohort_effs.values())

        # Metric 1: Current-batch efficacy
        report.current_batch_efficacy = compute_current_batch_efficacy(
            cohort_effs, edits, batch_size
        )

        # Metric 2: Latest-1K efficacy
        report.latest_1k_efficacy = compute_latest_1k_efficacy(
            cohort_effs, edits, batch_size
        )

        # Metric 3: First-1K retention
        report.first_1k_retention = compute_first_1k_retention(cohort_effs)

        # Metric 4: Age-binned retention curve
        report.retention_curve = compute_age_binned_retention(
            cohort_effs, edits, batch_size
        )

        # Metric 5: Retention AUC
        report.retention_auc = compute_retention_auc(
            cohort_effs, edits, batch_size
        )

    # --- Metric 6: Order variance ---
    if order_edits is None:
        # Use closest available scale
        for candidate in (edits, 7000, 3000):
            ov, n = compute_order_variance(seed, candidate, alg)
            if ov is not None:
                report.order_variance = ov
                report.order_variance_n_orderings = n
                break
    else:
        ov, n = compute_order_variance(seed, order_edits, alg)
        report.order_variance = ov
        report.order_variance_n_orderings = n

    # --- Metric 7: Concentration sensitivity ---
    # Only meaningful for AlphaEdit (coupling experiments used AlphaEdit)
    if alg == "AlphaEdit":
        report.concentration_sensitivity = compute_concentration_sensitivity(seed)

    # --- Metric 8: Cost ---
    runtime, memory = compute_cost(seed, alg)
    report.cost_seconds_per_batch = runtime
    report.cost_memory_mb = memory

    return report


def _load_seqreg_cohorts(seed: int, edits: int) -> Optional[Dict[int, float]]:
    """Load cohort efficacies from SeqReg full_eval JSON."""
    for ld in (1.0, 0.0):
        path = RESULTS / "memit_seqreg" / f"full_eval_seed{seed}_lp1.0_ld{ld}.json"
        if not path.exists():
            continue
        with open(path) as f:
            data = json.load(f)
        key = f"{edits}_edits"
        if key not in data:
            continue
        checkpoint = data[key]
        cohort_metrics = checkpoint.get("cohort_metrics", {})
        if cohort_metrics:
            return {
                int(k): v["efficacy"]
                for k, v in cohort_metrics.items()
                if "efficacy" in v
            }
    return None


# ─── Multi-Seed Aggregation ──────────────────────────────────────────────────


def evaluate_method_multiseed(
    alg: str,
    seeds: List[int],
    edits: int,
    batch_size: int = 100,
) -> Dict[str, object]:
    """Evaluate a method across multiple seeds, reporting mean ± std.

    Returns dict with metric_name → {"mean": float, "std": float, "n": int}.
    """
    reports = [evaluate_method(alg, s, edits, batch_size) for s in seeds]
    reports = [r for r in reports if r.cohort_efficacies is not None]

    if not reports:
        return {"method": alg, "edits": edits, "n_seeds": 0}

    metrics = [
        "current_batch_efficacy", "latest_1k_efficacy", "first_1k_retention",
        "retention_auc", "order_variance", "concentration_sensitivity",
        "cost_seconds_per_batch", "cost_memory_mb",
    ]

    result = {"method": alg, "edits": edits, "n_seeds": len(reports)}
    for metric in metrics:
        vals = [getattr(r, metric) for r in reports if getattr(r, metric) is not None]
        if vals:
            result[metric] = {"mean": float(np.mean(vals)), "std": float(np.std(vals)), "n": len(vals)}
        else:
            result[metric] = None

    return result


# ─── CLI ─────────────────────────────────────────────────────────────────────


def main():
    """Run protocol evaluation and print results."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Sequential-Memory Evaluation Protocol"
    )
    parser.add_argument("--alg", type=str, default="AlphaEdit",
                        help="Algorithm name")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--edits", type=int, default=5000)
    parser.add_argument("--batch-size", type=int, default=100)
    parser.add_argument("--all-methods", action="store_true",
                        help="Evaluate all available methods")
    parser.add_argument("--seeds", type=int, nargs="+", default=None,
                        help="Multiple seeds for aggregation")
    args = parser.parse_args()

    if args.all_methods:
        methods = ["AlphaEdit", "MEMIT", "MEMIT+SeqReg"]
        for alg in methods:
            print(f"\n{'='*60}")
            print(f"  {alg} | seed={args.seed} | edits={args.edits}")
            print(f"{'='*60}")
            report = evaluate_method(alg, args.seed, args.edits, args.batch_size)
            _print_report(report)
    elif args.seeds:
        result = evaluate_method_multiseed(
            args.alg, args.seeds, args.edits, args.batch_size
        )
        _print_multiseed(result)
    else:
        report = evaluate_method(args.alg, args.seed, args.edits, args.batch_size)
        _print_report(report)


def _print_report(report: MethodReport):
    """Pretty-print a single method report."""
    print(f"\n  Method: {report.method}")
    print(f"  Seed: {report.seed} | Edits: {report.edits} | Batch: {report.batch_size}")
    print(f"  Facts evaluated: {report.n_facts_evaluated}")
    print(f"  Cohorts: {report.n_cohorts}")
    print()
    print(f"  1. Current-batch efficacy:    {_fmt(report.current_batch_efficacy)}")
    print(f"  2. Latest-1K efficacy:        {_fmt(report.latest_1k_efficacy)}")
    print(f"  3. First-1K retention:        {_fmt(report.first_1k_retention)}")
    print(f"  4. Retention AUC:             {_fmt(report.retention_auc)}")
    print(f"  5. Order variance (CV%):      {_fmt(report.order_variance)}")
    print(f"     (from {report.order_variance_n_orderings or 0} orderings)")
    print(f"  6. Concentration sensitivity: {_fmt(report.concentration_sensitivity)}")
    print(f"  7. Cost (s/batch):            {_fmt(report.cost_seconds_per_batch)}")
    print(f"     Cost (MB state):           {_fmt(report.cost_memory_mb)}")

    if report.retention_curve:
        print(f"\n  Age-binned retention curve:")
        for age, eff in zip(report.retention_curve.ages, report.retention_curve.efficacies):
            bar = "#" * int(eff * 40)
            print(f"    age {age:5d} edits: {eff:.4f}  {bar}")


def _print_multiseed(result: Dict):
    """Pretty-print multi-seed aggregation."""
    print(f"\n  Method: {result['method']} | Edits: {result['edits']}")
    print(f"  Seeds: {result['n_seeds']}")
    print()
    for key in ["current_batch_efficacy", "latest_1k_efficacy", "first_1k_retention",
                "retention_auc", "order_variance", "concentration_sensitivity",
                "cost_seconds_per_batch", "cost_memory_mb"]:
        val = result.get(key)
        if val and isinstance(val, dict):
            print(f"  {key:30s}: {val['mean']:.4f} +/- {val['std']:.4f} (n={val['n']})")
        else:
            print(f"  {key:30s}: —")


def _fmt(val: Optional[float]) -> str:
    if val is None:
        return "—"
    return f"{val:.4f}"


if __name__ == "__main__":
    main()

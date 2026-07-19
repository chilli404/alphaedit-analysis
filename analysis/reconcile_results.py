#!/usr/bin/env python3
"""
Reconcile failure-curve results across seeds and checkpoints.

Downloads/reads tarballs from S3 (or local results/), groups by
(algorithm, seed, checkpoint), and reports:
  - Coverage gaps (missing seeds/checkpoints)
  - Case ID gaps (systematically missing evaluations)
  - Duplicate/invalid tarballs (tiny failed-run artifacts)
  - Metric reconciliation at key checkpoints (7K, 10K)
  - Actionable re-run recommendations

No GPU needed — reads existing JSON result files.

Usage:
    python analysis/reconcile_results.py --results_dir results/failure_curve_checkpointed
    python analysis/reconcile_results.py --s3_prefix /s3-data/continual-learning/alphaedit/results/failure_curve_checkpointed
    python analysis/reconcile_results.py --results_dir results/ --output analysis/reconciliation_report.json
"""

import argparse
import json
import tarfile
from pathlib import Path

import numpy as np


SEEDS = [42, 137, 2024, 7, 99]
ALGORITHMS = ["AlphaEdit", "MEMIT"]
EXPECTED_CHECKPOINTS = [2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000, 10000]
KEY_CHECKPOINTS = [7000, 10000]  # Focus for metric reconciliation


def find_result_tarballs(base_dir: Path) -> list[dict]:
    """Find all result tarballs and classify them."""
    tarballs = []
    for tar_path in sorted(base_dir.rglob("*.tar.gz")):
        info = {
            "path": str(tar_path),
            "size_bytes": tar_path.stat().st_size,
            "name": tar_path.name,
            "parent": str(tar_path.parent),
        }
        # Flag suspiciously small tarballs (< 1KB = likely failed runs)
        info["is_valid"] = info["size_bytes"] > 1024
        info["is_duplicate"] = (
            tar_path.name == "results.tar.gz"
            and (tar_path.parent / "alphaedit_results.tar.gz").exists()
        )
        tarballs.append(info)
    return tarballs


def find_result_jsons(base_dir: Path) -> list[Path]:
    """Find all per-case JSON result files (unpacked)."""
    return sorted(base_dir.rglob("*_edits-case_*.json"))


def parse_result_path(json_path: Path) -> dict | None:
    """
    Extract metadata from a result JSON path.

    Expected patterns:
        .../seed42/10000edits/alphaedit_results/AlphaEdit/run_000/100_edits-case_42.json
        .../AlphaEdit/run_000/100_edits-case_42.json
    """
    parts = json_path.parts
    info = {"path": str(json_path)}

    # Extract case_id from filename
    fname = json_path.stem  # e.g., "100_edits-case_42"
    if "-case_" in fname:
        try:
            info["case_id"] = int(fname.split("-case_")[1])
        except (ValueError, IndexError):
            return None
    else:
        return None

    # Extract num_edits from filename
    if "_edits" in fname:
        try:
            info["num_edits_per_batch"] = int(fname.split("_edits")[0])
        except (ValueError, IndexError):
            pass

    # Try to extract seed from path
    for part in parts:
        if part.startswith("seed"):
            try:
                info["seed"] = int(part[4:])
            except ValueError:
                pass
        # Extract total edits from path
        if part.endswith("edits") and part[:-5].isdigit():
            info["total_edits"] = int(part[:-5])

    # Extract algorithm from path
    for part in parts:
        if part in ALGORITHMS:
            info["algorithm"] = part

    return info


def check_coverage(results: list[dict]) -> dict:
    """Check which (algorithm, seed, checkpoint) combinations exist."""
    coverage = {}
    for r in results:
        alg = r.get("algorithm", "unknown")
        seed = r.get("seed", "unknown")
        total = r.get("total_edits", "unknown")
        key = (alg, seed, total)
        if key not in coverage:
            coverage[key] = {"count": 0, "case_ids": set()}
        coverage[key]["count"] += 1
        if "case_id" in r:
            coverage[key]["case_ids"].add(r["case_id"])
    return coverage


def find_case_id_gaps(case_ids: set, expected_max: int = 2000) -> list[int]:
    """Find missing case IDs in a set."""
    if not case_ids:
        return []
    expected = set(range(expected_max))
    # Only report gaps within the range of existing IDs
    max_seen = max(case_ids)
    expected_within_range = set(range(max_seen + 1))
    return sorted(expected_within_range - case_ids)


def compute_metric_stats(json_files: list[Path]) -> dict:
    """Compute aggregate metrics from a set of result JSONs."""
    from stats.aggregate import extract_metrics_from_case

    metrics_data = {"efficacy": [], "generalization": [], "specificity": []}

    for jf in json_files:
        try:
            with open(jf) as f:
                case_data = json.load(f)
            row = extract_metrics_from_case(case_data)
            for m in metrics_data:
                if row.get(m) is not None:
                    metrics_data[m].append(row[m])
        except (json.JSONDecodeError, KeyError):
            continue

    stats = {}
    for metric, values in metrics_data.items():
        if values:
            arr = np.array(values)
            stats[metric] = {
                "mean": float(np.mean(arr)),
                "std": float(np.std(arr, ddof=1)) if len(arr) > 1 else 0.0,
                "n": len(arr),
            }
    return stats


def reconcile(base_dir: Path, output_path: Path | None = None) -> dict:
    """Run full reconciliation and produce report."""
    report = {
        "base_dir": str(base_dir),
        "tarballs": [],
        "coverage_gaps": [],
        "case_id_gaps": {},
        "duplicate_tarballs": [],
        "metric_reconciliation": {},
        "recommendations": [],
    }

    # 1. Check tarballs
    tarballs = find_result_tarballs(base_dir)
    invalid = [t for t in tarballs if not t["is_valid"]]
    duplicates = [t for t in tarballs if t["is_duplicate"]]
    report["tarballs"] = {
        "total": len(tarballs),
        "valid": len(tarballs) - len(invalid),
        "invalid_small": invalid,
    }
    report["duplicate_tarballs"] = duplicates

    # 2. Find all result JSONs
    json_files = find_result_jsons(base_dir)
    results = [r for r in (parse_result_path(jf) for jf in json_files) if r is not None]

    # 3. Coverage analysis
    coverage = check_coverage(results)

    for alg in ALGORITHMS:
        for seed in SEEDS:
            for ckpt in EXPECTED_CHECKPOINTS:
                key = (alg, seed, ckpt)
                if key not in coverage:
                    report["coverage_gaps"].append({
                        "algorithm": alg,
                        "seed": seed,
                        "total_edits": ckpt,
                        "status": "missing",
                    })
                else:
                    # Check for case_id gaps
                    case_ids = coverage[key]["case_ids"]
                    gaps = find_case_id_gaps(case_ids)
                    if gaps:
                        gap_key = f"{alg}_seed{seed}_{ckpt}edits"
                        report["case_id_gaps"][gap_key] = {
                            "missing_ids": gaps[:20],  # First 20 for brevity
                            "total_missing": len(gaps),
                            "total_present": len(case_ids),
                        }

    # 4. Metric reconciliation at key checkpoints
    for ckpt in KEY_CHECKPOINTS:
        ckpt_stats = {}
        for alg in ALGORITHMS:
            alg_seeds = {}
            for seed in SEEDS:
                key = (alg, seed, ckpt)
                if key in coverage:
                    # Find JSON files for this combination
                    seed_files = [
                        Path(r["path"]) for r in results
                        if r.get("algorithm") == alg
                        and r.get("seed") == seed
                        and r.get("total_edits") == ckpt
                    ]
                    if seed_files:
                        alg_seeds[f"seed{seed}"] = compute_metric_stats(seed_files)
            if alg_seeds:
                ckpt_stats[alg] = alg_seeds
        if ckpt_stats:
            report["metric_reconciliation"][f"{ckpt}_edits"] = ckpt_stats

    # 5. Flag outliers (>2σ from cross-seed mean)
    for ckpt_key, ckpt_data in report["metric_reconciliation"].items():
        for alg, seed_data in ckpt_data.items():
            for metric in ["efficacy", "generalization", "specificity"]:
                values = [
                    s[metric]["mean"] for s in seed_data.values()
                    if metric in s
                ]
                if len(values) >= 3:
                    mean = np.mean(values)
                    std = np.std(values, ddof=1)
                    if std > 0:
                        for seed_key, stats in seed_data.items():
                            if metric in stats:
                                z = abs(stats[metric]["mean"] - mean) / std
                                if z > 2.0:
                                    report["recommendations"].append(
                                        f"OUTLIER: {alg} {seed_key} at {ckpt_key} "
                                        f"has {metric}={stats[metric]['mean']:.3f} "
                                        f"(z={z:.1f}, mean={mean:.3f}, std={std:.3f})"
                                    )

    # 6. Generate recommendations
    if report["coverage_gaps"]:
        missing_by_alg = {}
        for gap in report["coverage_gaps"]:
            alg = gap["algorithm"]
            if alg not in missing_by_alg:
                missing_by_alg[alg] = []
            missing_by_alg[alg].append(f"seed{gap['seed']}@{gap['total_edits']}")
        for alg, missing in missing_by_alg.items():
            report["recommendations"].append(
                f"RE-RUN NEEDED: {alg} missing {len(missing)} checkpoints: "
                + ", ".join(missing[:10])
            )

    if duplicates:
        report["recommendations"].append(
            f"CLEANUP: {len(duplicates)} duplicate/failed tarballs to remove"
        )

    # Print human-readable summary
    print("=" * 70)
    print("RECONCILIATION REPORT")
    print("=" * 70)
    print(f"\nBase directory: {base_dir}")
    print(f"Result JSONs found: {len(results)}")
    print(f"Tarballs: {report['tarballs']['total']} ({report['tarballs']['valid']} valid)")
    print(f"Coverage gaps: {len(report['coverage_gaps'])}")
    print(f"Case ID gap groups: {len(report['case_id_gaps'])}")
    print(f"Duplicate tarballs: {len(duplicates)}")

    if report["recommendations"]:
        print(f"\n{'─' * 70}")
        print("RECOMMENDATIONS:")
        for rec in report["recommendations"]:
            print(f"  • {rec}")

    print("=" * 70)

    # Save JSON report
    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            # Convert sets to lists for JSON serialization
            json.dump(report, f, indent=2, default=str)
        print(f"\nFull report saved to: {output_path}")

    return report


def main():
    parser = argparse.ArgumentParser(
        description="Reconcile failure-curve results across seeds and checkpoints"
    )
    parser.add_argument(
        "--results_dir",
        type=Path,
        default=None,
        help="Local results directory to scan",
    )
    parser.add_argument(
        "--s3_prefix",
        type=Path,
        default=None,
        help="S3-mounted path (e.g., /s3-data/.../results/failure_curve_checkpointed)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("analysis/reconciliation_report.json"),
        help="Output path for JSON report",
    )
    args = parser.parse_args()

    # Determine base directory
    if args.s3_prefix and args.s3_prefix.exists():
        base_dir = args.s3_prefix
    elif args.results_dir:
        base_dir = args.results_dir
    else:
        # Default: try local results first, then S3 mount
        local = Path("results/failure_curve_checkpointed")
        s3 = Path("/s3-data/continual-learning/alphaedit/results/failure_curve_checkpointed")
        if local.exists():
            base_dir = local
        elif s3.exists():
            base_dir = s3
        else:
            print("ERROR: No results directory found. Specify --results_dir or --s3_prefix.")
            raise SystemExit(1)

    reconcile(base_dir, args.output)


if __name__ == "__main__":
    main()

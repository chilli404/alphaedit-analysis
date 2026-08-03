#!/usr/bin/env python3
"""Post-GPU analysis for the interference-aware scheduling experiment.

Loads results from all experimental arms and produces:
  1. Headline table: cumulative efficacy at {2K, 5K, 7K, 10K} for each arm
  2. Prospective validation: geometry-predicted exposure vs actual retention
  3. Installation-quality equivalence check
  4. LaTeX table fragment + placeholder macros

Requires completed GPU runs (results in matched_ordering/ structure).

Usage:
    uv run python scheduling/analyze_scheduling.py --seeds 42
    uv run python scheduling/analyze_scheduling.py --seeds 42 2024
    uv run python scheduling/analyze_scheduling.py --seeds 42 --arms AlphaEdit:greedy_minmax,AlphaEdit:key_clustered
"""

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

RESULT_ROOT = Path(os.environ.get("RESULT_ROOT", PROJECT_ROOT / "results"))
REPORTS_DIR = PROJECT_ROOT / "scheduling" / "reports"
LATEX_DIR = PROJECT_ROOT / "scheduling" / "latex"

# Edit count checkpoints to report
REPORT_EDITS = [2000, 5000, 7000, 10000]

# Arms to compare (alg, ordering)
DEFAULT_ARMS = [
    ("AlphaEdit", "key_clustered"),     # existing baseline (best retention)
    ("AlphaEdit", "key_dispersed"),     # existing baseline (worst retention)
    ("AlphaEdit", "greedy_minmax"),     # headline method
    ("AlphaEdit", "cluster_topo"),      # secondary method
    ("AlphaEdit", "random"),            # reshuffling control
    ("MEMIT-Seq-lp1.0-ld0.0-cache0", "greedy_minmax"),  # algorithmic fix + scheduling
]


def load_arm_results(seed: int, alg: str, ordering: str) -> dict | None:
    """Load full_eval results for one experimental arm.

    Returns None if results not yet available.
    """
    eval_path = (RESULT_ROOT / "matched_ordering" / alg / ordering
                 / f"seed{seed}" / f"full_eval_seed{seed}.json")
    if not eval_path.exists():
        return None

    with open(eval_path) as f:
        return json.load(f)


def extract_checkpoint_metrics(results: dict, edits: int) -> dict | None:
    """Extract metrics for a specific edit count from full_eval results."""
    key = f"{edits}_edits"
    if key not in results:
        # Try numeric key
        if str(edits) in results:
            return results[str(edits)]
        return None
    return results[key]


def build_headline_table(all_arm_results: dict) -> str:
    """Build markdown headline table comparing all arms."""
    lines = [
        "## Headline Table: Scheduling Experiment Results",
        "",
        "| Arm | " + " | ".join(f"Eff@{e//1000}K" for e in REPORT_EDITS)
        + " | 1st-1K Ret | Latest-1K Eff |",
        "|-----|" + "|".join([":--------:"] * (len(REPORT_EDITS) + 2)) + "|",
    ]

    for (alg, ordering, seed), results in sorted(all_arm_results.items()):
        if results is None:
            continue
        arm_name = f"{alg}/{ordering}/s{seed}"

        eff_cols = []
        for edits in REPORT_EDITS:
            metrics = extract_checkpoint_metrics(results, edits)
            if metrics and "all_facts" in metrics:
                eff_cols.append(f"{metrics['all_facts']['efficacy']:.3f}")
            else:
                eff_cols.append("—")

        # First-1K retention and latest-1K efficacy at max checkpoint
        last_ckpt = extract_checkpoint_metrics(results, REPORT_EDITS[-1])
        if last_ckpt:
            first_1k = last_ckpt.get("first_1k", {}).get("efficacy", float("nan"))
            latest_1k = last_ckpt.get("latest_1k", {}).get("efficacy", float("nan"))
            first_1k_str = f"{first_1k:.3f}" if not np.isnan(first_1k) else "—"
            latest_1k_str = f"{latest_1k:.3f}" if not np.isnan(latest_1k) else "—"
        else:
            first_1k_str = "—"
            latest_1k_str = "—"

        lines.append(f"| {arm_name} | {' | '.join(eff_cols)} | {first_1k_str} | {latest_1k_str} |")

    return "\n".join(lines)


def prospective_validation(geometry_path: Path, arm_results: dict) -> str:
    """Correlate geometry-predicted exposure with actual retention.

    Uses the Phase 2 geometry metrics to predict which orderings should
    retain better, then checks against actual outcomes.
    """
    if not geometry_path.exists():
        return "## Prospective Validation\n\nGeometry metrics not found. Run validate_ordering.py first.\n"

    with open(geometry_path) as f:
        geometry = json.load(f)

    lines = [
        "## Prospective Validation",
        "",
        "Correlating geometry-predicted interference with actual retention:",
        "",
        "| Ordering | Predicted Exposure (frac>0.3) | Actual 1st-1K Retention@10K |",
        "|----------|:-----------------------------:|:---------------------------:|",
    ]

    data_points = []
    for ordering_name, geo_metrics in geometry.items():
        predicted = geo_metrics.get("first_1k", {}).get("frac_above_0.3", float("nan"))
        # Find actual retention
        actual = float("nan")
        for (alg, ordering, seed), results in arm_results.items():
            if ordering == ordering_name and alg == "AlphaEdit" and results:
                last_ckpt = extract_checkpoint_metrics(results, REPORT_EDITS[-1])
                if last_ckpt and "first_1k" in last_ckpt:
                    actual = last_ckpt["first_1k"].get("efficacy", float("nan"))
                break

        pred_str = f"{predicted:.4f}" if not np.isnan(predicted) else "—"
        act_str = f"{actual:.4f}" if not np.isnan(actual) else "—"
        lines.append(f"| {ordering_name} | {pred_str} | {act_str} |")

        if not np.isnan(predicted) and not np.isnan(actual):
            data_points.append((predicted, actual))

    if len(data_points) >= 3:
        preds, actuals = zip(*data_points)
        # Negative correlation expected (higher exposure → lower retention)
        corr = np.corrcoef(preds, actuals)[0, 1]
        lines.extend([
            "",
            f"Pearson correlation (exposure vs retention): r = {corr:.3f}",
            f"Expected: negative (more exposure → less retention).",
        ])

    return "\n".join(lines)


def installation_quality_check(arm_results: dict) -> str:
    """Check that installation quality (latest-batch efficacy) is equivalent."""
    lines = [
        "## Installation-Quality Equivalence",
        "",
        "Verifying that scheduling does not degrade edit installation:",
        "",
        "| Arm | Latest-1K Efficacy@10K | Deviation from canonical |",
        "|-----|:----------------------:|:------------------------:|",
    ]

    canonical_eff = None
    arm_effs = {}

    for (alg, ordering, seed), results in sorted(arm_results.items()):
        if results is None:
            continue
        last_ckpt = extract_checkpoint_metrics(results, REPORT_EDITS[-1])
        if not last_ckpt:
            continue
        latest_eff = last_ckpt.get("latest_1k", {}).get("efficacy", float("nan"))
        arm_effs[(alg, ordering, seed)] = latest_eff

        if ordering == "key_clustered" and alg == "AlphaEdit":
            canonical_eff = latest_eff

    for (alg, ordering, seed), eff in sorted(arm_effs.items()):
        arm_name = f"{alg}/{ordering}/s{seed}"
        if canonical_eff is not None and not np.isnan(eff) and not np.isnan(canonical_eff):
            dev = eff - canonical_eff
            dev_pct = dev / max(canonical_eff, 1e-8) * 100
            flag = " **FLAG**" if abs(dev_pct) > 3 else ""
            lines.append(f"| {arm_name} | {eff:.4f} | {dev_pct:+.1f}%{flag} |")
        else:
            lines.append(f"| {arm_name} | {eff:.4f} if not np.isnan(eff) else '—' | — |")

    lines.extend([
        "",
        "Flag threshold: >3% deviation from canonical (key_clustered) baseline.",
    ])

    return "\n".join(lines)


def emit_latex_macros(arm_results: dict, output_dir: Path) -> None:
    """Write LaTeX macros to scheduling/latex/scheduling_macros.tex."""
    output_dir.mkdir(parents=True, exist_ok=True)
    macros = []
    macros.append("% Auto-generated by scheduling/analyze_scheduling.py")
    macros.append("% Do NOT edit manually — re-run the analysis script to update.")
    macros.append("")

    for (alg, ordering, seed), results in sorted(arm_results.items()):
        if results is None:
            continue

        # Naming: \schedAlgOrderingSeedMetric
        prefix = ordering.replace("_", "").title().replace(" ", "")

        for edits in REPORT_EDITS:
            metrics = extract_checkpoint_metrics(results, edits)
            if metrics and "all_facts" in metrics:
                eff = metrics["all_facts"]["efficacy"]
                macro_name = f"\\sched{prefix}Eff{edits//1000}K"
                macros.append(f"\\newcommand{{{macro_name}}}{{{eff:.1f}\\%}}")

        # First-1K retention at 10K
        last_ckpt = extract_checkpoint_metrics(results, REPORT_EDITS[-1])
        if last_ckpt and "first_1k" in last_ckpt:
            ret = last_ckpt["first_1k"].get("efficacy", 0)
            macro_name = f"\\sched{prefix}FirstCohortTenK"
            macros.append(f"\\newcommand{{{macro_name}}}{{{ret:.1f}\\%}}")

    macros.append("")
    output_path = output_dir / "scheduling_macros.tex"
    with open(output_path, "w") as f:
        f.write("\n".join(macros))
    print(f"  LaTeX macros written to: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Post-GPU analysis for scheduling experiment"
    )
    parser.add_argument("--seeds", type=int, nargs="+", default=[42],
                        help="Seeds to analyze (default: 42)")
    parser.add_argument("--arms", type=str, default=None,
                        help="Comma-separated ALG:ORDERING pairs (default: all standard arms)")
    args = parser.parse_args()

    if args.arms:
        arms = []
        for pair in args.arms.split(","):
            alg, ordering = pair.split(":")
            arms.append((alg, ordering))
    else:
        arms = DEFAULT_ARMS

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*70}")
    print("Scheduling Experiment — Post-GPU Analysis")
    print(f"  Seeds: {args.seeds}")
    print(f"  Arms:  {len(arms)}")
    print(f"{'='*70}")

    # Load all available results
    all_arm_results = {}
    available = 0
    for seed in args.seeds:
        for alg, ordering in arms:
            results = load_arm_results(seed, alg, ordering)
            all_arm_results[(alg, ordering, seed)] = results
            if results:
                available += 1
                print(f"  Loaded: {alg}/{ordering}/seed{seed}")
            else:
                print(f"  Missing: {alg}/{ordering}/seed{seed}")

    if available == 0:
        print("\n  No results available yet. Run GPU experiments first:")
        for alg, ordering in arms:
            for seed in args.seeds:
                print(f"    bash scripts/run_scheduling_experiment.sh {seed} {alg} {ordering}")
        sys.exit(0)

    # Build headline table
    print(f"\n{'─'*70}")
    table = build_headline_table(all_arm_results)
    print(table)

    # Prospective validation (per seed)
    prospective_sections = []
    for seed in args.seeds:
        geometry_path = REPORTS_DIR / f"geometry_metrics_seed{seed}.json"
        seed_results = {k: v for k, v in all_arm_results.items() if k[2] == seed}
        section = prospective_validation(geometry_path, seed_results)
        prospective_sections.append(f"### Seed {seed}\n\n{section}")

    # Installation quality check
    install_check = installation_quality_check(all_arm_results)

    # Emit LaTeX macros
    emit_latex_macros(all_arm_results, LATEX_DIR)

    # Combine into full report
    report_lines = [
        "# Scheduling Experiment — Analysis Report",
        "",
        f"Seeds: {args.seeds}",
        f"Arms: {len(arms)} ({available} with results)",
        "",
        table,
        "",
        "\n\n".join(prospective_sections),
        "",
        install_check,
        "",
    ]

    report_path = REPORTS_DIR / "scheduling_analysis.md"
    with open(report_path, "w") as f:
        f.write("\n".join(report_lines))
    print(f"\n  Full report: {report_path}")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()

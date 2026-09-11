"""Reconcile MEMIT-Seq numerical contradiction (Criticism 10).

Main Table 2 reports seed 2024 oldest cohort: 91.2% vs 83.7% (7.5pp gap).
Appendix F reports: 88.9% vs 67.3% (21.6pp gap).

This script loads raw per-case JSONs and computes authoritative cohort metrics.

Usage:
    python -m analysis.reconcile_memit_seq
"""

import json
import os
from collections import defaultdict
from pathlib import Path

import numpy as np

RESULT_ROOT = Path(os.environ.get("RESULT_ROOT", Path(__file__).parent.parent / "results"))
FC_DIR = RESULT_ROOT / "failure_curve_checkpointed"

BATCH_SIZE = 100
SEEDS = [42, 2024]
CHECKPOINTS = [5000, 7000, 9000, 10000]
ALGORITHMS = {
    "AlphaEdit": "AlphaEdit",
    "MEMIT": "MEMIT",
    "MEMIT-Seq": "MEMIT-Seq-lp1.0-ld0.0-cache0",
}


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


def load_cohort_metrics(seed, edits, alg_dir):
    """Load per-cohort efficacy/paraphrase/neighborhood from raw case JSONs.

    Uses probability-preference metric (official AlphaEdit metric) as primary.
    Falls back to argmax if _probs data is unavailable.
    """
    run_dir = FC_DIR / f"seed{seed}" / f"{edits}edits" / alg_dir / "run_000"
    if not run_dir.exists():
        return None

    cohorts = defaultdict(lambda: defaultdict(list))
    for f_path in run_dir.glob("*_edits-case_*.json"):
        with open(f_path) as f:
            data = json.load(f)
        case_id = data.get("case_id")
        if case_id is None:
            continue
        post = data.get("post", {})
        cohort_idx = case_id // BATCH_SIZE

        # Probability-preference (primary)
        eff_prob = _prob_pref_rewrite(post.get("rewrite_prompts_probs", []))
        para_prob = _prob_pref_rewrite(post.get("paraphrase_prompts_probs", []))
        neigh_prob = _prob_pref_neighborhood(post.get("neighborhood_prompts_probs", []))

        # Use prob-pref if available, fallback to argmax
        for metric_name, prob_val, json_key in [
            ("efficacy", eff_prob, "rewrite_prompts_correct"),
            ("paraphrase", para_prob, "paraphrase_prompts_correct"),
            ("neighborhood", neigh_prob, "neighborhood_prompts_correct"),
        ]:
            if prob_val is not None:
                cohorts[cohort_idx][metric_name].append(prob_val)
            else:
                vals = post.get(json_key)
                if isinstance(vals, list) and vals:
                    cohorts[cohort_idx][metric_name].append(sum(vals) / len(vals))

    if not cohorts:
        return None

    return {
        idx: {k: float(np.mean(v)) for k, v in vals.items()}
        for idx, vals in sorted(cohorts.items())
    }


def cohort_summary(cohorts, total_edits):
    """Summarize into oldest/middle/newest thirds."""
    n_batches = total_edits // BATCH_SIZE
    third = n_batches // 3

    groups = {
        "oldest": range(0, third),
        "middle": range(third, 2 * third),
        "newest": range(2 * third, n_batches),
    }

    summary = {}
    for label, batch_range in groups.items():
        metrics = defaultdict(list)
        for idx in batch_range:
            if idx in cohorts:
                for k, v in cohorts[idx].items():
                    metrics[k].append(v)
        if metrics:
            summary[label] = {
                k: float(np.mean(v)) for k, v in metrics.items()
            }
            summary[label]["n_batches"] = len(metrics.get("efficacy", []))
        else:
            summary[label] = None
    return summary


def print_table(results):
    """Print formatted comparison table."""
    header = f"{'Seed':<6} {'Algorithm':<12} {'Edits':<7} {'Cohort':<8} {'Efficacy':>9} {'Paraphrase':>11} {'Neighborhood':>13}"
    print("=" * len(header))
    print(header)
    print("=" * len(header))

    for seed in SEEDS:
        for edits in CHECKPOINTS:
            for alg_name, alg_dir in ALGORITHMS.items():
                key = (seed, edits, alg_name)
                if key not in results:
                    continue
                summary = results[key]
                for cohort in ("oldest", "middle", "newest"):
                    if summary.get(cohort) is None:
                        continue
                    m = summary[cohort]
                    print(
                        f"{seed:<6} {alg_name:<12} {edits:<7} {cohort:<8} "
                        f"{m['efficacy']*100:>8.1f}% "
                        f"{m.get('paraphrase', 0)*100:>10.1f}% "
                        f"{m.get('neighborhood', 0)*100:>12.1f}%"
                    )
            print("-" * len(header))


def print_critical_comparison(results):
    """Print the specific comparison from Criticism 10."""
    print("\n" + "=" * 70)
    print("CRITICAL COMPARISON: Oldest cohort at 10K edits (Criticism 10)")
    print("=" * 70)
    print(f"\n{'Seed':<6} {'Algorithm':<12} {'Oldest Eff':>11} {'Oldest Para':>12} {'Oldest Neigh':>13}")
    print("-" * 60)

    for seed in SEEDS:
        for alg_name in ["AlphaEdit", "MEMIT-Seq"]:
            key = (seed, 10000, alg_name)
            if key not in results:
                print(f"{seed:<6} {alg_name:<12} {'N/A':>11}")
                continue
            s = results[key]
            if s.get("oldest") is None:
                print(f"{seed:<6} {alg_name:<12} {'N/A':>11}")
                continue
            m = s["oldest"]
            print(
                f"{seed:<6} {alg_name:<12} "
                f"{m['efficacy']*100:>10.1f}% "
                f"{m.get('paraphrase', 0)*100:>11.1f}% "
                f"{m.get('neighborhood', 0)*100:>12.1f}%"
            )
        if seed == 42:
            print()

    # Compute the gap
    for seed in SEEDS:
        ae_key = (seed, 10000, "AlphaEdit")
        ms_key = (seed, 10000, "MEMIT-Seq")
        if ae_key in results and ms_key in results:
            ae = results[ae_key].get("oldest")
            ms = results[ms_key].get("oldest")
            if ae and ms:
                gap = (ms["efficacy"] - ae["efficacy"]) * 100
                print(f"\n  Seed {seed} gap (MEMIT-Seq - AlphaEdit): {gap:+.1f} pp")


def print_per_batch_detail(seed, edits, alg_name, alg_dir):
    """Print per-batch breakdown for detailed reconciliation."""
    cohorts = load_cohort_metrics(seed, edits, alg_dir)
    if cohorts is None:
        return
    n_batches = edits // BATCH_SIZE
    print(f"\n--- Per-batch detail: seed={seed}, edits={edits}, alg={alg_name} ---")
    print(f"{'Batch':<6} {'Efficacy':>9} {'Paraphrase':>11} {'Neighborhood':>13} {'N':>4}")

    # Show first 10 (oldest) and last 10 (newest)
    show_indices = sorted(cohorts.keys())[:10] + ["..."] + sorted(cohorts.keys())[-10:]
    for idx in show_indices:
        if idx == "...":
            print("  ...")
            continue
        if idx not in cohorts:
            continue
        m = cohorts[idx]
        print(
            f"{idx:<6} "
            f"{m.get('efficacy', 0)*100:>8.1f}% "
            f"{m.get('paraphrase', 0)*100:>10.1f}% "
            f"{m.get('neighborhood', 0)*100:>12.1f}%"
        )


def main():
    print("Reconciling MEMIT-Seq numbers from raw evaluation output")
    print(f"Result root: {RESULT_ROOT}")
    print(f"Failure curve dir: {FC_DIR}")
    print()

    # Load all cohort data
    results = {}
    for seed in SEEDS:
        for edits in CHECKPOINTS:
            for alg_name, alg_dir in ALGORITHMS.items():
                cohorts = load_cohort_metrics(seed, edits, alg_dir)
                if cohorts is not None:
                    summary = cohort_summary(cohorts, edits)
                    results[(seed, edits, alg_name)] = summary

    # Print full table
    print_table(results)

    # Print critical comparison
    print_critical_comparison(results)

    # Per-batch detail for the contested numbers
    for seed in SEEDS:
        print_per_batch_detail(seed, 10000, "AlphaEdit", "AlphaEdit")
        print_per_batch_detail(seed, 10000, "MEMIT-Seq", "MEMIT-Seq-lp1.0-ld0.0-cache0")

    # Summary
    print("\n" + "=" * 70)
    print("AUTHORITATIVE NUMBERS (from raw per-case JSONs)")
    print("=" * 70)
    print("\nUse these to reconcile Main Table 2 vs Appendix F.")
    print("The paper previously reported two conflicting sets:")
    print("  Main Table 2: seed 2024 oldest cohort 91.2% vs 83.7% (7.5pp)")
    print("  Appendix F:   seed 2024 oldest cohort 88.9% vs 67.3% (21.6pp)")
    print("\nThe ground truth from raw output:")
    for seed in SEEDS:
        for alg_name in ["AlphaEdit", "MEMIT-Seq"]:
            key = (seed, 10000, alg_name)
            if key in results and results[key].get("oldest"):
                m = results[key]["oldest"]
                print(f"  seed {seed} {alg_name:<12} oldest efficacy: {m['efficacy']*100:.1f}%")


if __name__ == "__main__":
    main()

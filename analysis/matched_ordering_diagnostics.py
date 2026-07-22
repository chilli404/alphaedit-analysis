#!/usr/bin/env python3
"""
Matched Ordering Diagnostics — Local (no GPU) cohort balance checks.

Verifies that the clustered/dispersed orderings do NOT have systematic
difficulty differences by edit age (1K cohorts). Since the same facts are
used in both orderings, global difficulty is matched, but temporal placement
may introduce age-conditioned bias.

For every 1K cohort, compares:
  - Relation composition (entropy, top-relation fraction)
  - Target length (token count)
  - Prompt length (token count)
  - Subject length (character count)

Also computes:
  - Per-batch relation diversity (already validated by generator)
  - Cross-stream cohort alignment summary

Usage:
    uv run python -m analysis.matched_ordering_diagnostics \
        --seed 42 --output_dir results/matched_ordering/diagnostics

    # Or directly:
    uv run python analysis/matched_ordering_diagnostics.py --seed 42
"""

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def load_stream(path: Path) -> list:
    with open(path) as f:
        return json.load(f)


def cohort_stats(records: list, cohort_size: int = 1000) -> list:
    """Compute per-cohort statistics."""
    n_cohorts = len(records) // cohort_size
    cohorts = []

    for c in range(n_cohorts):
        cohort = records[c * cohort_size: (c + 1) * cohort_size]

        relations = [r["requested_rewrite"]["relation_id"] for r in cohort]
        targets = [r["requested_rewrite"]["target_new"]["str"] for r in cohort]
        subjects = [r["requested_rewrite"]["subject"] for r in cohort]
        prompts = [r["requested_rewrite"]["prompt"] for r in cohort]

        # Relation composition
        rel_counts = Counter(relations)
        n = len(relations)
        rel_probs = np.array(list(rel_counts.values())) / n
        rel_entropy = -np.sum(rel_probs * np.log2(rel_probs + 1e-10))
        top_rel_frac = max(rel_counts.values()) / n
        n_unique_rels = len(rel_counts)

        # Length statistics
        target_lengths = [len(t.split()) for t in targets]
        subject_lengths = [len(s) for s in subjects]
        prompt_lengths = [len(p.split()) for p in prompts]

        cohorts.append({
            "cohort_idx": c,
            "edit_range": f"{c * cohort_size + 1}-{(c + 1) * cohort_size}",
            "n_unique_relations": n_unique_rels,
            "relation_entropy": float(rel_entropy),
            "top_relation_fraction": float(top_rel_frac),
            "top_relation": rel_counts.most_common(1)[0][0],
            "mean_target_length": float(np.mean(target_lengths)),
            "std_target_length": float(np.std(target_lengths)),
            "mean_subject_length": float(np.mean(subject_lengths)),
            "std_subject_length": float(np.std(subject_lengths)),
            "mean_prompt_length": float(np.mean(prompt_lengths)),
            "std_prompt_length": float(np.std(prompt_lengths)),
        })

    return cohorts


def batch_diversity(records: list, batch_size: int = 100) -> list:
    """Number of unique relations per batch."""
    n_batches = len(records) // batch_size
    diversities = []
    for b in range(n_batches):
        batch = records[b * batch_size: (b + 1) * batch_size]
        rels = set(r["requested_rewrite"]["relation_id"] for r in batch)
        diversities.append(len(rels))
    return diversities


def compare_cohorts(clust_cohorts: list, disp_cohorts: list) -> dict:
    """Compare cohort stats between orderings."""
    comparisons = {
        "target_length_max_diff": 0.0,
        "subject_length_max_diff": 0.0,
        "prompt_length_max_diff": 0.0,
        "relation_entropy_max_diff": 0.0,
    }

    for cc, dc in zip(clust_cohorts, disp_cohorts):
        comparisons["target_length_max_diff"] = max(
            comparisons["target_length_max_diff"],
            abs(cc["mean_target_length"] - dc["mean_target_length"])
        )
        comparisons["subject_length_max_diff"] = max(
            comparisons["subject_length_max_diff"],
            abs(cc["mean_subject_length"] - dc["mean_subject_length"])
        )
        comparisons["prompt_length_max_diff"] = max(
            comparisons["prompt_length_max_diff"],
            abs(cc["mean_prompt_length"] - dc["mean_prompt_length"])
        )
        comparisons["relation_entropy_max_diff"] = max(
            comparisons["relation_entropy_max_diff"],
            abs(cc["relation_entropy"] - dc["relation_entropy"])
        )

    return comparisons


def main():
    parser = argparse.ArgumentParser(
        description="Local diagnostics for matched ordering streams"
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--stream_dir", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--cohort_size", type=int, default=1000)
    parser.add_argument("--batch_size", type=int, default=100)
    args = parser.parse_args()

    # Resolve paths
    if args.stream_dir:
        stream_dir = Path(args.stream_dir)
    else:
        stream_dir = PROJECT_ROOT / "results" / "matched_ordering"

    if args.output_dir:
        out_dir = Path(args.output_dir)
    else:
        out_dir = stream_dir / "diagnostics"
    out_dir.mkdir(parents=True, exist_ok=True)

    clust_path = stream_dir / "orderings" / f"clustered_seed{args.seed}.json"
    disp_path = stream_dir / "orderings" / f"dispersed_seed{args.seed}.json"

    if not clust_path.exists() or not disp_path.exists():
        print(f"ERROR: Stream files not found in {stream_dir}")
        sys.exit(1)

    print(f"\n{'='*70}")
    print("Matched Ordering Diagnostics (Local)")
    print(f"  Seed: {args.seed}")
    print(f"  Streams: {stream_dir}")
    print(f"  Cohort size: {args.cohort_size}")
    print(f"{'='*70}")

    # Load streams
    print("\n  Loading streams...")
    clustered = load_stream(clust_path)
    dispersed = load_stream(disp_path)
    print(f"  Clustered: {len(clustered)} records")
    print(f"  Dispersed: {len(dispersed)} records")

    # Verify same facts
    clust_ids = set(r["case_id"] for r in clustered)
    disp_ids = set(r["case_id"] for r in dispersed)
    assert clust_ids == disp_ids, "Streams contain different facts!"
    print(f"  Verified: identical fact sets ({len(clust_ids)} case_ids)")

    # Cohort statistics
    print(f"\n  Computing cohort statistics (cohort_size={args.cohort_size})...")
    clust_cohorts = cohort_stats(clustered, args.cohort_size)
    disp_cohorts = cohort_stats(dispersed, args.cohort_size)

    # Print cohort comparison table
    print(f"\n  {'─'*72}")
    print(f"  CLUSTERED cohorts:")
    print(f"  {'Cohort':<10} {'Edits':<12} {'Rels':>5} {'Entropy':>8} {'TopFrac':>8} "
          f"{'TgtLen':>7} {'SubjLen':>8} {'PmtLen':>7}")
    print(f"  {'─'*72}")
    for c in clust_cohorts:
        print(f"  {c['cohort_idx']:<10} {c['edit_range']:<12} "
              f"{c['n_unique_relations']:>5} {c['relation_entropy']:>8.2f} "
              f"{c['top_relation_fraction']:>8.3f} "
              f"{c['mean_target_length']:>7.2f} {c['mean_subject_length']:>8.1f} "
              f"{c['mean_prompt_length']:>7.1f}")

    print(f"\n  {'─'*72}")
    print(f"  DISPERSED cohorts:")
    print(f"  {'Cohort':<10} {'Edits':<12} {'Rels':>5} {'Entropy':>8} {'TopFrac':>8} "
          f"{'TgtLen':>7} {'SubjLen':>8} {'PmtLen':>7}")
    print(f"  {'─'*72}")
    for c in disp_cohorts:
        print(f"  {c['cohort_idx']:<10} {c['edit_range']:<12} "
              f"{c['n_unique_relations']:>5} {c['relation_entropy']:>8.2f} "
              f"{c['top_relation_fraction']:>8.3f} "
              f"{c['mean_target_length']:>7.2f} {c['mean_subject_length']:>8.1f} "
              f"{c['mean_prompt_length']:>7.1f}")

    # Cross-ordering comparison
    print(f"\n  {'─'*72}")
    print(f"  CROSS-ORDERING COHORT COMPARISON:")
    comparisons = compare_cohorts(clust_cohorts, disp_cohorts)
    print(f"    Max target length diff between same-cohort:  {comparisons['target_length_max_diff']:.3f} words")
    print(f"    Max subject length diff between same-cohort: {comparisons['subject_length_max_diff']:.1f} chars")
    print(f"    Max prompt length diff between same-cohort:  {comparisons['prompt_length_max_diff']:.1f} words")
    print(f"    Max relation entropy diff between cohorts:   {comparisons['relation_entropy_max_diff']:.3f} bits")

    # Key issue check: does clustered place easy/hard relations at specific ages?
    print(f"\n  {'─'*72}")
    print(f"  CLUSTERED RELATION PLACEMENT (which relations dominate which cohorts):")
    for c in clust_cohorts:
        print(f"    Cohort {c['cohort_idx']} ({c['edit_range']}): "
              f"top={c['top_relation']} ({c['top_relation_fraction']:.0%}), "
              f"{c['n_unique_relations']} unique rels")

    # Batch diversity
    clust_div = batch_diversity(clustered, args.batch_size)
    disp_div = batch_diversity(dispersed, args.batch_size)
    print(f"\n  {'─'*72}")
    print(f"  BATCH DIVERSITY (relations per batch):")
    print(f"    Clustered: mean={np.mean(clust_div):.1f}, min={np.min(clust_div)}, "
          f"max={np.max(clust_div)}, std={np.std(clust_div):.1f}")
    print(f"    Dispersed: mean={np.mean(disp_div):.1f}, min={np.min(disp_div)}, "
          f"max={np.max(disp_div)}, std={np.std(disp_div):.1f}")

    # Save results
    results = {
        "seed": args.seed,
        "n_records": len(clustered),
        "cohort_size": args.cohort_size,
        "batch_size": args.batch_size,
        "clustered_cohorts": clust_cohorts,
        "dispersed_cohorts": disp_cohorts,
        "cross_comparison": comparisons,
        "batch_diversity": {
            "clustered": {"mean": float(np.mean(clust_div)), "min": int(np.min(clust_div)),
                         "max": int(np.max(clust_div)), "std": float(np.std(clust_div))},
            "dispersed": {"mean": float(np.mean(disp_div)), "min": int(np.min(disp_div)),
                         "max": int(np.max(disp_div)), "std": float(np.std(disp_div))},
        },
    }

    out_path = out_dir / f"cohort_balance_seed{args.seed}.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  Saved: {out_path}")

    # Assessment
    print(f"\n  {'='*72}")
    print("  ASSESSMENT:")
    # Clustered has low relation entropy per cohort (1-2 rels dominate)
    # Dispersed has high entropy (many relations per cohort)
    clust_mean_entropy = np.mean([c["relation_entropy"] for c in clust_cohorts])
    disp_mean_entropy = np.mean([c["relation_entropy"] for c in disp_cohorts])
    print(f"    Clustered mean cohort entropy: {clust_mean_entropy:.2f} bits")
    print(f"    Dispersed mean cohort entropy: {disp_mean_entropy:.2f} bits")

    # Check difficulty balance
    clust_tgt_var = np.std([c["mean_target_length"] for c in clust_cohorts])
    disp_tgt_var = np.std([c["mean_target_length"] for c in disp_cohorts])
    print(f"    Clustered cross-cohort target length std: {clust_tgt_var:.3f}")
    print(f"    Dispersed cross-cohort target length std: {disp_tgt_var:.3f}")

    if clust_tgt_var > 0.3:
        print(f"    ⚠ WARNING: Clustered ordering shows substantial target length "
              f"variation across cohorts ({clust_tgt_var:.3f}). This may indicate "
              f"difficulty confound by edit age.")
    else:
        print(f"    ✓ Target length variation across cohorts is small — no obvious "
              f"difficulty confound detected.")

    print(f"  {'='*72}")


if __name__ == "__main__":
    main()

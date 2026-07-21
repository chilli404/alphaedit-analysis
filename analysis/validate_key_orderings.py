#!/usr/bin/env python3
"""
Validate key-clustered/key-dispersed orderings and report detailed statistics.

Reports:
  - Structural validation (same case_ids, no duplicates, permutation-only)
  - Key alignment verification (case_id-based, not positional)
  - Detailed within-batch cosine statistics (median, percentiles, tail fractions)
  - Mean batch maximum cosine
  - Fraction of pairs above thresholds (0.2, 0.3, 0.5)
  - Future-key exposure by cohort
  - Cohort-level difficulty balance (relation composition, target/subject length)
  - Prefix geometry at 1K-4K checkpoints (effective rank, condition, spectral entropy, max eigval share)

Usage:
    uv run python analysis/validate_key_orderings.py \
        --seed 42 \
        --keys_path results/matched_ordering/key_geometry/keys_seed42_layer6.npz \
        --stream_dir results/matched_ordering
"""

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def load_stream(path):
    with open(path) as f:
        return json.load(f)


def structural_validation(key_clustered, key_dispersed):
    """Verify streams are valid permutations of the same records."""
    print("\n  STRUCTURAL VALIDATION")
    print(f"  {'─'*60}")

    kc_ids = [r["case_id"] for r in key_clustered]
    kd_ids = [r["case_id"] for r in key_dispersed]

    # Same length
    assert len(kc_ids) == len(kd_ids), f"Length mismatch: {len(kc_ids)} vs {len(kd_ids)}"
    print(f"    Length: {len(kc_ids)} == {len(kd_ids)} ✓")

    # Same case_ids (as sets)
    kc_set = set(kc_ids)
    kd_set = set(kd_ids)
    assert kc_set == kd_set, f"Case ID sets differ: {len(kc_set - kd_set)} only in clustered, {len(kd_set - kc_set)} only in dispersed"
    print(f"    Same case_id set: {len(kc_set)} == {len(kd_set)} ✓")

    # No duplicates
    assert len(kc_ids) == len(kc_set), f"Duplicates in key_clustered: {len(kc_ids) - len(kc_set)}"
    assert len(kd_ids) == len(kd_set), f"Duplicates in key_dispersed: {len(kd_ids) - len(kd_set)}"
    print(f"    No duplicates: ✓")

    # Differ by permutation
    same_positions = sum(a == b for a, b in zip(kc_ids, kd_ids))
    print(f"    Matching positions: {same_positions}/{len(kc_ids)} ({same_positions/len(kc_ids)*100:.1f}%)")
    assert same_positions < len(kc_ids) * 0.5, f"Too many matching positions ({same_positions}) — streams may be identical"
    print(f"    Differ by permutation: ✓")

    # First-1K set overlap
    kc_first1k = set(kc_ids[:1000])
    kd_first1k = set(kd_ids[:1000])
    overlap = len(kc_first1k & kd_first1k)
    print(f"    First-1K overlap: {overlap}/1000 ({overlap/10:.1f}%)")

    return True


def key_alignment_verification(keys, saved_case_ids, key_clustered, key_dispersed):
    """Verify that key_index correctly maps case_ids to key positions."""
    print("\n  KEY ALIGNMENT VERIFICATION")
    print(f"  {'─'*60}")

    key_index = {cid: i for i, cid in enumerate(saved_case_ids)}

    # Check that all stream case_ids are in key_index
    kc_ids = [r["case_id"] for r in key_clustered]
    kd_ids = [r["case_id"] for r in key_dispersed]
    missing_kc = [cid for cid in kc_ids if cid not in key_index]
    missing_kd = [cid for cid in kd_ids if cid not in key_index]
    assert not missing_kc, f"{len(missing_kc)} key_clustered case_ids not in key_index"
    assert not missing_kd, f"{len(missing_kd)} key_dispersed case_ids not in key_index"
    print(f"    All case_ids mapped: ✓")

    # Verify that different orderings produce different key sequences
    kc_first10_indices = [key_index[cid] for cid in kc_ids[:10]]
    kd_first10_indices = [key_index[cid] for cid in kd_ids[:10]]
    print(f"    Key-clustered first 10 key indices: {kc_first10_indices}")
    print(f"    Key-dispersed first 10 key indices: {kd_first10_indices}")
    assert kc_first10_indices != kd_first10_indices, "First 10 indices match — ordering bug!"
    print(f"    Indices differ: ✓")

    # Verify the actual key vectors differ
    kc_batch0_keys = keys[[key_index[cid] for cid in kc_ids[:100]]]
    kd_batch0_keys = keys[[key_index[cid] for cid in kd_ids[:100]]]
    frobenius_diff = np.linalg.norm(kc_batch0_keys - kd_batch0_keys)
    print(f"    Batch-0 key matrix Frobenius diff: {frobenius_diff:.4f}")
    assert frobenius_diff > 0.01, "Batch-0 keys are identical — critical bug!"
    print(f"    Different key matrices: ✓")

    return key_index


def detailed_cosine_statistics(keys, ordering, batch_size, key_index, label):
    """Compute detailed within-batch cosine statistics."""
    n_batches = len(ordering) // batch_size
    batch_means = []
    batch_medians = []
    batch_maxes = []
    all_pairs = []

    for b in range(n_batches):
        batch_records = ordering[b * batch_size: (b + 1) * batch_size]
        indices = [key_index[r["case_id"]] for r in batch_records]
        batch_keys = keys[indices]

        # Normalize
        norms = np.linalg.norm(batch_keys, axis=1, keepdims=True)
        normed = batch_keys / np.maximum(norms, 1e-8)

        # Cosine matrix
        cos_matrix = normed @ normed.T
        n = len(indices)
        mask = np.triu(np.ones((n, n), dtype=bool), k=1)
        pair_cosines = cos_matrix[mask]

        batch_means.append(float(pair_cosines.mean()))
        batch_medians.append(float(np.median(pair_cosines)))
        batch_maxes.append(float(pair_cosines.max()))
        all_pairs.append(pair_cosines)

    all_pairs_flat = np.concatenate(all_pairs)

    stats = {
        "mean": float(np.mean(batch_means)),
        "median": float(np.median(batch_means)),
        "mean_of_medians": float(np.mean(batch_medians)),
        "mean_batch_max": float(np.mean(batch_maxes)),
        "p90": float(np.percentile(all_pairs_flat, 90)),
        "p95": float(np.percentile(all_pairs_flat, 95)),
        "p99": float(np.percentile(all_pairs_flat, 99)),
        "frac_above_0.2": float((all_pairs_flat > 0.2).mean()),
        "frac_above_0.3": float((all_pairs_flat > 0.3).mean()),
        "frac_above_0.5": float((all_pairs_flat > 0.5).mean()),
        "n_pairs_total": len(all_pairs_flat),
    }

    print(f"\n    {label}:")
    print(f"      Mean within-batch cosine:   {stats['mean']:.4f}")
    print(f"      Median within-batch cosine: {stats['mean_of_medians']:.4f}")
    print(f"      Mean batch-maximum cosine:  {stats['mean_batch_max']:.4f}")
    print(f"      90th percentile (all pairs): {stats['p90']:.4f}")
    print(f"      95th percentile (all pairs): {stats['p95']:.4f}")
    print(f"      99th percentile (all pairs): {stats['p99']:.4f}")
    print(f"      Fraction > 0.2: {stats['frac_above_0.2']:.4f} ({int(stats['frac_above_0.2'] * stats['n_pairs_total'])} pairs)")
    print(f"      Fraction > 0.3: {stats['frac_above_0.3']:.4f} ({int(stats['frac_above_0.3'] * stats['n_pairs_total'])} pairs)")
    print(f"      Fraction > 0.5: {stats['frac_above_0.5']:.4f} ({int(stats['frac_above_0.5'] * stats['n_pairs_total'])} pairs)")

    return stats


def future_key_exposure_by_cohort(keys, ordering, batch_size, key_index, cohort_size=1000, lookahead=10):
    """Mean and max future-key exposure per cohort."""
    n_batches = len(ordering) // batch_size
    batches_per_cohort = cohort_size // batch_size
    cohort_exposures = []

    for b in range(n_batches):
        batch_records = ordering[b * batch_size: (b + 1) * batch_size]
        future_start = (b + 1) * batch_size
        future_end = min((b + 1 + lookahead) * batch_size, len(ordering))

        if future_end <= future_start:
            cohort_exposures.append(0.0)
            continue

        future_records = ordering[future_start:future_end]
        idx_curr = [key_index[r["case_id"]] for r in batch_records]
        idx_future = [key_index[r["case_id"]] for r in future_records]

        kc = keys[idx_curr]
        kf = keys[idx_future]
        nc = kc / np.maximum(np.linalg.norm(kc, axis=1, keepdims=True), 1e-8)
        nf = kf / np.maximum(np.linalg.norm(kf, axis=1, keepdims=True), 1e-8)

        cross = nc @ nf.T
        cohort_exposures.append(float(cross.max()))

    # Aggregate by cohort
    n_cohorts = len(ordering) // cohort_size
    cohort_means = []
    for c in range(n_cohorts):
        start_b = c * batches_per_cohort
        end_b = (c + 1) * batches_per_cohort
        cohort_vals = cohort_exposures[start_b:end_b]
        cohort_means.append(float(np.mean(cohort_vals)))

    return cohort_means


def prefix_geometry(keys, ordering, key_index, checkpoints):
    """Prefix cache geometry at each checkpoint."""
    results = {}

    for t in checkpoints:
        prefix_records = ordering[:t]
        indices = [key_index[r["case_id"]] for r in prefix_records]
        prefix_keys = keys[indices]

        # Gram matrix (t×t)
        gram = prefix_keys @ prefix_keys.T
        eigvals = np.linalg.eigvalsh(gram)
        eigvals = np.sort(eigvals)[::-1]
        eigvals = np.maximum(eigvals, 0)
        svs = np.sqrt(eigvals)
        svs_pos = svs[svs > 1e-10]

        if len(svs_pos) > 0:
            probs = svs_pos / svs_pos.sum()
            effective_rank = float(np.exp(-np.sum(probs * np.log(probs + 1e-10))))
            spectral_entropy = float(-np.sum(probs * np.log2(probs + 1e-10)))
            top_sv_share = float(svs_pos[0] / svs_pos.sum())
            condition = float(svs_pos[0] / svs_pos[-1]) if len(svs_pos) >= 2 else float("inf")
        else:
            effective_rank = 0.0
            spectral_entropy = 0.0
            top_sv_share = 1.0
            condition = float("inf")

        results[t] = {
            "effective_rank": round(effective_rank, 2),
            "condition": round(condition, 2) if condition != float("inf") else "inf",
            "spectral_entropy": round(spectral_entropy, 3),
            "max_eigval_share": round(top_sv_share, 6),
            "numerical_rank": int((svs > 1e-5).sum()),
        }

    return results


def cohort_difficulty_balance(ordering, cohort_size=1000):
    """Check relation composition and surface difficulty by cohort."""
    n_cohorts = len(ordering) // cohort_size
    cohorts = []

    for c in range(n_cohorts):
        cohort = ordering[c * cohort_size: (c + 1) * cohort_size]
        relations = [r["requested_rewrite"]["relation_id"] for r in cohort]
        targets = [r["requested_rewrite"]["target_new"]["str"] for r in cohort]
        subjects = [r["requested_rewrite"]["subject"] for r in cohort]

        rel_counts = Counter(relations)
        n_unique_rels = len(rel_counts)
        top_rel_frac = max(rel_counts.values()) / len(relations)
        probs = np.array(list(rel_counts.values())) / len(relations)
        rel_entropy = float(-np.sum(probs * np.log2(probs + 1e-10)))

        cohorts.append({
            "n_unique_relations": n_unique_rels,
            "top_relation_fraction": round(top_rel_frac, 3),
            "relation_entropy": round(rel_entropy, 2),
            "mean_target_len": round(np.mean([len(t.split()) for t in targets]), 2),
            "mean_subject_len": round(np.mean([len(s) for s in subjects]), 1),
        })

    return cohorts


def main():
    parser = argparse.ArgumentParser(
        description="Validate key-clustered orderings and report detailed statistics"
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--keys_path", type=str, required=True)
    parser.add_argument("--stream_dir", type=str, default="results/matched_ordering")
    parser.add_argument("--batch_size", type=int, default=100)
    args = parser.parse_args()

    stream_dir = Path(args.stream_dir)
    if not stream_dir.is_absolute():
        stream_dir = PROJECT_ROOT / args.stream_dir

    keys_path = Path(args.keys_path)
    if not keys_path.is_absolute():
        keys_path = PROJECT_ROOT / args.keys_path

    kc_path = stream_dir / f"key_clustered_seed{args.seed}.json"
    kd_path = stream_dir / f"key_dispersed_seed{args.seed}.json"

    if not kc_path.exists() or not kd_path.exists():
        print(f"ERROR: Stream files not found")
        print(f"  Tried: {kc_path}")
        print(f"  Tried: {kd_path}")
        sys.exit(1)

    print(f"\n{'='*70}")
    print("Key-Ordering Validation & Detailed Statistics")
    print(f"  Seed: {args.seed}")
    print(f"  Keys: {keys_path}")
    print(f"  Streams: {stream_dir}")
    print(f"{'='*70}")

    # Load
    print("\n  Loading data...")
    npz = np.load(keys_path)
    keys = npz["keys"]
    saved_case_ids = npz["case_ids"].tolist()
    key_clustered = load_stream(kc_path)
    key_dispersed = load_stream(kd_path)
    print(f"  Keys: {keys.shape}, Streams: {len(key_clustered)} records each")

    # 1. Structural validation
    structural_validation(key_clustered, key_dispersed)

    # 2. Key alignment verification
    key_index = key_alignment_verification(keys, saved_case_ids, key_clustered, key_dispersed)

    # 3. Detailed cosine statistics
    print(f"\n  DETAILED WITHIN-BATCH COSINE STATISTICS")
    print(f"  {'─'*60}")
    kc_stats = detailed_cosine_statistics(keys, key_clustered, args.batch_size, key_index, "KEY-CLUSTERED")
    kd_stats = detailed_cosine_statistics(keys, key_dispersed, args.batch_size, key_index, "KEY-DISPERSED")

    print(f"\n    RATIOS (clustered / dispersed):")
    print(f"      Mean:       {kc_stats['mean']/kd_stats['mean']:.2f}x")
    print(f"      Median:     {kc_stats['mean_of_medians']/kd_stats['mean_of_medians']:.2f}x")
    print(f"      Batch-max:  {kc_stats['mean_batch_max']/kd_stats['mean_batch_max']:.2f}x")
    print(f"      P90:        {kc_stats['p90']/kd_stats['p90']:.2f}x")
    print(f"      P95:        {kc_stats['p95']/kd_stats['p95']:.2f}x")
    print(f"      P99:        {kc_stats['p99']/kd_stats['p99']:.2f}x")
    print(f"      Frac>0.2:   {kc_stats['frac_above_0.2']/max(kd_stats['frac_above_0.2'], 1e-10):.2f}x")
    print(f"      Frac>0.3:   {kc_stats['frac_above_0.3']/max(kd_stats['frac_above_0.3'], 1e-10):.2f}x")

    # 4. Future-key exposure by cohort
    print(f"\n  FUTURE-KEY EXPOSURE BY COHORT (lookahead=10 batches)")
    print(f"  {'─'*60}")
    kc_future = future_key_exposure_by_cohort(keys, key_clustered, args.batch_size, key_index)
    kd_future = future_key_exposure_by_cohort(keys, key_dispersed, args.batch_size, key_index)

    print(f"    {'Cohort':<12} {'KC Exposure':>12} {'KD Exposure':>12} {'Ratio':>8}")
    for c, (kc_f, kd_f) in enumerate(zip(kc_future, kd_future)):
        ratio = kc_f / max(kd_f, 1e-10)
        print(f"    {c} ({c*1000+1}-{(c+1)*1000}){'':<2} {kc_f:>12.4f} {kd_f:>12.4f} {ratio:>8.2f}x")

    # 5. Prefix geometry
    print(f"\n  PREFIX GEOMETRY AT CHECKPOINTS")
    print(f"  {'─'*60}")
    checkpoints = [1000, 2000, 3000, 4000, 5000]
    kc_prefix = prefix_geometry(keys, key_clustered, key_index, checkpoints)
    kd_prefix = prefix_geometry(keys, key_dispersed, key_index, checkpoints)

    print(f"    {'Edits':<8} {'KC EffRank':>10} {'KD EffRank':>10} {'KC Cond':>10} {'KD Cond':>10} {'KC Entropy':>11} {'KD Entropy':>11}")
    for t in checkpoints:
        kc = kc_prefix[t]
        kd = kd_prefix[t]
        kc_c = f"{kc['condition']}" if kc['condition'] != 'inf' else '∞'
        kd_c = f"{kd['condition']}" if kd['condition'] != 'inf' else '∞'
        print(f"    {t:<8} {kc['effective_rank']:>10.2f} {kd['effective_rank']:>10.2f} "
              f"{kc_c:>10} {kd_c:>10} {kc['spectral_entropy']:>11.3f} {kd['spectral_entropy']:>11.3f}")

    print(f"\n    Max eigval share:")
    print(f"    {'Edits':<8} {'KC':>12} {'KD':>12} {'Ratio':>8}")
    for t in checkpoints:
        kc_s = kc_prefix[t]['max_eigval_share']
        kd_s = kd_prefix[t]['max_eigval_share']
        ratio = kc_s / max(kd_s, 1e-10)
        print(f"    {t:<8} {kc_s:>12.6f} {kd_s:>12.6f} {ratio:>8.2f}x")

    # 6. Cohort difficulty balance
    print(f"\n  COHORT DIFFICULTY BALANCE")
    print(f"  {'─'*60}")
    print(f"\n    Key-clustered:")
    print(f"    {'Cohort':<10} {'Rels':>5} {'TopFrac':>8} {'Entropy':>8} {'TgtLen':>7} {'SubjLen':>8}")
    kc_cohorts = cohort_difficulty_balance(key_clustered)
    for c, ch in enumerate(kc_cohorts):
        print(f"    {c:<10} {ch['n_unique_relations']:>5} {ch['top_relation_fraction']:>8.3f} "
              f"{ch['relation_entropy']:>8.2f} {ch['mean_target_len']:>7.2f} {ch['mean_subject_len']:>8.1f}")

    print(f"\n    Key-dispersed:")
    print(f"    {'Cohort':<10} {'Rels':>5} {'TopFrac':>8} {'Entropy':>8} {'TgtLen':>7} {'SubjLen':>8}")
    kd_cohorts = cohort_difficulty_balance(key_dispersed)
    for c, ch in enumerate(kd_cohorts):
        print(f"    {c:<10} {ch['n_unique_relations']:>5} {ch['top_relation_fraction']:>8.3f} "
              f"{ch['relation_entropy']:>8.2f} {ch['mean_target_len']:>7.2f} {ch['mean_subject_len']:>8.1f}")

    # Max cross-cohort differences
    max_tgt_diff = max(abs(kc_cohorts[i]['mean_target_len'] - kd_cohorts[i]['mean_target_len']) for i in range(len(kc_cohorts)))
    max_subj_diff = max(abs(kc_cohorts[i]['mean_subject_len'] - kd_cohorts[i]['mean_subject_len']) for i in range(len(kc_cohorts)))
    kc_tgt_range = max(c['mean_target_len'] for c in kc_cohorts) - min(c['mean_target_len'] for c in kc_cohorts)
    kd_tgt_range = max(c['mean_target_len'] for c in kd_cohorts) - min(c['mean_target_len'] for c in kd_cohorts)
    print(f"\n    Max target-length diff (same cohort, across streams): {max_tgt_diff:.3f}")
    print(f"    Max subject-length diff (same cohort, across streams): {max_subj_diff:.1f}")
    print(f"    KC target-length range across cohorts: {kc_tgt_range:.3f}")
    print(f"    KD target-length range across cohorts: {kd_tgt_range:.3f}")

    # Final assessment
    print(f"\n  {'='*70}")
    print("  FINAL ASSESSMENT")
    print(f"  {'='*70}")

    # Key-space separation
    cosine_ratio = kc_stats['mean'] / kd_stats['mean']
    tail_ratio = kc_stats['frac_above_0.2'] / max(kd_stats['frac_above_0.2'], 1e-10)
    print(f"    Mean cosine ratio: {cosine_ratio:.2f}x")
    print(f"    Tail (>0.2) ratio: {tail_ratio:.2f}x")

    # Prefix divergence
    prefix_1k_diff = abs(kc_prefix[1000]['effective_rank'] - kd_prefix[1000]['effective_rank'])
    prefix_3k_diff = abs(kc_prefix[3000]['effective_rank'] - kd_prefix[3000]['effective_rank'])
    print(f"    Prefix eff-rank diff at 1K: {prefix_1k_diff:.2f}")
    print(f"    Prefix eff-rank diff at 3K: {prefix_3k_diff:.2f}")
    print(f"    Prefix geometry diverges at intermediate points: {'✓' if prefix_1k_diff > 1.0 else '⚠ weak'}")

    if cosine_ratio > 1.3 and tail_ratio > 1.5:
        print(f"    ✓ Sufficient key-space manipulation for meaningful experiment")
    elif cosine_ratio > 1.1:
        print(f"    ~ Moderate manipulation — experiment may show small effect")
    else:
        print(f"    ⚠ Weak manipulation — unlikely to produce behavioral difference")

    # Save
    out_path = stream_dir / f"validation_report_seed{args.seed}.json"
    report = {
        "seed": args.seed,
        "structural": {"n_records": len(key_clustered), "matching_positions": sum(a == b for a, b in zip([r["case_id"] for r in key_clustered], [r["case_id"] for r in key_dispersed]))},
        "cosine_stats": {"key_clustered": kc_stats, "key_dispersed": kd_stats, "ratios": {"mean": cosine_ratio, "tail_0.2": tail_ratio}},
        "future_exposure": {"key_clustered": kc_future, "key_dispersed": kd_future},
        "prefix_geometry": {"key_clustered": {str(k): v for k, v in kc_prefix.items()}, "key_dispersed": {str(k): v for k, v in kd_prefix.items()}},
        "cohort_balance": {"key_clustered": kc_cohorts, "key_dispersed": kd_cohorts},
    }
    with open(out_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\n  Report saved: {out_path}")
    print(f"  {'='*70}")


if __name__ == "__main__":
    main()

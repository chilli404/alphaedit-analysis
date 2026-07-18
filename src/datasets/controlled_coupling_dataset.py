#!/usr/bin/env python3
"""
Controlled Coupling Dataset Generator for the predictive divergence experiment.

Generates two matched edit streams of equal length with controlled semantic overlap:

  LOW-COUPLING stream:
    Edits selected to minimize subject/relation overlap. Each edit has a unique
    subject not shared with any other edit within a 20-batch window.

  HIGH-COUPLING stream:
    Edits clustered by subject — packs subjects into consecutive batches so the
    same subject appears 3-5 times with different relations or targets. Plus
    synthetic conflicts (same subject+relation, different target) for maximum
    key-space overlap.

Matching constraints:
  - Same total length
  - Same batch size
  - Same MCF source pool
  - Validated by validate_stream_properties()

Output format: list of evaluate.py-compatible records (dicts with "requested_rewrite",
"case_id", etc.).

Usage:
    # As module
    from src.datasets.controlled_coupling_dataset import generate_controlled_streams
    low, high = generate_controlled_streams(data_dir, seed=42, stream_length=5000)

    # CLI
    uv run python src/datasets/controlled_coupling_dataset.py \\
        --seed 42 --stream_length 5000 --batch_size 100 \\
        --output_dir results/controlled_coupling
"""

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple


def load_counterfact(data_dir: Path) -> list:
    """Load the original multi_counterfact.json dataset."""
    cf_path = data_dir / "multi_counterfact.json"
    if not cf_path.exists():
        raise FileNotFoundError(
            f"multi_counterfact.json not found at {cf_path}. "
            "Run the main experiment first to auto-download it."
        )
    with open(cf_path, "r") as f:
        return json.load(f)


def build_indexes(data: list) -> Tuple[Dict, Dict, Dict]:
    """Build lookup indexes for efficient selection."""
    by_subject = defaultdict(list)
    by_relation = defaultdict(list)
    by_subject_relation = defaultdict(list)

    for record in data:
        rw = record["requested_rewrite"]
        subject = rw["subject"]
        relation = rw["relation_id"]
        by_subject[subject].append(record)
        by_relation[relation].append(record)
        by_subject_relation[(subject, relation)].append(record)

    return dict(by_subject), dict(by_relation), dict(by_subject_relation)


def generate_low_coupling_stream(
    data: list,
    by_subject: Dict,
    rng: random.Random,
    stream_length: int,
    batch_size: int,
    window_size: int = 20,
) -> list:
    """
    Generate a low-coupling stream where no subject repeats within
    a window of `window_size` batches.

    Strategy: shuffle records, then greedily select records whose subject
    hasn't appeared in the last window_size*batch_size edits.
    """
    # Shuffle candidate pool
    pool = list(data)
    rng.shuffle(pool)

    stream = []
    recent_subjects = []  # sliding window of recent subjects
    window_edits = window_size * batch_size
    used_ids = set()

    for record in pool:
        if len(stream) >= stream_length:
            break

        subject = record["requested_rewrite"]["subject"]

        # Check if subject appeared in recent window
        if subject in recent_subjects[-window_edits:]:
            continue

        if record["case_id"] in used_ids:
            continue

        stream.append(record)
        recent_subjects.append(subject)
        used_ids.add(record["case_id"])

    # If we didn't get enough, fill with any remaining (relaxing constraint)
    if len(stream) < stream_length:
        remaining = [r for r in pool if r["case_id"] not in used_ids]
        rng.shuffle(remaining)
        for record in remaining:
            if len(stream) >= stream_length:
                break
            stream.append(record)
            used_ids.add(record["case_id"])

    return stream[:stream_length]


def generate_high_coupling_stream(
    data: list,
    by_subject: Dict,
    by_relation: Dict,
    by_subject_relation: Dict,
    rng: random.Random,
    stream_length: int,
    batch_size: int,
    repeats_per_subject: int = 4,
) -> list:
    """
    Generate a high-coupling stream where subjects are packed into
    consecutive batches (3-5 appearances each), plus synthetic conflicts.

    Strategy:
    1. Select subjects with multiple records (>= repeats_per_subject)
    2. For each selected subject, pack all its records into a cluster
    3. Add synthetic conflicts (same subject+relation, swapped target)
    4. Arrange clusters sequentially so related edits are adjacent
    """
    stream = []
    used_ids = set()
    synthetic_id = 300000

    # Find subjects with enough records for clustering
    clusterable_subjects = [
        (subj, recs) for subj, recs in by_subject.items()
        if len(recs) >= 2
    ]
    rng.shuffle(clusterable_subjects)

    for subject, records in clusterable_subjects:
        if len(stream) >= stream_length:
            break

        available = [r for r in records if r["case_id"] not in used_ids]
        if len(available) < 2:
            continue

        # Take up to repeats_per_subject records for this subject
        cluster_records = available[:repeats_per_subject]
        for r in cluster_records:
            stream.append(r)
            used_ids.add(r["case_id"])

        # Add synthetic conflicts: swap targets between pairs
        if len(cluster_records) >= 2:
            for i in range(0, len(cluster_records) - 1, 2):
                if len(stream) >= stream_length:
                    break
                original = cluster_records[i]
                donor = cluster_records[i + 1]

                # Synthesize: same subject as original, same relation, but donor's target
                conflict = json.loads(json.dumps(original))
                conflict["case_id"] = synthetic_id
                conflict["requested_rewrite"]["target_new"] = (
                    donor["requested_rewrite"]["target_new"]
                )
                synthetic_id += 1
                stream.append(conflict)

    # If not enough from clustering, add more same-relation records in bursts
    if len(stream) < stream_length:
        relations = list(by_relation.keys())
        rng.shuffle(relations)
        for relation in relations:
            if len(stream) >= stream_length:
                break
            rel_records = [r for r in by_relation[relation] if r["case_id"] not in used_ids]
            rng.shuffle(rel_records)
            # Add burst of same-relation records
            burst = rel_records[:batch_size]
            for r in burst:
                if len(stream) >= stream_length:
                    break
                stream.append(r)
                used_ids.add(r["case_id"])

    # Final fill if needed (any remaining records)
    if len(stream) < stream_length:
        all_remaining = [r for r in data if r["case_id"] not in used_ids]
        rng.shuffle(all_remaining)
        for r in all_remaining:
            if len(stream) >= stream_length:
                break
            stream.append(r)
            used_ids.add(r["case_id"])

    return stream[:stream_length]


def validate_stream_properties(
    low_stream: list, high_stream: list, batch_size: int = 100
) -> dict:
    """
    Validate matching constraints and compute overlap statistics.

    Returns dict with validation results and coupling metrics.
    """
    results = {
        "low_length": len(low_stream),
        "high_length": len(high_stream),
        "lengths_match": len(low_stream) == len(high_stream),
        "batch_size": batch_size,
        "num_batches": len(low_stream) // batch_size,
    }

    # Subject overlap within each stream
    def compute_overlap_stats(stream: list, label: str) -> dict:
        subjects = [r["requested_rewrite"]["subject"] for r in stream]
        unique_subjects = set(subjects)
        subject_counts = defaultdict(int)
        for s in subjects:
            subject_counts[s] += 1
        repeat_subjects = {s for s, c in subject_counts.items() if c > 1}

        # Within-batch subject repetition
        n_batches = len(stream) // batch_size
        intra_batch_repeats = 0
        inter_batch_subject_overlap = 0
        batch_subjects = []
        for b in range(n_batches):
            batch = stream[b * batch_size:(b + 1) * batch_size]
            batch_subjs = [r["requested_rewrite"]["subject"] for r in batch]
            unique_in_batch = set(batch_subjs)
            intra_batch_repeats += len(batch_subjs) - len(unique_in_batch)
            batch_subjects.append(unique_in_batch)

        # Count subject overlap between consecutive batches
        for i in range(1, len(batch_subjects)):
            inter_batch_subject_overlap += len(
                batch_subjects[i] & batch_subjects[i - 1]
            )

        return {
            f"{label}_total_subjects": len(subjects),
            f"{label}_unique_subjects": len(unique_subjects),
            f"{label}_repeat_subjects": len(repeat_subjects),
            f"{label}_subject_reuse_ratio": len(subjects) / max(len(unique_subjects), 1),
            f"{label}_intra_batch_repeats": intra_batch_repeats,
            f"{label}_inter_batch_overlap": inter_batch_subject_overlap,
        }

    results.update(compute_overlap_stats(low_stream, "low"))
    results.update(compute_overlap_stats(high_stream, "high"))

    # Coupling differential
    results["coupling_differential"] = (
        results["high_subject_reuse_ratio"] - results["low_subject_reuse_ratio"]
    )
    results["valid"] = (
        results["lengths_match"]
        and results["coupling_differential"] > 0.5
    )

    return results


def generate_controlled_streams(
    data_dir: Path,
    seed: int,
    stream_length: int = 5000,
    batch_size: int = 100,
) -> Tuple[list, list]:
    """
    Main generation logic. Returns (low_coupling_stream, high_coupling_stream).

    Both streams are evaluate.py-compatible lists of records.
    """
    rng = random.Random(seed)

    data = load_counterfact(data_dir)
    by_subject, by_relation, by_subject_relation = build_indexes(data)

    print(f"  Source pool: {len(data)} records, "
          f"{len(by_subject)} unique subjects, "
          f"{len(by_relation)} unique relations")

    # Generate streams (using separate RNG forks for independence)
    rng_low = random.Random(rng.randint(0, 2**32))
    rng_high = random.Random(rng.randint(0, 2**32))

    print("  Generating low-coupling stream...")
    low_stream = generate_low_coupling_stream(
        data, by_subject, rng_low, stream_length, batch_size
    )

    print("  Generating high-coupling stream...")
    high_stream = generate_high_coupling_stream(
        data, by_subject, by_relation, by_subject_relation,
        rng_high, stream_length, batch_size
    )

    # Reassign sequential case_ids for evaluate.py compatibility
    for i, record in enumerate(low_stream):
        record["case_id"] = i

    for i, record in enumerate(high_stream):
        record["case_id"] = i

    return low_stream, high_stream


def main():
    parser = argparse.ArgumentParser(
        description="Generate controlled coupling streams for capacity experiment"
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--stream_length", type=int, default=5000,
                        help="Number of edits per stream")
    parser.add_argument("--batch_size", type=int, default=100,
                        help="Edits per batch")
    parser.add_argument("--data_dir", type=str, default=None,
                        help="Directory containing multi_counterfact.json")
    parser.add_argument("--output_dir", type=str, default="results/controlled_coupling",
                        help="Output directory for generated streams")
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parent.parent.parent
    if args.data_dir:
        data_dir = Path(args.data_dir)
    else:
        data_dir = project_root / "vendor" / "AlphaEdit" / "data"

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("Controlled Coupling Dataset Generator")
    print(f"  Seed:          {args.seed}")
    print(f"  Stream length: {args.stream_length}")
    print(f"  Batch size:    {args.batch_size}")
    print(f"  Data dir:      {data_dir}")
    print(f"  Output dir:    {output_dir}")
    print("=" * 70)

    low_stream, high_stream = generate_controlled_streams(
        data_dir=data_dir,
        seed=args.seed,
        stream_length=args.stream_length,
        batch_size=args.batch_size,
    )

    # Validate
    print("\nValidating stream properties...")
    props = validate_stream_properties(low_stream, high_stream, args.batch_size)
    print(f"  Low stream:  {props['low_unique_subjects']} unique subjects, "
          f"reuse ratio = {props['low_subject_reuse_ratio']:.2f}")
    print(f"  High stream: {props['high_unique_subjects']} unique subjects, "
          f"reuse ratio = {props['high_subject_reuse_ratio']:.2f}")
    print(f"  Coupling differential: {props['coupling_differential']:.2f}")
    print(f"  Inter-batch overlap (low):  {props['low_inter_batch_overlap']}")
    print(f"  Inter-batch overlap (high): {props['high_inter_batch_overlap']}")
    print(f"  Valid: {props['valid']}")

    # Save streams
    low_path = output_dir / f"low_coupling_seed{args.seed}.json"
    high_path = output_dir / f"high_coupling_seed{args.seed}.json"
    props_path = output_dir / f"stream_properties_seed{args.seed}.json"

    with open(low_path, "w") as f:
        json.dump(low_stream, f)
    with open(high_path, "w") as f:
        json.dump(high_stream, f)
    with open(props_path, "w") as f:
        json.dump(props, f, indent=2)

    print(f"\n  Low stream:  {low_path} ({len(low_stream)} records)")
    print(f"  High stream: {high_path} ({len(high_stream)} records)")
    print(f"  Properties:  {props_path}")
    print("\n=== Done ===")


if __name__ == "__main__":
    main()

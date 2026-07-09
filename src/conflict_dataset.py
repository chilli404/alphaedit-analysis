#!/usr/bin/env python3
"""
Conflict sequence dataset generator for AlphaEdit stress testing.

Generates ordered edit pairs from MultiCounterFact where the same subject
is edited with conflicting object values sequentially. This tests whether
AlphaEdit's null-space constraint correctly handles superseding edits.

The test evaluates:
1. Does the model adopt the LATEST edit value? (recency fidelity)
2. Does the earlier edit still partially interfere? (conflict residue)
3. How does AlphaEdit compare to MEMIT under conflicting sequences?

Output: A modified multi_counterfact.json placed in the AlphaEdit data dir,
formatted identically to the original so evaluate.py can consume it directly.
"""

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path


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


def find_conflict_pairs(data: list, max_pairs: int = 100) -> list:
    """
    Find subjects that appear multiple times in the dataset with different
    target objects. These form natural conflict pairs.

    If not enough natural conflicts exist, synthesize them by swapping
    target objects between records that share the same relation.
    """
    # Group by (subject, relation) to find natural conflicts
    subject_relation_groups = defaultdict(list)
    for record in data:
        rw = record["requested_rewrite"]
        key = (rw["subject"], rw["relation_id"])
        subject_relation_groups[key].append(record)

    # Collect natural conflicts (same subject+relation, different target)
    natural_conflicts = []
    for key, records in subject_relation_groups.items():
        if len(records) >= 2:
            for i in range(len(records) - 1):
                natural_conflicts.append((records[i], records[i + 1]))

    # If we have enough natural conflicts, use them
    if len(natural_conflicts) >= max_pairs:
        return natural_conflicts[:max_pairs]

    # Otherwise, synthesize conflicts by pairing records with same relation
    # but different subjects, then swapping target_true for the second record
    relation_groups = defaultdict(list)
    for record in data:
        rw = record["requested_rewrite"]
        relation_groups[rw["relation_id"]].append(record)

    synthetic_conflicts = list(natural_conflicts)
    for relation_id, records in relation_groups.items():
        if len(synthetic_conflicts) >= max_pairs:
            break
        if len(records) < 2:
            continue

        # Create pairs where we'll use the same subject but change the target
        for i in range(0, len(records) - 1, 2):
            if len(synthetic_conflicts) >= max_pairs:
                break

            original = records[i]
            donor = records[i + 1]

            # Create a synthetic conflicting edit for the same subject
            # by using a different target_new value
            conflict_record = json.loads(json.dumps(original))  # deep copy
            conflict_record["case_id"] = 100000 + len(synthetic_conflicts)
            conflict_record["requested_rewrite"]["target_new"] = (
                donor["requested_rewrite"]["target_new"]
            )

            synthetic_conflicts.append((original, conflict_record))

    return synthetic_conflicts[:max_pairs]


def build_conflict_sequence(pairs: list, seed: int) -> list:
    """
    Build a sequential dataset where conflicting edits are interleaved.

    For each pair (A, B) where A and B target the same subject with
    different objects:
    - First apply edit A (original target)
    - Then apply edit B (superseding target)
    - The evaluation should show that B's target is active

    The output is a flat list of records in application order,
    with metadata fields added to track conflict relationships.
    """
    rng = random.Random(seed)

    sequence = []
    for pair_idx, (first, second) in enumerate(pairs):
        # Add conflict metadata to track relationships
        first_entry = json.loads(json.dumps(first))
        second_entry = json.loads(json.dumps(second))

        # Assign sequential case_ids
        base_id = pair_idx * 2
        first_entry["case_id"] = base_id
        second_entry["case_id"] = base_id + 1

        # Add conflict tracking (will be preserved in output JSONs)
        first_entry["conflict_metadata"] = {
            "pair_id": pair_idx,
            "position": "first",
            "superseded_by": base_id + 1,
        }
        second_entry["conflict_metadata"] = {
            "pair_id": pair_idx,
            "position": "second",
            "supersedes": base_id,
        }

        sequence.append(first_entry)
        sequence.append(second_entry)

    return sequence


def main():
    parser = argparse.ArgumentParser(
        description="Generate conflict sequence dataset for AlphaEdit stress test"
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed for pair selection"
    )
    parser.add_argument(
        "--max_pairs",
        type=int,
        default=100,
        help="Maximum number of conflict pairs (total records = 2 * max_pairs)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Output directory (typically vendor/AlphaEdit/data)",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    data_dir = output_dir

    print(f"=== Conflict Dataset Generator ===")
    print(f"  Seed: {args.seed}")
    print(f"  Max pairs: {args.max_pairs}")
    print(f"  Output: {output_dir}")

    # Load original dataset
    print("\nLoading multi_counterfact.json...")
    data = load_counterfact(data_dir)
    print(f"  Loaded {len(data)} records")

    # Find conflict pairs
    print("\nFinding conflict pairs...")
    pairs = find_conflict_pairs(data, max_pairs=args.max_pairs)
    print(f"  Found {len(pairs)} pairs ({len(pairs) * 2} total edits)")

    # Build sequential conflict dataset
    print("\nBuilding conflict sequence...")
    sequence = build_conflict_sequence(pairs, seed=args.seed)

    # Save as conflict_counterfact.json (separate from main dataset)
    output_path = output_dir / "conflict_counterfact.json"
    with open(output_path, "w") as f:
        json.dump(sequence, f, indent=1)

    print(f"\n  Saved to: {output_path}")
    print(f"  Total records: {len(sequence)}")
    print(f"  Conflict pairs: {len(pairs)}")

    # Print sample
    if sequence:
        sample = sequence[0]
        rw = sample["requested_rewrite"]
        print(f"\n  Sample edit:")
        print(f"    Subject: {rw['subject']}")
        print(f"    Prompt:  {rw['prompt'].format(rw['subject'])}")
        print(f"    Target:  {rw['target_new']['str']}")

    print("\n=== Done ===")


if __name__ == "__main__":
    main()

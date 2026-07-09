#!/usr/bin/env python3
"""
Coupling dataset generator for semantic coupling stress test.

Generates edit sequences with labeled coupling types to test whether
AlphaEdit's null-space projection removes more of the update direction
when edits are semantically related to previously-preserved knowledge.

Coupling type taxonomy:
  Type 0 (UNRELATED):       Different subject, different relation
  Type 1 (RELATION_MATCH):  Same relation, different subject
  Type 2 (SUBJECT_MATCH):   Same subject, different relation
  Type 3 (FULL_CONFLICT):   Same subject + same relation, different target

Each coupling type contributes anchor-probe pairs. The anchor establishes
preserved knowledge; the probe (applied later) tests how much of its
update direction is removed by the null-space projection.

Output: JSON compatible with evaluate.py's dataset format, with added
coupling_metadata per record.
"""

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple


COUPLING_TYPES = {
    0: "unrelated",
    1: "relation_match",
    2: "subject_match",
    3: "full_conflict",
}


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
    """Build lookup indexes for efficient pair selection."""
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


def select_unrelated_pairs(
    data: list, max_pairs: int, rng: random.Random, used_ids: set
) -> List[Tuple[dict, dict]]:
    """Select pairs where subject AND relation both differ (Type 0)."""
    available = [r for r in data if r["case_id"] not in used_ids]
    rng.shuffle(available)

    pairs = []
    i = 0
    while len(pairs) < max_pairs and i + 1 < len(available):
        a = available[i]
        b = available[i + 1]
        rw_a = a["requested_rewrite"]
        rw_b = b["requested_rewrite"]
        if rw_a["subject"] != rw_b["subject"] and rw_a["relation_id"] != rw_b["relation_id"]:
            pairs.append((a, b))
            used_ids.add(a["case_id"])
            used_ids.add(b["case_id"])
        i += 2

    return pairs


def select_relation_match_pairs(
    by_relation: Dict, max_pairs: int, rng: random.Random, used_ids: set
) -> List[Tuple[dict, dict]]:
    """Select pairs sharing relation but with different subjects (Type 1)."""
    pairs = []
    relations = list(by_relation.keys())
    rng.shuffle(relations)

    for relation in relations:
        if len(pairs) >= max_pairs:
            break
        records = [r for r in by_relation[relation] if r["case_id"] not in used_ids]
        if len(records) < 2:
            continue

        rng.shuffle(records)
        for i in range(0, len(records) - 1, 2):
            if len(pairs) >= max_pairs:
                break
            a, b = records[i], records[i + 1]
            if a["requested_rewrite"]["subject"] != b["requested_rewrite"]["subject"]:
                pairs.append((a, b))
                used_ids.add(a["case_id"])
                used_ids.add(b["case_id"])

    return pairs


def select_subject_match_pairs(
    by_subject: Dict, max_pairs: int, rng: random.Random, used_ids: set
) -> List[Tuple[dict, dict]]:
    """Select pairs sharing subject but with different relations (Type 2)."""
    pairs = []
    subjects = [s for s, recs in by_subject.items() if len(recs) >= 2]
    rng.shuffle(subjects)

    for subject in subjects:
        if len(pairs) >= max_pairs:
            break
        records = [r for r in by_subject[subject] if r["case_id"] not in used_ids]
        if len(records) < 2:
            continue

        rng.shuffle(records)
        for i in range(0, len(records) - 1, 2):
            if len(pairs) >= max_pairs:
                break
            a, b = records[i], records[i + 1]
            if a["requested_rewrite"]["relation_id"] != b["requested_rewrite"]["relation_id"]:
                pairs.append((a, b))
                used_ids.add(a["case_id"])
                used_ids.add(b["case_id"])

    return pairs


def select_full_conflict_pairs(
    by_subject_relation: Dict, by_relation: Dict,
    max_pairs: int, rng: random.Random, used_ids: set
) -> List[Tuple[dict, dict]]:
    """
    Select pairs with same subject + same relation but different targets (Type 3).
    Synthesize if not enough natural conflicts exist.
    """
    pairs = []

    # First try natural conflicts
    keys = list(by_subject_relation.keys())
    rng.shuffle(keys)
    for key in keys:
        if len(pairs) >= max_pairs:
            break
        records = [r for r in by_subject_relation[key] if r["case_id"] not in used_ids]
        if len(records) >= 2:
            # Different target values
            for i in range(0, len(records) - 1, 2):
                if len(pairs) >= max_pairs:
                    break
                a, b = records[i], records[i + 1]
                if (a["requested_rewrite"]["target_new"]["str"]
                        != b["requested_rewrite"]["target_new"]["str"]):
                    pairs.append((a, b))
                    used_ids.add(a["case_id"])
                    used_ids.add(b["case_id"])

    # Synthesize if needed (swap targets between same-relation records)
    if len(pairs) < max_pairs:
        relations = list(by_relation.keys())
        rng.shuffle(relations)
        synthetic_id_counter = 200000

        for relation in relations:
            if len(pairs) >= max_pairs:
                break
            records = [r for r in by_relation[relation] if r["case_id"] not in used_ids]
            if len(records) < 2:
                continue

            rng.shuffle(records)
            for i in range(0, len(records) - 1, 2):
                if len(pairs) >= max_pairs:
                    break

                original = records[i]
                donor = records[i + 1]

                # Synthesize: same subject as original, but donor's target
                conflict = json.loads(json.dumps(original))
                conflict["case_id"] = synthetic_id_counter
                conflict["requested_rewrite"]["target_new"] = (
                    donor["requested_rewrite"]["target_new"]
                )
                synthetic_id_counter += 1

                pairs.append((original, conflict))
                used_ids.add(original["case_id"])
                used_ids.add(conflict["case_id"])

    return pairs


def build_coupling_sequence(
    type_pairs: Dict[int, List[Tuple[dict, dict]]],
    warmup_data: list,
    warmup_count: int,
    rng: random.Random,
) -> list:
    """
    Build the final edit sequence with coupling metadata.

    Structure:
      [warmup edits] + [interleaved anchor-probe pairs from all types]

    Within the non-warmup section, pairs from all types are shuffled
    together. Each pair is: anchor (first), then probe (second), with
    at least 1 other edit between anchor and probe.
    """
    sequence = []
    case_id_counter = 0

    # Warmup: unrelated edits to build up cache_c state
    rng.shuffle(warmup_data)
    for record in warmup_data[:warmup_count]:
        entry = json.loads(json.dumps(record))
        entry["case_id"] = case_id_counter
        entry["coupling_metadata"] = {
            "coupling_type": -1,
            "coupling_type_name": "warmup",
            "role": "warmup",
            "pair_id": None,
            "anchor_case_id": None,
        }
        sequence.append(entry)
        case_id_counter += 1

    # Collect all pairs with their type labels
    all_pairs = []
    for coupling_type, pairs in type_pairs.items():
        for pair_idx, (anchor_record, probe_record) in enumerate(pairs):
            all_pairs.append((coupling_type, pair_idx, anchor_record, probe_record))

    rng.shuffle(all_pairs)

    # Build interleaved sequence: anchor then probe for each pair
    # Pairs are shuffled but within each pair, anchor always precedes probe
    for coupling_type, pair_idx, anchor_record, probe_record in all_pairs:
        anchor_id = case_id_counter
        probe_id = case_id_counter + 1

        anchor_entry = json.loads(json.dumps(anchor_record))
        anchor_entry["case_id"] = anchor_id
        anchor_entry["coupling_metadata"] = {
            "coupling_type": coupling_type,
            "coupling_type_name": COUPLING_TYPES[coupling_type],
            "role": "anchor",
            "pair_id": f"{coupling_type}_{pair_idx}",
            "anchor_case_id": None,
        }

        probe_entry = json.loads(json.dumps(probe_record))
        probe_entry["case_id"] = probe_id
        probe_entry["coupling_metadata"] = {
            "coupling_type": coupling_type,
            "coupling_type_name": COUPLING_TYPES[coupling_type],
            "role": "probe",
            "pair_id": f"{coupling_type}_{pair_idx}",
            "anchor_case_id": anchor_id,
        }

        sequence.append(anchor_entry)
        sequence.append(probe_entry)
        case_id_counter += 2

    return sequence


def generate_coupling_dataset(
    data_dir: Path,
    seed: int,
    max_pairs_per_type: int,
    warmup_count: int,
) -> list:
    """Main generation logic. Returns the coupling dataset as a list of records."""
    rng = random.Random(seed)

    data = load_counterfact(data_dir)
    by_subject, by_relation, by_subject_relation = build_indexes(data)

    used_ids: set = set()

    # Select pairs for each type
    type_pairs = {}

    # Type 3 first (hardest to find, most constrained)
    type_pairs[3] = select_full_conflict_pairs(
        by_subject_relation, by_relation, max_pairs_per_type, rng, used_ids
    )

    # Type 2 (same subject, different relation)
    type_pairs[2] = select_subject_match_pairs(
        by_subject, max_pairs_per_type, rng, used_ids
    )

    # Type 1 (same relation, different subject)
    type_pairs[1] = select_relation_match_pairs(
        by_relation, max_pairs_per_type, rng, used_ids
    )

    # Type 0 (unrelated)
    type_pairs[0] = select_unrelated_pairs(
        data, max_pairs_per_type, rng, used_ids
    )

    # Warmup edits from remaining unused records
    warmup_pool = [r for r in data if r["case_id"] not in used_ids]

    # Build sequence
    sequence = build_coupling_sequence(type_pairs, warmup_pool, warmup_count, rng)

    return sequence


def main():
    parser = argparse.ArgumentParser(
        description="Generate coupling-labeled dataset for semantic coupling stress test"
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--max_pairs_per_type", type=int, default=60,
        help="Maximum anchor-probe pairs per coupling type"
    )
    parser.add_argument(
        "--warmup_count", type=int, default=20,
        help="Number of unrelated warmup edits at start"
    )
    parser.add_argument(
        "--data_dir", type=str, default=None,
        help="Directory containing multi_counterfact.json (default: vendor/AlphaEdit/data)"
    )
    parser.add_argument(
        "--output_path", type=str, required=True,
        help="Output JSON file path"
    )
    args = parser.parse_args()

    # Resolve data directory
    project_root = Path(__file__).resolve().parent.parent
    if args.data_dir:
        data_dir = Path(args.data_dir)
    else:
        data_dir = project_root / "vendor" / "AlphaEdit" / "data"

    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print("=== Coupling Dataset Generator ===")
    print(f"  Seed: {args.seed}")
    print(f"  Max pairs/type: {args.max_pairs_per_type}")
    print(f"  Warmup edits: {args.warmup_count}")
    print(f"  Data dir: {data_dir}")
    print(f"  Output: {output_path}")

    sequence = generate_coupling_dataset(
        data_dir=data_dir,
        seed=args.seed,
        max_pairs_per_type=args.max_pairs_per_type,
        warmup_count=args.warmup_count,
    )

    # Summary
    type_counts = defaultdict(int)
    role_counts = defaultdict(int)
    for record in sequence:
        meta = record["coupling_metadata"]
        type_counts[meta["coupling_type_name"]] += 1
        role_counts[meta["role"]] += 1

    print(f"\n  Total records: {len(sequence)}")
    print(f"  By type: {dict(type_counts)}")
    print(f"  By role: {dict(role_counts)}")

    # Save
    with open(output_path, "w") as f:
        json.dump(sequence, f, indent=1)

    print(f"\n  Saved to: {output_path}")
    print("=== Done ===")


if __name__ == "__main__":
    main()

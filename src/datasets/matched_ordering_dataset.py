#!/usr/bin/env python3
"""
Matched Ordering Dataset Generator for clean coupling experiment.

Takes the SAME 5000 unique, non-conflicting MCF records and produces
two orderings that differ ONLY in temporal arrangement:

  CLUSTERED ordering:
    Records grouped by relation_id so that batches contain edits with
    similar prompt templates (→ similar key directions).

  DISPERSED ordering:
    Records with the same relation_id spread maximally across batches
    (round-robin assignment), minimizing within-batch key similarity.

Guarantees:
  - Identical facts, subjects, targets, base difficulty
  - Zero duplicates
  - Zero target conflicts (each subject appears exactly once)
  - Same total independent constraints
  - Only temporal/geometric arrangement differs

The causal claim this supports:
  "Temporal concentration of similar edits changes long-horizon retention."

Usage:
    uv run python src/datasets/matched_ordering_dataset.py \
        --seed 42 --stream_length 5000 --batch_size 100 \
        --output_dir results/matched_ordering

    # Then run:
    #   AlphaEdit on clustered ordering
    #   AlphaEdit on dispersed ordering
    #   MEMIT-seq on clustered ordering
    #   MEMIT-seq on dispersed ordering
"""

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path
from typing import List, Tuple


def load_counterfact(data_dir: Path) -> list:
    """Load the original multi_counterfact.json dataset."""
    cf_path = data_dir / "multi_counterfact.json"
    if not cf_path.exists():
        raise FileNotFoundError(
            f"multi_counterfact.json not found at {cf_path}. "
            "Run scripts/link_dsets.sh or download_datasets.sh first."
        )
    with open(cf_path, "r") as f:
        return json.load(f)


def select_clean_pool(
    data: list,
    rng: random.Random,
    stream_length: int,
) -> list:
    """
    Select unique, non-conflicting records from MCF.

    Rules:
      - Each subject appears exactly once
      - No duplicate (subject, target) pairs
      - No conflicting targets for same subject
      - Records chosen randomly from the full pool
    """
    # Shuffle to avoid positional bias
    pool = list(data)
    rng.shuffle(pool)

    selected = []
    used_subjects = set()
    used_case_ids = set()

    for record in pool:
        if len(selected) >= stream_length:
            break

        rw = record["requested_rewrite"]
        subject = rw["subject"]
        case_id = record["case_id"]

        # Each subject exactly once
        if subject in used_subjects:
            continue
        if case_id in used_case_ids:
            continue

        selected.append(record)
        used_subjects.add(subject)
        used_case_ids.add(case_id)

    if len(selected) < stream_length:
        print(f"  WARNING: Only found {len(selected)} unique-subject records "
              f"(requested {stream_length})")

    return selected


def create_clustered_ordering(
    records: list,
    rng: random.Random,
    batch_size: int,
) -> list:
    """
    Arrange records so that same-relation edits are grouped in consecutive batches.

    Strategy: sort by relation_id, then shuffle within each relation group,
    then arrange groups contiguously. This maximizes within-batch key similarity
    because records with the same relation template produce similar key vectors.
    """
    # Group by relation
    by_relation = defaultdict(list)
    for record in records:
        rel = record["requested_rewrite"]["relation_id"]
        by_relation[rel].append(record)

    # Shuffle within each relation group (for randomness within clusters)
    for rel in by_relation:
        rng.shuffle(by_relation[rel])

    # Sort relations by group size (largest first → maximizes cluster density)
    sorted_relations = sorted(by_relation.keys(), key=lambda r: len(by_relation[r]), reverse=True)

    # Pack groups contiguously
    clustered = []
    for rel in sorted_relations:
        clustered.extend(by_relation[rel])

    return clustered


def create_dispersed_ordering(
    records: list,
    rng: random.Random,
    batch_size: int,
) -> list:
    """
    Arrange records so that same-relation edits are spread maximally across batches.

    Strategy: round-robin assignment from each relation group into sequential
    positions. This minimizes within-batch key similarity by ensuring each
    batch contains edits from as many different relations as possible.
    """
    # Group by relation
    by_relation = defaultdict(list)
    for record in records:
        rel = record["requested_rewrite"]["relation_id"]
        by_relation[rel].append(record)

    # Shuffle within each relation group
    for rel in by_relation:
        rng.shuffle(by_relation[rel])

    # Round-robin: interleave records from different relations
    # Shuffle the relation order for randomness
    relation_ids = list(by_relation.keys())
    rng.shuffle(relation_ids)

    # Build queues for each relation
    queues = {rel: list(by_relation[rel]) for rel in relation_ids}

    dispersed = []
    while len(dispersed) < len(records):
        # One pass through all relations with remaining records
        added_this_round = False
        for rel in relation_ids:
            if queues[rel]:
                dispersed.append(queues[rel].pop(0))
                added_this_round = True
                if len(dispersed) >= len(records):
                    break
        if not added_this_round:
            break

    return dispersed


def validate_orderings(
    clean_pool: list,
    clustered: list,
    dispersed: list,
    batch_size: int,
) -> dict:
    """Validate that orderings are correct and compute concentration metrics."""
    # Same records
    assert len(clustered) == len(dispersed) == len(clean_pool)
    assert set(r["case_id"] for r in clustered) == set(r["case_id"] for r in clean_pool)
    assert set(r["case_id"] for r in dispersed) == set(r["case_id"] for r in clean_pool)

    # No duplicates
    assert len(set(r["case_id"] for r in clustered)) == len(clustered)
    assert len(set(r["case_id"] for r in dispersed)) == len(dispersed)

    # No subject conflicts
    for ordering in [clustered, dispersed]:
        subjects = [r["requested_rewrite"]["subject"] for r in ordering]
        assert len(subjects) == len(set(subjects)), "Duplicate subjects found!"

    # Compute within-batch relation concentration
    def batch_concentration(ordering):
        """Fraction of within-batch pairs sharing a relation."""
        n_batches = len(ordering) // batch_size
        concentrations = []
        for b in range(n_batches):
            batch = ordering[b * batch_size: (b + 1) * batch_size]
            relations = [r["requested_rewrite"]["relation_id"] for r in batch]
            n = len(relations)
            if n < 2:
                continue
            # Count pairs sharing same relation
            from collections import Counter
            counts = Counter(relations)
            same_pairs = sum(c * (c - 1) for c in counts.values())
            total_pairs = n * (n - 1)
            concentrations.append(same_pairs / total_pairs if total_pairs > 0 else 0)
        return concentrations

    clust_conc = batch_concentration(clustered)
    disp_conc = batch_concentration(dispersed)

    # Relation diversity per batch
    def batch_diversity(ordering):
        n_batches = len(ordering) // batch_size
        diversities = []
        for b in range(n_batches):
            batch = ordering[b * batch_size: (b + 1) * batch_size]
            relations = set(r["requested_rewrite"]["relation_id"] for r in batch)
            diversities.append(len(relations))
        return diversities

    clust_div = batch_diversity(clustered)
    disp_div = batch_diversity(dispersed)

    import numpy as np
    props = {
        "n_records": len(clean_pool),
        "n_unique_subjects": len(set(r["requested_rewrite"]["subject"] for r in clean_pool)),
        "n_unique_relations": len(set(r["requested_rewrite"]["relation_id"] for r in clean_pool)),
        "batch_size": batch_size,
        "n_batches": len(clean_pool) // batch_size,
        "clustered": {
            "mean_batch_concentration": float(np.mean(clust_conc)),
            "max_batch_concentration": float(np.max(clust_conc)),
            "mean_relations_per_batch": float(np.mean(clust_div)),
            "min_relations_per_batch": int(np.min(clust_div)),
        },
        "dispersed": {
            "mean_batch_concentration": float(np.mean(disp_conc)),
            "max_batch_concentration": float(np.max(disp_conc)),
            "mean_relations_per_batch": float(np.mean(disp_div)),
            "min_relations_per_batch": int(np.min(disp_div)),
        },
        "concentration_ratio": float(np.mean(clust_conc) / max(np.mean(disp_conc), 1e-10)),
    }
    return props


def main():
    parser = argparse.ArgumentParser(
        description="Generate matched clustered/dispersed orderings from clean MCF pool"
    )
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--stream_length", type=int, default=5000)
    parser.add_argument("--batch_size", type=int, default=100)
    parser.add_argument("--data_dir", type=str, default=None,
                        help="Directory containing multi_counterfact.json")
    parser.add_argument("--output_dir", type=str, default="results/matched_ordering")
    args = parser.parse_args()

    # Resolve data directory
    project_root = Path(__file__).resolve().parent.parent.parent
    if args.data_dir:
        data_dir = Path(args.data_dir)
    else:
        candidates = [
            project_root / "vendor" / "AlphaEdit" / "data",
            Path("/s3-data/continual-learning/alphaedit/dsets"),
            Path.home() / "Projects" / "alphaedit-analysis" / "vendor" / "AlphaEdit" / "data",
        ]
        data_dir = None
        for c in candidates:
            if (c / "multi_counterfact.json").exists():
                data_dir = c
                break
        if data_dir is None:
            print("ERROR: Cannot find multi_counterfact.json")
            print("  Tried:", [str(c) for c in candidates])
            sys.exit(1)

    import sys
    import numpy as np

    rng = random.Random(args.seed)

    print(f"\n{'='*70}")
    print("Matched Ordering Dataset Generator")
    print(f"  Seed:          {args.seed}")
    print(f"  Stream length: {args.stream_length}")
    print(f"  Batch size:    {args.batch_size}")
    print(f"  Data dir:      {data_dir}")
    print(f"{'='*70}")

    # Load MCF
    print("\n  Loading MultiCounterFact...")
    data = load_counterfact(data_dir)
    print(f"  Total MCF records: {len(data)}")

    # Select clean pool (unique subjects, no conflicts)
    print(f"\n  Selecting {args.stream_length} unique, non-conflicting records...")
    clean_pool = select_clean_pool(data, rng, args.stream_length)
    print(f"  Selected: {len(clean_pool)} records")

    n_subjects = len(set(r["requested_rewrite"]["subject"] for r in clean_pool))
    n_relations = len(set(r["requested_rewrite"]["relation_id"] for r in clean_pool))
    print(f"  Unique subjects: {n_subjects}")
    print(f"  Unique relations: {n_relations}")

    # Generate orderings (use separate RNG streams for independence)
    rng_clust = random.Random(args.seed + 1000)
    rng_disp = random.Random(args.seed + 2000)

    print("\n  Generating clustered ordering...")
    clustered = create_clustered_ordering(clean_pool, rng_clust, args.batch_size)

    print("  Generating dispersed ordering...")
    dispersed = create_dispersed_ordering(clean_pool, rng_disp, args.batch_size)

    # Validate
    print("\n  Validating orderings...")
    props = validate_orderings(clean_pool, clustered, dispersed, args.batch_size)

    print(f"\n  Concentration metrics:")
    print(f"    Clustered: {props['clustered']['mean_batch_concentration']:.4f} mean pair-sharing "
          f"({props['clustered']['mean_relations_per_batch']:.1f} relations/batch)")
    print(f"    Dispersed: {props['dispersed']['mean_batch_concentration']:.4f} mean pair-sharing "
          f"({props['dispersed']['mean_relations_per_batch']:.1f} relations/batch)")
    print(f"    Concentration ratio: {props['concentration_ratio']:.1f}x")

    # Save outputs
    out_dir = Path(args.output_dir)
    if not out_dir.is_absolute():
        out_dir = project_root / args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    clust_path = out_dir / f"clustered_seed{args.seed}.json"
    disp_path = out_dir / f"dispersed_seed{args.seed}.json"
    props_path = out_dir / f"stream_properties_seed{args.seed}.json"

    with open(clust_path, "w") as f:
        json.dump(clustered, f)
    with open(disp_path, "w") as f:
        json.dump(dispersed, f)
    with open(props_path, "w") as f:
        json.dump(props, f, indent=2)

    print(f"\n  Output:")
    print(f"    Clustered: {clust_path}")
    print(f"    Dispersed: {disp_path}")
    print(f"    Properties: {props_path}")
    print(f"\n{'='*70}")
    print("Done. Run experiments with:")
    print(f"  bash scripts/run_memit_seq_coupling.sh {args.seed} clustered")
    print(f"  bash scripts/run_memit_seq_coupling.sh {args.seed} dispersed")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()

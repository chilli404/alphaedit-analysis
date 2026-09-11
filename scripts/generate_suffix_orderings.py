#!/usr/bin/env python3
"""Generate suffix-switch orderings for the pre-collapse rescue experiment.

Takes the fb_high_exposure ordering, keeps the first PREFIX_BATCHES batches
identical, and reorders the remaining batches using:
  - suffix_random:  random permutation of remaining batches
  - suffix_spread:  farthest-neighbor greedy (minimize temporal concentration)

Also generates late-branch variants (branch at batch 69 = 7K edits).

Usage:
    uv run python scripts/generate_suffix_orderings.py --seed 42
    uv run python scripts/generate_suffix_orderings.py --seed 42 --seed 2024 --seed 137
"""

import argparse
import json
import os
import random
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src" / "datasets"))

from generate_orderings import (
    compute_batch_centroids,
    order_batches_low_exposure,
)


BATCH_SIZE = 100


def load_keys(keys_path: Path):
    npz = np.load(keys_path)
    all_keys = npz["keys"]
    all_case_ids = npz["case_ids"].tolist()
    return all_keys, {int(cid): i for i, cid in enumerate(all_case_ids)}


def generate_suffix_orderings(
    stream: list,
    keys: np.ndarray,
    case_id_to_idx: dict,
    seed: int,
    branch_batch: int,
):
    """Generate suffix-reordered streams branching at branch_batch.

    Returns dict of {name: reordered_stream}.
    """
    split = (branch_batch + 1) * BATCH_SIZE
    prefix_records = stream[:split]
    suffix_records = stream[split:]

    # Split suffix into its original batches
    suffix_batches = [
        suffix_records[i : i + BATCH_SIZE]
        for i in range(0, len(suffix_records), BATCH_SIZE)
    ]
    n_suffix = len(suffix_batches)

    # Compute centroids for suffix batches
    centroids = compute_batch_centroids(suffix_batches, keys, case_id_to_idx)

    # Random suffix: shuffle batch order
    rng = random.Random(seed + 9000 + branch_batch)
    perm = list(range(n_suffix))
    rng.shuffle(perm)
    random_suffix = [r for idx in perm for r in suffix_batches[idx]]

    # Spread suffix: farthest-neighbor greedy on suffix batches only
    spread_suffix = order_batches_low_exposure(
        suffix_batches, centroids, random.Random(seed + 9500 + branch_batch)
    )

    tag = "" if branch_batch == 49 else "_late"

    return {
        f"suffix_random{tag}": prefix_records + random_suffix,
        f"suffix_spread{tag}": prefix_records + spread_suffix,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Generate suffix-switch orderings for rescue experiment"
    )
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 2024, 137])
    parser.add_argument(
        "--keys_path",
        type=str,
        default="results/key_vectors/full_mcf/keys_seed42_layer6.npz",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="results/matched_ordering/orderings",
    )
    args = parser.parse_args()

    keys_path = Path(args.keys_path)
    if not keys_path.is_absolute():
        keys_path = PROJECT_ROOT / keys_path
    out_dir = Path(args.output_dir)
    if not out_dir.is_absolute():
        out_dir = PROJECT_ROOT / args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    keys, case_id_to_idx = load_keys(keys_path)
    print(f"Loaded keys: {keys.shape}")

    for seed in args.seeds:
        src_path = out_dir / f"fb_high_exposure_seed{seed}.json"
        if not src_path.exists():
            print(f"  SKIP seed {seed}: {src_path} not found")
            continue

        with open(src_path) as f:
            stream = json.load(f)
        print(f"\nSeed {seed}: {len(stream)} records")

        for branch_batch, label in [(49, "5K branch"), (69, "7K branch")]:
            orderings = generate_suffix_orderings(
                stream, keys, case_id_to_idx, seed, branch_batch
            )
            for name, ordering in orderings.items():
                path = out_dir / f"{name}_seed{seed}.json"
                with open(path, "w") as f:
                    json.dump(ordering, f)
                print(f"  {label}: {path.name} ({len(ordering)} records)")

    print("\nDone. Launch with:")
    print(
        "  sky launch sky/suffix_switch.yaml --cluster ae-suffix-random-s42 "
        "--env SEED=42 --env SUFFIX_TYPE=suffix_random --detach-run -y"
    )


if __name__ == "__main__":
    main()

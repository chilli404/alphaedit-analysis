#!/usr/bin/env python3
"""Generate interference-aware scheduling orderings.

Loads the same 10K-record pool as existing orderings for a given seed,
runs the scheduling algorithms, validates, and saves ordering files
compatible with run_matched_ordering.sh.

Requires:
  - Existing ordering file (to extract record pool):
    results/matched_ordering/orderings/key_clustered_seed{SEED}.json
  - Precomputed key vectors:
    results/key_vectors/full_mcf/keys_seed42_layer6.npz

Usage:
    uv run python scheduling/generate_scheduling_orderings.py --seed 42
    uv run python scheduling/generate_scheduling_orderings.py --seed 2024 --methods greedy_minmax,random
    uv run python scheduling/generate_scheduling_orderings.py --seed 42 --output_dir results/matched_ordering/orderings
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scheduling.interference_scheduler import build_ordering

RESULT_ROOT = Path(os.environ.get("RESULT_ROOT", PROJECT_ROOT / "results"))
ORDERINGS_DIR = RESULT_ROOT / "matched_ordering" / "orderings"
KEYS_PATH = RESULT_ROOT / "key_vectors" / "full_mcf" / "keys_seed42_layer6.npz"

ALL_METHODS = ["greedy_minmax", "cluster_topo", "random"]


def load_existing_record_pool(seed: int) -> list:
    """Load the record pool from an existing ordering file.

    All orderings for a seed share the same record set. We load from
    key_clustered (arbitrary choice) to extract the canonical pool.
    """
    candidates = [
        ORDERINGS_DIR / f"key_clustered_seed{seed}.json",
        ORDERINGS_DIR / f"clustered_seed{seed}.json",
        ORDERINGS_DIR / f"key_dispersed_seed{seed}.json",
    ]
    for path in candidates:
        if path.exists():
            print(f"  Loading record pool from: {path.name}")
            with open(path) as f:
                return json.load(f)

    raise FileNotFoundError(
        f"No existing ordering file found for seed {seed}. "
        f"Tried: {[str(p) for p in candidates]}. "
        f"Run 'uv run python src/datasets/generate_orderings.py --seed {seed}' first."
    )


def load_and_filter_keys(records: list) -> np.ndarray:
    """Load key vectors and filter/reorder to match the record pool.

    Returns:
        keys: (N, D) float32, L2-normalized, in same order as records.
    """
    if not KEYS_PATH.exists():
        raise FileNotFoundError(
            f"Key vectors not found at {KEYS_PATH}. "
            "These are needed for cosine computation."
        )

    print(f"  Loading keys from: {KEYS_PATH.name}")
    npz = np.load(KEYS_PATH)
    all_keys = npz["keys"].astype(np.float32)
    all_case_ids = npz["case_ids"].tolist()

    # Build lookup: case_id -> row index in full key matrix
    key_idx_by_id = {cid: i for i, cid in enumerate(all_case_ids)}

    # Filter to record pool's case_ids, maintaining record order
    record_case_ids = [r["case_id"] for r in records]
    missing = [cid for cid in record_case_ids if cid not in key_idx_by_id]
    if missing:
        raise ValueError(
            f"{len(missing)} records in pool have no precomputed key. "
            f"First missing case_id: {missing[0]}"
        )

    indices = [key_idx_by_id[cid] for cid in record_case_ids]
    keys = all_keys[indices]
    print(f"  Selected keys: shape={keys.shape}")

    # L2 normalize
    norms = np.linalg.norm(keys, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-8)
    keys = keys / norms

    return keys


def main():
    parser = argparse.ArgumentParser(
        description="Generate interference-aware scheduling orderings"
    )
    parser.add_argument("--seed", type=int, required=True,
                        help="Seed (must match existing ordering files)")
    parser.add_argument("--methods", type=str, default=",".join(ALL_METHODS),
                        help="Comma-separated methods to generate (default: all)")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Output directory (default: results/matched_ordering/orderings/)")
    parser.add_argument("--n_clusters", type=int, default=50,
                        help="Number of clusters for cluster_topo method")
    parser.add_argument("--batch_size", type=int, default=100,
                        help="Batch size for scheduling")
    args = parser.parse_args()

    methods = [m.strip() for m in args.methods.split(",")]
    for m in methods:
        if m not in ALL_METHODS:
            print(f"ERROR: Unknown method '{m}'. Valid: {ALL_METHODS}")
            sys.exit(1)

    output_dir = Path(args.output_dir) if args.output_dir else ORDERINGS_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*70}")
    print("Interference-Aware Scheduling — Ordering Generator")
    print(f"  Seed:       {args.seed}")
    print(f"  Methods:    {methods}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Output:     {output_dir}")
    print(f"{'='*70}")

    # ── Step 1: Load record pool ──────────────────────────────────────────────

    print(f"\n  Step 1: Load record pool")
    records = load_existing_record_pool(args.seed)
    N = len(records)
    print(f"  Pool size: {N} records")

    # ── Step 2: Load and filter keys ──────────────────────────────────────────

    print(f"\n  Step 2: Load and normalize keys")
    keys = load_and_filter_keys(records)

    # ── Step 3: Generate orderings ────────────────────────────────────────────

    for method in methods:
        print(f"\n  Step 3: Generating ordering: {method}")
        t0 = time.time()

        perm = build_ordering(
            keys,
            method=method,
            batch_size=args.batch_size,
            seed=args.seed,
            n_clusters=args.n_clusters,
            verbose=True,
        )

        elapsed = time.time() - t0
        print(f"    Completed in {elapsed:.1f}s")

        # Apply permutation to records
        ordered_records = [records[i] for i in perm]

        # Validate: same case_ids, different order
        orig_ids = set(r["case_id"] for r in records)
        new_ids = set(r["case_id"] for r in ordered_records)
        assert orig_ids == new_ids, "Case ID mismatch after permutation!"
        assert len(ordered_records) == N

        # Save
        out_path = output_dir / f"{method}_seed{args.seed}.json"
        with open(out_path, "w") as f:
            json.dump(ordered_records, f)
        size_mb = out_path.stat().st_size / 1e6
        print(f"    Saved: {out_path.name} ({size_mb:.1f} MB)")

        # Quick diagnostic: first-100 max-cos
        cos_first_100 = keys[perm[:100]] @ keys[perm[100:]].T
        max_cos_first_100 = cos_first_100.max(axis=1).mean()
        print(f"    First-100 mean max-cos-to-subsequent: {max_cos_first_100:.4f}")

    # ── Done ──────────────────────────────────────────────────────────────────

    print(f"\n{'='*70}")
    print("Done. Run experiments with:")
    for method in methods:
        print(f"  bash scripts/run_scheduling_experiment.sh {args.seed} AlphaEdit {method}")
    print(f"\nOr validate geometry first:")
    print(f"  uv run python scheduling/validate_ordering.py --seed {args.seed}")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Generate interference-aware scheduling orderings.

Loads the same 10K-record pool as existing orderings for a given seed,
runs the scheduling algorithms, validates, and saves ordering files
compatible with run_matched_ordering.sh.

Requires:
  - Existing ordering file (to extract record pool):
    results/matched_ordering/orderings/key_clustered_seed{SEED}.json
  - Precomputed key vectors WITH MATCHING SEED:
    results/matched_ordering/key_geometry/keys_seed{SEED}_layer6.npz
    OR results/key_vectors/full_mcf/keys_seed{SEED}_layer6.npz

IMPORTANT: Key vectors must be seed-matched. The record pool is seed-dependent
(different seeds draw different 10K subsets of MCF). Using keys extracted for
a different seed's pool will produce an ordering that does NOT minimize
interference on the actual stream it runs on.

Usage:
    uv run python scheduling/generate_scheduling_orderings.py --seed 42
    uv run python scheduling/generate_scheduling_orderings.py --seed 2024 --methods greedy_minmax,random
    uv run python scheduling/generate_scheduling_orderings.py --seed 42 --keys_file results/key_vectors/full_mcf/keys_seed42_layer6.npz
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


def resolve_key_path(seed: int, explicit_path: str | None = None) -> Path:
    """Resolve the key vector file for a given seed. Hard-fails on mismatch.

    Resolution order (when no explicit path given):
      1. results/matched_ordering/key_geometry/keys_seed{SEED}_layer6.npz
      2. results/key_vectors/full_mcf/keys_seed{SEED}_layer6.npz

    Refuses to fall back to a different seed's file.
    """
    if explicit_path:
        p = Path(explicit_path)
        if not p.exists():
            raise FileNotFoundError(f"Explicit key file not found: {p}")
        return p

    candidates = [
        RESULT_ROOT / "matched_ordering" / "key_geometry" / f"keys_seed{seed}_layer6.npz",
        RESULT_ROOT / "key_vectors" / "full_mcf" / f"keys_seed{seed}_layer6.npz",
    ]
    for path in candidates:
        if path.exists():
            return path

    # ─── Hard guard: refuse to use wrong-seed keys ───
    existing_42 = RESULT_ROOT / "key_vectors" / "full_mcf" / "keys_seed42_layer6.npz"
    hint = ""
    if existing_42.exists() and seed != 42:
        hint = (
            f"\n\n  NOTE: keys_seed42_layer6.npz exists but CANNOT be used for seed {seed}.\n"
            f"  The record pool is seed-dependent — using another seed's keys produces\n"
            f"  an ordering that does not minimize interference on the actual stream.\n"
            f"\n  To extract keys for seed {seed}'s pool (requires GPU, ~30min):\n"
            f"    uv run python analysis/matched_ordering_key_geometry.py --seed {seed} --layer 6\n"
            f"\n  OR if you have verified that keys_seed42_layer6.npz covers ALL 20,877 MCF records\n"
            f"  (not just the seed-42 pool), you may symlink:\n"
            f"    ln -s $(pwd)/results/key_vectors/full_mcf/keys_seed42_layer6.npz \\\n"
            f"          results/key_vectors/full_mcf/keys_seed{seed}_layer6.npz"
        )

    raise FileNotFoundError(
        f"No seed-matched key vectors found for seed {seed}.\n"
        f"  Checked: {[str(p) for p in candidates]}{hint}"
    )


def load_and_filter_keys(records: list, seed: int, keys_file: str | None = None) -> np.ndarray:
    """Load key vectors and filter/reorder to match the record pool.

    Args:
        records: Record pool (list of dicts with 'case_id').
        seed: The generation seed — used to resolve the key file.
        keys_file: Optional explicit path to key file (bypasses seed resolution).

    Returns:
        keys: (N, D) float64, L2-normalized, in same order as records.
    """
    keys_path = resolve_key_path(seed, keys_file)
    print(f"  Loading keys from: {keys_path}")

    npz = np.load(keys_path)
    all_keys = npz["keys"].astype(np.float64)
    all_case_ids = npz["case_ids"].tolist()

    # Build lookup: case_id -> row index in full key matrix
    key_idx_by_id = {cid: i for i, cid in enumerate(all_case_ids)}

    # Filter to record pool's case_ids, maintaining record order
    record_case_ids = [r["case_id"] for r in records]
    missing = [cid for cid in record_case_ids if cid not in key_idx_by_id]
    if missing:
        raise ValueError(
            f"{len(missing)}/{len(record_case_ids)} records in the seed-{seed} pool "
            f"have no key in {keys_path.name}.\n"
            f"  First missing case_id: {missing[0]}\n"
            f"  This confirms the key file does NOT cover this seed's record pool.\n"
            f"  Extract keys for seed {seed}: "
            f"uv run python analysis/matched_ordering_key_geometry.py --seed {seed} --layer 6"
        )

    indices = [key_idx_by_id[cid] for cid in record_case_ids]
    keys = all_keys[indices]
    print(f"  Key coverage: {len(indices)}/{len(record_case_ids)} records matched (100%)")
    print(f"  Key shape: {keys.shape}, dtype: {keys.dtype}")

    # L2 normalize (float64)
    norms = np.linalg.norm(keys, axis=1, keepdims=True)
    zero_norm = (norms < 1e-10).sum()
    if zero_norm > 0:
        print(f"  WARNING: {int(zero_norm)} keys have near-zero norm (clamped to 1e-8)")
    norms = np.maximum(norms, 1e-8)
    keys = keys / norms

    return keys


def main():
    parser = argparse.ArgumentParser(
        description="Generate interference-aware scheduling orderings"
    )
    parser.add_argument("--seed", type=int, required=True,
                        help="Seed (must match existing ordering files AND key vectors)")
    parser.add_argument("--methods", type=str, default=",".join(ALL_METHODS),
                        help="Comma-separated methods to generate (default: all)")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Output directory (default: results/matched_ordering/orderings/)")
    parser.add_argument("--keys_file", type=str, default=None,
                        help="Explicit path to key vectors .npz (overrides seed-based resolution)")
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

    print(f"\n  Step 2: Load and normalize keys (seed-matched)")
    keys = load_and_filter_keys(records, seed=args.seed, keys_file=args.keys_file)

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

        # ── Post-generation exposure validation ──────────────────────────────
        print(f"\n    Exposure validation:")

        # First-1K frac>0.3 (the headline metric)
        first_1k = min(1000, N)
        first_1k_keys = keys[perm[:first_1k]]
        subsequent_keys = keys[perm[first_1k:]]
        if len(subsequent_keys) > 0:
            # Chunk to manage memory
            max_cos_values = np.zeros(first_1k)
            chunk_sz = 500
            for ci in range(0, first_1k, chunk_sz):
                ce = min(ci + chunk_sz, first_1k)
                cos_chunk = first_1k_keys[ci:ce] @ subsequent_keys.T
                max_cos_values[ci:ce] = cos_chunk.max(axis=1)

            frac_03 = float(np.mean(max_cos_values > 0.3))
            frac_04 = float(np.mean(max_cos_values > 0.4))
            mean_max = float(np.mean(max_cos_values))
            print(f"      frac>0.3: {frac_03:.4f}  frac>0.4: {frac_04:.4f}  mean_max_cos: {mean_max:.4f}")

            # Sanity check for greedy: expect near-zero frac>0.4
            if method == "greedy_minmax" and frac_04 > 0.01:
                print(f"      ⚠ WARNING: greedy_minmax frac>0.4 = {frac_04:.4f} — "
                      f"expected near 0. Check key coverage.")
        else:
            print(f"      (insufficient subsequent keys for validation)")

        # Within-batch cosine (first 10 batches)
        n_val_batches = min(10, N // args.batch_size)
        wb_cos_vals = []
        for vb in range(n_val_batches):
            bk = keys[perm[vb * args.batch_size:(vb + 1) * args.batch_size]]
            cm = bk @ bk.T
            n_bk = len(bk)
            mask = np.triu(np.ones((n_bk, n_bk), dtype=bool), k=1)
            wb_cos_vals.append(float(cm[mask].mean()))
        if wb_cos_vals:
            print(f"      within_batch_cos (first {n_val_batches} batches): {np.mean(wb_cos_vals):.4f}")

    # ── Done ──────────────────────────────────────────────────────────────────

    print(f"\n{'='*70}")
    print("Done. Run experiments with:")
    for method in methods:
        print(f"  bash scripts/run_matched_ordering.sh {args.seed} AlphaEdit {method}")
    print(f"\nOr validate geometry first:")
    print(f"  uv run python scheduling/validate_ordering.py --seed {args.seed}")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()

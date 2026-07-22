#!/usr/bin/env python3
"""
Key-Clustered Ordering Dataset Generator.

Uses precomputed MEMIT key vectors (from the base model) to construct
orderings that directly manipulate key-space geometry:

  KEY-CLUSTERED ordering:
    Records grouped so that each batch contains keys with HIGH mutual
    cosine similarity. Uses spherical k-means on L2-normalized keys
    to partition into ~50 clusters, then packs batches from single
    clusters (or nearby clusters).

  KEY-DISPERSED ordering:
    Records arranged so that each batch contains keys from MANY different
    clusters, minimizing within-batch cosine similarity. Round-robin
    assignment from clusters into batches.

This directly manipulates the editor's key representation rather than
using relation_id as a proxy (which only produces 1.31x key-space effect).

Guarantees:
  - Identical facts, subjects, targets
  - Zero duplicates, zero conflicts
  - Same total key set (global spectrum identical)
  - Only within-batch and prefix key geometry differs

Requires:
  - Precomputed keys from matched_ordering_key_geometry.py:
    results/matched_ordering/key_geometry/keys_seed42_layer6.npz
  - Corresponding stream file (either clustered or dispersed — same facts):
    results/matched_ordering/orderings/clustered_seed42.json

Usage:
    uv run python src/datasets/key_clustered_ordering_dataset.py \
        --seed 42 \
        --keys_path results/matched_ordering/key_geometry/keys_seed42_layer6.npz \
        --stream_path results/matched_ordering/orderings/clustered_seed42.json \
        --n_clusters 50 --batch_size 100 \
        --output_dir results/matched_ordering/orderings
"""

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path

import numpy as np


def spherical_kmeans(keys: np.ndarray, n_clusters: int, max_iter: int = 50, seed: int = 42) -> np.ndarray:
    """Spherical k-means clustering on L2-normalized keys.

    Returns cluster assignments (N,) array of ints.
    """
    rng = np.random.default_rng(seed)
    N, D = keys.shape

    # L2-normalize
    norms = np.linalg.norm(keys, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-8)
    normed = keys / norms

    # Initialize centroids by random selection
    init_idx = rng.choice(N, size=n_clusters, replace=False)
    centroids = normed[init_idx].copy()

    assignments = np.zeros(N, dtype=np.int32)

    for iteration in range(max_iter):
        # Assign each point to nearest centroid (cosine = dot product on unit vectors)
        sims = normed @ centroids.T  # (N, k)
        new_assignments = sims.argmax(axis=1)

        # Check convergence
        changed = (new_assignments != assignments).sum()
        assignments = new_assignments

        if changed == 0:
            print(f"      Spherical k-means converged at iteration {iteration + 1}")
            break

        # Update centroids
        for c in range(n_clusters):
            mask = assignments == c
            if mask.sum() > 0:
                centroids[c] = normed[mask].mean(axis=0)
                # Re-normalize centroid
                cn = np.linalg.norm(centroids[c])
                if cn > 1e-8:
                    centroids[c] /= cn
    else:
        print(f"      Spherical k-means: {max_iter} iterations, {changed} still changing")

    return assignments


def create_key_clustered_ordering(
    records: list,
    assignments: np.ndarray,
    rng: random.Random,
    batch_size: int,
) -> list:
    """Arrange records so that each batch contains keys from the same cluster.

    Strategy: group records by cluster, sort clusters by size (largest first),
    pack groups contiguously. Shuffle within each cluster for randomness.
    """
    # Group by cluster
    by_cluster = defaultdict(list)
    for i, record in enumerate(records):
        by_cluster[int(assignments[i])].append(record)

    # Shuffle within each cluster
    for c in by_cluster:
        rng.shuffle(by_cluster[c])

    # Sort clusters by size (largest first) for densest packing
    sorted_clusters = sorted(by_cluster.keys(), key=lambda c: len(by_cluster[c]), reverse=True)

    # Pack contiguously
    ordered = []
    for c in sorted_clusters:
        ordered.extend(by_cluster[c])

    return ordered


def create_key_dispersed_ordering(
    records: list,
    assignments: np.ndarray,
    rng: random.Random,
    batch_size: int,
) -> list:
    """Arrange records so that each batch draws from MANY different clusters.

    Strategy: round-robin from each cluster into sequential positions.
    This maximizes within-batch cluster diversity.
    """
    # Group by cluster
    by_cluster = defaultdict(list)
    for i, record in enumerate(records):
        by_cluster[int(assignments[i])].append(record)

    # Shuffle within each cluster
    for c in by_cluster:
        rng.shuffle(by_cluster[c])

    # Shuffle cluster order
    cluster_ids = list(by_cluster.keys())
    rng.shuffle(cluster_ids)

    # Build queues
    queues = {c: list(by_cluster[c]) for c in cluster_ids}

    # Round-robin interleave
    ordered = []
    while len(ordered) < len(records):
        added = False
        for c in cluster_ids:
            if queues[c]:
                ordered.append(queues[c].pop(0))
                added = True
                if len(ordered) >= len(records):
                    break
        if not added:
            break

    return ordered


def validate_and_report(
    records: list,
    key_clustered: list,
    key_dispersed: list,
    keys: np.ndarray,
    assignments: np.ndarray,
    batch_size: int,
) -> dict:
    """Validate orderings and compute key-geometry metrics."""
    # Same records
    assert len(key_clustered) == len(key_dispersed) == len(records)
    clust_ids = set(r["case_id"] for r in key_clustered)
    disp_ids = set(r["case_id"] for r in key_dispersed)
    orig_ids = set(r["case_id"] for r in records)
    assert clust_ids == disp_ids == orig_ids

    # No duplicates
    assert len(clust_ids) == len(key_clustered)

    # Build case_id → key index mapping from original records
    case_id_to_idx = {records[i]["case_id"]: i for i in range(len(records))}

    # Normalize keys for cosine computation
    norms = np.linalg.norm(keys, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-8)
    normed_keys = keys / norms

    def compute_within_batch_cosine(ordering):
        n_batches = len(ordering) // batch_size
        sims = []
        for b in range(n_batches):
            batch = ordering[b * batch_size: (b + 1) * batch_size]
            indices = [case_id_to_idx[r["case_id"]] for r in batch]
            batch_normed = normed_keys[indices]
            cos_matrix = batch_normed @ batch_normed.T
            n = len(indices)
            mask = np.triu(np.ones((n, n), dtype=bool), k=1)
            sims.append(float(cos_matrix[mask].mean()))
        return sims

    def compute_batch_cluster_diversity(ordering):
        n_batches = len(ordering) // batch_size
        diversities = []
        for b in range(n_batches):
            batch = ordering[b * batch_size: (b + 1) * batch_size]
            batch_clusters = set(int(assignments[case_id_to_idx[r["case_id"]]]) for r in batch)
            diversities.append(len(batch_clusters))
        return diversities

    print("    Computing within-batch cosine...")
    clust_cosines = compute_within_batch_cosine(key_clustered)
    disp_cosines = compute_within_batch_cosine(key_dispersed)

    print("    Computing batch cluster diversity...")
    clust_diversity = compute_batch_cluster_diversity(key_clustered)
    disp_diversity = compute_batch_cluster_diversity(key_dispersed)

    # Cluster statistics
    cluster_sizes = np.bincount(assignments)

    props = {
        "n_records": len(records),
        "n_clusters": int(assignments.max() + 1),
        "batch_size": batch_size,
        "n_batches": len(records) // batch_size,
        "cluster_sizes": {
            "mean": float(cluster_sizes.mean()),
            "min": int(cluster_sizes.min()),
            "max": int(cluster_sizes.max()),
            "std": float(cluster_sizes.std()),
        },
        "key_clustered": {
            "mean_within_batch_cosine": float(np.mean(clust_cosines)),
            "std_within_batch_cosine": float(np.std(clust_cosines)),
            "min_within_batch_cosine": float(np.min(clust_cosines)),
            "max_within_batch_cosine": float(np.max(clust_cosines)),
            "mean_clusters_per_batch": float(np.mean(clust_diversity)),
            "min_clusters_per_batch": int(np.min(clust_diversity)),
        },
        "key_dispersed": {
            "mean_within_batch_cosine": float(np.mean(disp_cosines)),
            "std_within_batch_cosine": float(np.std(disp_cosines)),
            "min_within_batch_cosine": float(np.min(disp_cosines)),
            "max_within_batch_cosine": float(np.max(disp_cosines)),
            "mean_clusters_per_batch": float(np.mean(disp_diversity)),
            "min_clusters_per_batch": int(np.min(disp_diversity)),
        },
        "cosine_ratio": float(np.mean(clust_cosines) / max(np.mean(disp_cosines), 1e-10)),
    }

    return props


def main():
    parser = argparse.ArgumentParser(
        description="Generate key-clustered/key-dispersed orderings using MEMIT key vectors"
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--keys_path", type=str, required=True,
                        help="Path to precomputed keys .npz (from key_geometry.py)")
    parser.add_argument("--stream_path", type=str, required=True,
                        help="Path to any stream JSON with the same records (for metadata)")
    parser.add_argument("--n_clusters", type=int, default=50,
                        help="Number of spherical k-means clusters")
    parser.add_argument("--batch_size", type=int, default=100)
    parser.add_argument("--output_dir", type=str, default="results/matched_ordering/orderings")
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parent.parent.parent

    print(f"\n{'='*70}")
    print("Key-Clustered Ordering Dataset Generator")
    print(f"  Seed:       {args.seed}")
    print(f"  Keys:       {args.keys_path}")
    print(f"  Stream:     {args.stream_path}")
    print(f"  Clusters:   {args.n_clusters}")
    print(f"  Batch size: {args.batch_size}")
    print(f"{'='*70}")

    # Load keys
    print("\n  Loading precomputed keys...")
    npz = np.load(args.keys_path)
    keys = npz["keys"]
    saved_case_ids = npz["case_ids"].tolist()
    print(f"  Keys shape: {keys.shape}")
    print(f"  Case IDs: {len(saved_case_ids)}")

    # Load stream records (for metadata — subjects, targets, prompts)
    print(f"\n  Loading stream records...")
    stream_path = Path(args.stream_path)
    if not stream_path.is_absolute():
        stream_path = project_root / stream_path
    with open(stream_path) as f:
        all_records = json.load(f)
    print(f"  Records: {len(all_records)}")

    # Build case_id → key index mapping
    key_idx_by_id = {cid: i for i, cid in enumerate(saved_case_ids)}

    # Filter to records that have keys, preserving stream order
    records = [r for r in all_records if r["case_id"] in key_idx_by_id]
    key_indices = [key_idx_by_id[r["case_id"]] for r in records]
    keys = keys[key_indices]
    print(f"  Aligned: {len(records)} records with keys (from {len(saved_case_ids)} total keys)")

    # Spherical k-means clustering
    print(f"\n  Running spherical k-means (k={args.n_clusters})...")
    assignments = spherical_kmeans(keys, args.n_clusters, max_iter=100, seed=args.seed)

    cluster_sizes = np.bincount(assignments)
    print(f"  Clusters: {len(cluster_sizes)} non-empty")
    print(f"  Cluster sizes: mean={cluster_sizes.mean():.1f}, "
          f"min={cluster_sizes.min()}, max={cluster_sizes.max()}, "
          f"std={cluster_sizes.std():.1f}")

    # Generate orderings
    rng_clust = random.Random(args.seed + 3000)
    rng_disp = random.Random(args.seed + 4000)

    print("\n  Generating key-clustered ordering...")
    key_clustered = create_key_clustered_ordering(records, assignments, rng_clust, args.batch_size)

    print("  Generating key-dispersed ordering...")
    key_dispersed = create_key_dispersed_ordering(records, assignments, rng_disp, args.batch_size)

    # Validate and compute metrics
    print("\n  Validating and computing metrics...")
    props = validate_and_report(records, key_clustered, key_dispersed, keys, assignments, args.batch_size)

    print(f"\n  Key-geometry metrics:")
    print(f"    Key-clustered: {props['key_clustered']['mean_within_batch_cosine']:.4f} "
          f"mean within-batch cosine ({props['key_clustered']['mean_clusters_per_batch']:.1f} clusters/batch)")
    print(f"    Key-dispersed: {props['key_dispersed']['mean_within_batch_cosine']:.4f} "
          f"mean within-batch cosine ({props['key_dispersed']['mean_clusters_per_batch']:.1f} clusters/batch)")
    print(f"    Cosine ratio: {props['cosine_ratio']:.2f}x")

    # Save outputs
    out_dir = Path(args.output_dir)
    if not out_dir.is_absolute():
        out_dir = project_root / args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    clust_path = out_dir / f"key_clustered_seed{args.seed}.json"
    disp_path = out_dir / f"key_dispersed_seed{args.seed}.json"
    diag_dir = out_dir.parent / "diagnostics"
    diag_dir.mkdir(parents=True, exist_ok=True)
    props_path = diag_dir / f"key_stream_properties_seed{args.seed}.json"

    with open(clust_path, "w") as f:
        json.dump(key_clustered, f)
    with open(disp_path, "w") as f:
        json.dump(key_dispersed, f)
    with open(props_path, "w") as f:
        json.dump(props, f, indent=2)

    print(f"\n  Output:")
    print(f"    Key-clustered: {clust_path}")
    print(f"    Key-dispersed: {disp_path}")
    print(f"    Properties:    {props_path}")
    print(f"\n{'='*70}")
    print("Done. Run experiments with:")
    print(f"  bash scripts/run_matched_ordering.sh {args.seed} memit_seq key_clustered")
    print(f"  bash scripts/run_matched_ordering.sh {args.seed} memit_seq key_dispersed")
    print(f"  bash scripts/run_matched_ordering.sh {args.seed} alphaedit key_clustered")
    print(f"  bash scripts/run_matched_ordering.sh {args.seed} alphaedit key_dispersed")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()

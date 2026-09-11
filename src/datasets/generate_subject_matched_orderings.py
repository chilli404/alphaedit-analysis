#!/usr/bin/env python3
"""Generate subject-matched ordering datasets that isolate key-geometry from relation identity.

Creates TWO orderings from the same 5K unique-subject pool:

  subj_matched_clustered: high within-batch key cosine similarity
  subj_matched_dispersed: low within-batch key cosine similarity

CRITICAL PROPERTY: Per-batch relation distribution is IDENTICAL between orderings.
This eliminates the confound that key-similarity manipulation changes semantic
(relation-type) recurrence patterns.

Algorithm:
  1. Compute batch centroids via spherical k-means on all keys (ignoring relation)
  2. Compute per-batch relation quotas (stratified assignment)
  3. For each batch & relation slot:
     - "clustered": greedily pick the available record MOST similar to batch centroid
     - "dispersed": greedily pick the available record LEAST similar to batch centroid
  4. Per-batch relation counts are identical by construction

Requires:
  - Precomputed full-MCF key vectors:
    results/key_vectors/full_mcf/keys_seed42_layer6.npz

Usage:
    uv run python src/datasets/generate_subject_matched_orderings.py --seed 2024
    uv run python src/datasets/generate_subject_matched_orderings.py --seed 42 --stream_length 5000
"""

import argparse
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


# ─── Reuse record selection from generate_orderings ──────────────────────────


def load_counterfact(data_dir: Path) -> list:
    cf_path = data_dir / "multi_counterfact.json"
    if not cf_path.exists():
        raise FileNotFoundError(
            f"multi_counterfact.json not found at {cf_path}. "
            "Run scripts/link_dsets.sh or download_datasets.sh first."
        )
    with open(cf_path, "r") as f:
        return json.load(f)


def select_clean_pool(data: list, rng: random.Random, stream_length: int) -> list:
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
        if subject in used_subjects or case_id in used_case_ids:
            continue
        selected.append(record)
        used_subjects.add(subject)
        used_case_ids.add(case_id)

    if len(selected) < stream_length:
        print(f"  WARNING: Only found {len(selected)} unique-subject records "
              f"(requested {stream_length})")
    return selected


# ─── Stratified Batch Assignment ─────────────────────────────────────────────


def build_batch_quotas(
    records: list,
    batch_size: int,
    n_batches: int,
) -> list[dict[str, int]]:
    """Compute per-batch relation quotas.

    Returns: list of dicts, quotas[batch_idx][relation_id] = count.
    Every batch has the SAME relation distribution (quotas sum to batch_size).
    """
    usable = n_batches * batch_size

    by_relation = defaultdict(list)
    for i, record in enumerate(records[:usable]):
        rel = record["requested_rewrite"]["relation_id"]
        by_relation[rel].append(i)

    sorted_relations = sorted(by_relation.keys(),
                              key=lambda r: len(by_relation[r]), reverse=True)

    quotas = [{} for _ in range(n_batches)]
    batch_fill = [0] * n_batches

    for rel in sorted_relations:
        n_rel = len(by_relation[rel])
        base_quota = n_rel // n_batches
        remainder = n_rel % n_batches

        batch_order = sorted(range(n_batches), key=lambda b: (batch_fill[b], b))
        extra_batches = set(batch_order[:remainder])

        for b in range(n_batches):
            count = base_quota + (1 if b in extra_batches else 0)
            if count > 0:
                quotas[b][rel] = count
            batch_fill[b] += count

    assert all(f == batch_size for f in batch_fill), \
        f"Batch fill not uniform: min={min(batch_fill)}, max={max(batch_fill)}"

    return quotas


# ─── Main Ordering Construction ──────────────────────────────────────────────


def build_subject_matched_ordering(
    records: list,
    keys: np.ndarray,
    batch_size: int,
    mode: str,
    seed: int,
    quotas: list[dict[str, int]],
) -> list:
    """Build ordering using batch-centroid attraction/repulsion.

    Each batch gets a "target direction" (its k-means centroid). For each relation
    slot in that batch:
      - CLUSTERED: pick the available record MOST similar to batch centroid
      - DISPERSED: pick the available record LEAST similar to batch centroid

    This creates cross-relation pairing differences while keeping per-batch
    relation distribution identical.
    """
    rng_py = random.Random(seed + (7000 if mode == "clustered" else 8000))

    n_batches = len(records) // batch_size
    usable = n_batches * batch_size

    # Normalize keys
    norms = np.linalg.norm(keys[:usable], axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-8)
    normed_keys = keys[:usable] / norms

    # Cluster to get batch centroids (target directions)
    centroids = _spherical_kmeans_centroids(normed_keys, n_batches, seed=seed)

    # Build per-relation pools
    by_relation = defaultdict(list)
    for i, record in enumerate(records[:usable]):
        rel = record["requested_rewrite"]["relation_id"]
        by_relation[rel].append(i)

    # Precompute: sims[i, b] = cosine(key_i, centroid_b)
    sims = normed_keys @ centroids.T  # (usable, n_batches)

    # Track availability per relation
    rel_available = {rel: set(indices) for rel, indices in by_relation.items()}

    batch_contents = [[] for _ in range(n_batches)]

    # Fill batches — process in random order to avoid systematic bias
    batch_order = list(range(n_batches))
    rng_py.shuffle(batch_order)

    for b in batch_order:
        for rel, count in quotas[b].items():
            available = rel_available[rel]
            if not available or count == 0:
                continue

            avail_list = list(available)
            scores = sims[avail_list, b]

            # Handle NaN scores (from zero-norm keys)
            valid_mask = np.isfinite(scores)
            if not valid_mask.any():
                chosen_indices = avail_list[:count]
            else:
                if mode == "clustered":
                    # Pick records MOST similar to batch centroid
                    order = np.argsort(scores)
                    # Take from the end (highest similarity), skip NaN
                    valid_order = [i for i in reversed(order) if valid_mask[i]]
                    chosen_indices = [avail_list[i] for i in valid_order[:count]]
                else:
                    # Pick records LEAST similar to batch centroid
                    # Replace NaN with +inf so they sort to the end
                    scores_clean = np.where(valid_mask, scores, np.inf)
                    order = np.argsort(scores_clean)
                    chosen_indices = [avail_list[order[i]] for i in range(min(count, len(order)))]

            for record_idx in chosen_indices:
                batch_contents[b].append(record_idx)
                available.discard(record_idx)

    # Shuffle within each batch
    for batch in batch_contents:
        rng_py.shuffle(batch)

    ordering = []
    for batch in batch_contents:
        ordering.extend(batch)

    return [records[i] for i in ordering]


def _spherical_kmeans_centroids(normed_keys: np.ndarray, n_clusters: int, seed: int = 42, max_iter: int = 50) -> np.ndarray:
    """Spherical k-means — returns centroids (n_clusters, D)."""
    rng = np.random.default_rng(seed)
    N, D = normed_keys.shape

    init_idx = rng.choice(N, size=n_clusters, replace=False)
    centroids = normed_keys[init_idx].copy()

    assignments = np.zeros(N, dtype=np.int32)
    for _ in range(max_iter):
        sims = normed_keys @ centroids.T
        new_assignments = sims.argmax(axis=1)
        if np.array_equal(new_assignments, assignments):
            break
        assignments = new_assignments
        for c in range(n_clusters):
            mask = assignments == c
            if mask.sum() > 0:
                centroids[c] = normed_keys[mask].mean(axis=0)
                cn = np.linalg.norm(centroids[c])
                if cn > 1e-8:
                    centroids[c] /= cn

    return centroids


# ─── Validation ──────────────────────────────────────────────────────────────


def validate_relation_matching(
    clustered: list,
    dispersed: list,
    batch_size: int,
) -> dict:
    """Verify per-batch relation distributions match between orderings."""
    n_batches = len(clustered) // batch_size

    mismatches = 0
    total_slots = 0

    for b in range(n_batches):
        c_batch = clustered[b * batch_size: (b + 1) * batch_size]
        d_batch = dispersed[b * batch_size: (b + 1) * batch_size]

        c_rels = Counter(r["requested_rewrite"]["relation_id"] for r in c_batch)
        d_rels = Counter(r["requested_rewrite"]["relation_id"] for r in d_batch)

        all_rels = set(c_rels.keys()) | set(d_rels.keys())
        for rel in all_rels:
            diff = abs(c_rels.get(rel, 0) - d_rels.get(rel, 0))
            mismatches += diff
            total_slots += max(c_rels.get(rel, 0), d_rels.get(rel, 0))

    return {
        "n_batches": n_batches,
        "total_relation_slots": total_slots,
        "mismatched_slots": mismatches,
        "match_rate": 1.0 - (mismatches / max(total_slots, 1)),
    }


def compute_cosine_metrics(ordering: list, keys: np.ndarray, case_id_to_idx: dict, batch_size: int) -> dict:
    n_batches = len(ordering) // batch_size
    norms = np.linalg.norm(keys, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-8)
    normed = keys / norms

    cosines = []
    for b in range(n_batches):
        batch = ordering[b * batch_size: (b + 1) * batch_size]
        indices = [case_id_to_idx[r["case_id"]] for r in batch]
        batch_normed = normed[indices]
        cos_matrix = batch_normed @ batch_normed.T
        n = len(indices)
        mask = np.triu(np.ones((n, n), dtype=bool), k=1)
        batch_cos = cos_matrix[mask]
        valid = batch_cos[np.isfinite(batch_cos)]
        cosines.append(float(valid.mean()) if len(valid) > 0 else 0.0)

    return {
        "mean_within_batch_cosine": float(np.mean(cosines)),
        "std_within_batch_cosine": float(np.std(cosines)),
        "min_batch_cosine": float(np.min(cosines)),
        "max_batch_cosine": float(np.max(cosines)),
    }


# ─── Main ────────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(
        description="Generate subject-matched ordering datasets (relation-controlled key-geometry manipulation)"
    )
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--stream_length", type=int, default=5000)
    parser.add_argument("--batch_size", type=int, default=100)
    parser.add_argument("--keys_path", type=str,
                        default="results/key_vectors/full_mcf/keys_seed42_layer6.npz")
    parser.add_argument("--data_dir", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default="results/matched_ordering")
    args = parser.parse_args()

    if args.data_dir:
        data_dir = Path(args.data_dir)
    else:
        candidates = [
            PROJECT_ROOT / "data" / "dsets",
            PROJECT_ROOT / "vendor" / "AlphaEdit" / "data",
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
            sys.exit(1)

    keys_path = Path(args.keys_path)
    if not keys_path.is_absolute():
        keys_path = PROJECT_ROOT / args.keys_path

    out_base = Path(args.output_dir)
    if not out_base.is_absolute():
        out_base = PROJECT_ROOT / args.output_dir
    ord_dir = out_base / "orderings"
    diag_dir = out_base / "diagnostics"
    ord_dir.mkdir(parents=True, exist_ok=True)
    diag_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*70}")
    print("Subject-Matched Ordering Generator")
    print(f"  Seed:          {args.seed}")
    print(f"  Stream length: {args.stream_length}")
    print(f"  Batch size:    {args.batch_size}")
    print(f"  Keys:          {keys_path}")
    print(f"  Data dir:      {data_dir}")
    print(f"  Output:        {out_base}")
    print(f"{'='*70}")

    # ── Step 1: Load data and select records ──────────────────────────────────

    print(f"\n  Loading MultiCounterFact...")
    all_data = load_counterfact(data_dir)
    print(f"  Total MCF records: {len(all_data)}")

    print(f"\n  Selecting {args.stream_length} unique, non-conflicting records...")
    rng = random.Random(args.seed)
    records = select_clean_pool(all_data, rng, args.stream_length)
    print(f"  Selected: {len(records)} records")

    n_relations = len(set(r["requested_rewrite"]["relation_id"] for r in records))
    print(f"  Unique relations: {n_relations}")

    # ── Step 2: Load and filter keys ─────────────────────────────────────────

    print(f"\n  Loading keys...")
    if not keys_path.exists():
        print(f"  ERROR: Keys file not found at {keys_path}")
        sys.exit(1)

    npz = np.load(keys_path)
    all_keys = npz["keys"]
    all_case_ids = npz["case_ids"].tolist()

    key_idx_by_id = {cid: i for i, cid in enumerate(all_case_ids)}
    record_case_ids = [r["case_id"] for r in records]
    missing = [cid for cid in record_case_ids if cid not in key_idx_by_id]
    if missing:
        print(f"  WARNING: {len(missing)} records have no precomputed key — dropping them")
        records = [r for r in records if r["case_id"] in key_idx_by_id]
        record_case_ids = [r["case_id"] for r in records]

    key_indices = [key_idx_by_id[cid] for cid in record_case_ids]
    keys = all_keys[key_indices]
    print(f"  Selected keys: {keys.shape}")

    case_id_to_idx = {records[i]["case_id"]: i for i in range(len(records))}

    # ── Step 3: Build shared quotas and generate orderings ──────────────────

    n_batches = len(records) // args.batch_size
    print(f"\n  Building batch quotas ({n_batches} batches)...")
    quotas = build_batch_quotas(records, args.batch_size, n_batches)
    total_slots = sum(sum(q.values()) for q in quotas)
    n_rels_in_quotas = len(set(r for q in quotas for r in q.keys()))
    print(f"    Template covers {total_slots} record slots")
    print(f"    Relations in template: {n_rels_in_quotas}")

    print(f"\n  Generating subject-matched clustered ordering...")
    sm_clustered = build_subject_matched_ordering(
        records, keys, args.batch_size, "clustered", args.seed, quotas
    )

    print(f"  Generating subject-matched dispersed ordering...")
    sm_dispersed = build_subject_matched_ordering(
        records, keys, args.batch_size, "dispersed", args.seed, quotas
    )

    # ── Step 4: Validate ─────────────────────────────────────────────────────

    print(f"\n  Validating relation matching...")
    match_report = validate_relation_matching(sm_clustered, sm_dispersed, args.batch_size)
    print(f"    Match rate: {match_report['match_rate']:.4f}")
    print(f"    Mismatched slots: {match_report['mismatched_slots']} / {match_report['total_relation_slots']}")

    print(f"\n  Computing key-geometry metrics...")
    clust_metrics = compute_cosine_metrics(sm_clustered, keys, case_id_to_idx, args.batch_size)
    disp_metrics = compute_cosine_metrics(sm_dispersed, keys, case_id_to_idx, args.batch_size)

    cosine_ratio = clust_metrics["mean_within_batch_cosine"] / max(disp_metrics["mean_within_batch_cosine"], 1e-10)
    print(f"    Clustered mean cosine: {clust_metrics['mean_within_batch_cosine']:.4f}")
    print(f"    Dispersed mean cosine: {disp_metrics['mean_within_batch_cosine']:.4f}")
    print(f"    Cosine ratio: {cosine_ratio:.2f}x")

    # Compare to unconstrained key_clustered/key_dispersed (1.76x)
    if cosine_ratio > 1.3:
        print(f"    STRONG: substantial geometry manipulation despite relation control")
    elif cosine_ratio > 1.15:
        print(f"    OK: meaningful manipulation — expect measurable effect")
    elif cosine_ratio > 1.05:
        print(f"    MARGINAL: moderate manipulation — effect size may be small")
    else:
        print(f"    WARNING: Weak manipulation — this is expected when relation≈key structure")
        print(f"    NOTE: A weak ratio itself is informative — it quantifies how much of the")
        print(f"          key-geometry effect is explained by relation structure alone.")

    # Verify case_id integrity
    clust_ids = set(r["case_id"] for r in sm_clustered)
    disp_ids = set(r["case_id"] for r in sm_dispersed)
    orig_ids = set(r["case_id"] for r in records[:len(sm_clustered)])
    assert clust_ids == orig_ids, "Clustered ordering has wrong case_ids"
    assert disp_ids == orig_ids, "Dispersed ordering has wrong case_ids"

    # ── Step 5: Save ─────────────────────────────────────────────────────────

    print(f"\n  Saving outputs...")
    for name, ordering in [("subj_matched_clustered", sm_clustered),
                           ("subj_matched_dispersed", sm_dispersed)]:
        path = ord_dir / f"{name}_seed{args.seed}.json"
        with open(path, "w") as f:
            json.dump(ordering, f)
        print(f"    {path.name}")

    report = {
        "seed": args.seed,
        "n_records": len(records),
        "n_usable": len(sm_clustered),
        "batch_size": args.batch_size,
        "n_batches": len(sm_clustered) // args.batch_size,
        "n_relations": n_relations,
        "relation_matching": match_report,
        "key_geometry": {
            "subj_matched_clustered": clust_metrics,
            "subj_matched_dispersed": disp_metrics,
            "cosine_ratio": cosine_ratio,
        },
    }
    report_path = diag_dir / f"subject_matched_report_seed{args.seed}.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"    {report_path.name}")

    # ── Done ─────────────────────────────────────────────────────────────────

    print(f"\n{'='*70}")
    print("Done. Run experiments with:")
    print(f"  bash scripts/run_matched_ordering.sh {args.seed} AlphaEdit subj_matched_clustered")
    print(f"  bash scripts/run_matched_ordering.sh {args.seed} AlphaEdit subj_matched_dispersed")
    print(f"  bash scripts/run_matched_ordering.sh {args.seed} MEMIT-Seq-lp1.0-ld0.0-cache0 subj_matched_clustered")
    print(f"  bash scripts/run_matched_ordering.sh {args.seed} MEMIT-Seq-lp1.0-ld0.0-cache0 subj_matched_dispersed")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()

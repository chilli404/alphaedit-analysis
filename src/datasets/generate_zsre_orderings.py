#!/usr/bin/env python3
"""Generate matched ordering datasets for ZsRE (one seed).

Replicates the key-geometry ordering experiment from MCF on a second benchmark
to demonstrate the memory-space mechanism generalizes beyond MultiCounterFact.

ZsRE has no explicit relation_id, so we derive semantic groups from the
prompt's relation phrase (text between subject and "?") and also generate
key-geometry orderings via spherical k-means on precomputed key vectors.

Produces 4 orderings of the same N records (unique subjects only):
  SEMANTIC orderings (by derived relation):
    - clustered: same-relation edits grouped together
    - dispersed: same-relation edits spread across batches

  KEY-GEOMETRY orderings (by key-vector similarity):
    - key_clustered: similar keys grouped in same batches
    - key_dispersed: similar keys spread across batches

Requires:
  - ZsRE dataset (zsre_mend_eval.json) in data/dsets/ or vendor/AlphaEdit/data/
  - Precomputed key vectors: results/key_vectors/zsre/keys_seed{SEED}_layer6.npz

Usage:
    uv run python src/datasets/generate_zsre_orderings.py --seed 42
    uv run python src/datasets/generate_zsre_orderings.py --seed 42 --stream_length 3000
"""

import argparse
import json
import os
import random
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


# ─── ZsRE Loading ────────────────────────────────────────────────────────────


def load_zsre(data_dir: Path) -> list:
    """Load zsre_mend_eval.json and normalize to MCF-like format."""
    zsre_path = data_dir / "zsre_mend_eval.json"
    if not zsre_path.exists():
        raise FileNotFoundError(
            f"zsre_mend_eval.json not found at {zsre_path}. "
            "Run scripts/download_datasets.sh first."
        )
    with open(zsre_path, "r") as f:
        raw = json.load(f)

    records = []
    for i, rec in enumerate(raw):
        subject = rec["subject"]
        src = rec["src"]
        prompt_template = src.replace(subject, "{}")
        relation = extract_relation(src, subject)

        records.append({
            "case_id": i,
            "requested_rewrite": {
                "prompt": prompt_template,
                "subject": subject,
                "target_new": {"str": rec["answers"][0]},
                "target_true": {"str": "<|endoftext|>"},
                "relation_id": relation,
            },
            "paraphrase_prompts": [rec["rephrase"]],
            "neighborhood_prompts": [],
            "attribute_prompts": [],
            "generation_prompts": [],
            "_raw_src": src,
        })
    return records


def extract_relation(src: str, subject: str) -> str:
    """Extract a relation identifier from the ZsRE prompt.

    ZsRE prompts are typically of the form:
      "Subject relation phrase?"
    or
      "What is the relation of Subject?"

    We extract the non-subject portion as the relation key.
    """
    relation = src.replace(subject, "").strip()
    relation = re.sub(r'\s+', ' ', relation)
    relation = relation.rstrip("?").strip()
    return relation


# ─── Record Selection ────────────────────────────────────────────────────────


def select_clean_pool(
    data: list,
    rng: random.Random,
    stream_length: int,
    pool_limit: int | None = None,
) -> list:
    """Select unique, non-conflicting records from ZsRE.

    pool_limit: Only consider the first pool_limit records. If None, searches
    the entire dataset (needed to reach 10K+ unique subjects).
    """
    pool = list(data[:pool_limit]) if pool_limit else list(data)
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

        if subject in used_subjects:
            continue
        if case_id in used_case_ids:
            continue

        selected.append(record)
        used_subjects.add(subject)
        used_case_ids.add(case_id)

    if len(selected) < stream_length:
        print(f"  WARNING: Only found {len(selected)} unique-subject records "
              f"from first {limit} (requested {stream_length})")

    return selected


# ─── Semantic Orderings ──────────────────────────────────────────────────────


def create_clustered_ordering(records: list, rng: random.Random) -> list:
    """Group same-relation edits in consecutive batches."""
    by_relation = defaultdict(list)
    for record in records:
        rel = record["requested_rewrite"]["relation_id"]
        by_relation[rel].append(record)

    for rel in by_relation:
        rng.shuffle(by_relation[rel])

    sorted_relations = sorted(
        by_relation.keys(), key=lambda r: len(by_relation[r]), reverse=True
    )

    clustered = []
    for rel in sorted_relations:
        clustered.extend(by_relation[rel])
    return clustered


def create_dispersed_ordering(records: list, rng: random.Random) -> list:
    """Spread same-relation edits maximally across batches (round-robin)."""
    by_relation = defaultdict(list)
    for record in records:
        rel = record["requested_rewrite"]["relation_id"]
        by_relation[rel].append(record)

    for rel in by_relation:
        rng.shuffle(by_relation[rel])

    relation_ids = list(by_relation.keys())
    rng.shuffle(relation_ids)

    queues = {rel: list(by_relation[rel]) for rel in relation_ids}

    dispersed = []
    while len(dispersed) < len(records):
        added = False
        for rel in relation_ids:
            if queues[rel]:
                dispersed.append(queues[rel].pop(0))
                added = True
                if len(dispersed) >= len(records):
                    break
        if not added:
            break
    return dispersed


# ─── Key-Geometry Orderings ──────────────────────────────────────────────────


def spherical_kmeans(keys: np.ndarray, n_clusters: int, max_iter: int = 100, seed: int = 42) -> np.ndarray:
    """Spherical k-means clustering on L2-normalized keys."""
    rng = np.random.default_rng(seed)
    N, D = keys.shape

    norms = np.linalg.norm(keys, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-8)
    normed = keys / norms

    init_idx = rng.choice(N, size=n_clusters, replace=False)
    centroids = normed[init_idx].copy()

    assignments = np.zeros(N, dtype=np.int32)

    for iteration in range(max_iter):
        sims = normed @ centroids.T
        new_assignments = sims.argmax(axis=1)

        changed = (new_assignments != assignments).sum()
        assignments = new_assignments

        if changed == 0:
            print(f"    Spherical k-means converged at iteration {iteration + 1}")
            break

        for c in range(n_clusters):
            mask = assignments == c
            if mask.sum() > 0:
                centroids[c] = normed[mask].mean(axis=0)
                cn = np.linalg.norm(centroids[c])
                if cn > 1e-8:
                    centroids[c] /= cn
    else:
        print(f"    Spherical k-means: {max_iter} iterations, {changed} still changing")

    return assignments


def create_key_clustered_ordering(records: list, assignments: np.ndarray, rng: random.Random) -> list:
    """Group same-cluster keys in consecutive batches."""
    by_cluster = defaultdict(list)
    for i, record in enumerate(records):
        by_cluster[int(assignments[i])].append(record)

    for c in by_cluster:
        rng.shuffle(by_cluster[c])

    sorted_clusters = sorted(by_cluster.keys(), key=lambda c: len(by_cluster[c]), reverse=True)

    ordered = []
    for c in sorted_clusters:
        ordered.extend(by_cluster[c])
    return ordered


def create_key_dispersed_ordering(records: list, assignments: np.ndarray, rng: random.Random) -> list:
    """Spread same-cluster keys maximally across batches (round-robin)."""
    by_cluster = defaultdict(list)
    for i, record in enumerate(records):
        by_cluster[int(assignments[i])].append(record)

    for c in by_cluster:
        rng.shuffle(by_cluster[c])

    cluster_ids = list(by_cluster.keys())
    rng.shuffle(cluster_ids)

    queues = {c: list(by_cluster[c]) for c in cluster_ids}

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


# ─── Validation ──────────────────────────────────────────────────────────────


def validate_and_compute_metrics(
    records: list,
    clustered: list,
    dispersed: list,
    key_clustered: list,
    key_dispersed: list,
    keys: np.ndarray,
    assignments: np.ndarray,
    batch_size: int,
) -> dict:
    """Validate all orderings and compute metrics."""
    orig_ids = set(r["case_id"] for r in records)

    for name, ordering in [("clustered", clustered), ("dispersed", dispersed),
                           ("key_clustered", key_clustered), ("key_dispersed", key_dispersed)]:
        ord_ids = set(r["case_id"] for r in ordering)
        assert ord_ids == orig_ids, f"{name}: case_id mismatch"
        assert len(ordering) == len(records), f"{name}: length mismatch"
        assert len(set(r["case_id"] for r in ordering)) == len(ordering), f"{name}: duplicates"

    for name, ordering in [("clustered", clustered), ("dispersed", dispersed),
                           ("key_clustered", key_clustered), ("key_dispersed", key_dispersed)]:
        subjects = [r["requested_rewrite"]["subject"] for r in ordering]
        assert len(subjects) == len(set(subjects)), f"{name}: duplicate subjects"

    def batch_concentration(ordering):
        n_batches = len(ordering) // batch_size
        concentrations = []
        for b in range(n_batches):
            batch = ordering[b * batch_size: (b + 1) * batch_size]
            relations = [r["requested_rewrite"]["relation_id"] for r in batch]
            counts = Counter(relations)
            same_pairs = sum(c * (c - 1) for c in counts.values())
            total_pairs = len(relations) * (len(relations) - 1)
            concentrations.append(same_pairs / total_pairs if total_pairs > 0 else 0)
        return concentrations

    clust_conc = batch_concentration(clustered)
    disp_conc = batch_concentration(dispersed)

    case_id_to_idx = {records[i]["case_id"]: i for i in range(len(records))}
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

    kc_cosines = compute_within_batch_cosine(key_clustered)
    kd_cosines = compute_within_batch_cosine(key_dispersed)
    kc_diversity = compute_batch_cluster_diversity(key_clustered)
    kd_diversity = compute_batch_cluster_diversity(key_dispersed)

    cluster_sizes = np.bincount(assignments)

    report = {
        "dataset": "zsre",
        "n_records": len(records),
        "batch_size": batch_size,
        "n_batches": len(records) // batch_size,
        "n_unique_relations": len(set(r["requested_rewrite"]["relation_id"] for r in records)),
        "semantic": {
            "clustered_mean_concentration": float(np.mean(clust_conc)),
            "dispersed_mean_concentration": float(np.mean(disp_conc)),
            "concentration_ratio": float(np.mean(clust_conc) / max(np.mean(disp_conc), 1e-10)),
        },
        "key_geometry": {
            "n_clusters": int(assignments.max() + 1),
            "cluster_sizes_mean": float(cluster_sizes.mean()),
            "cluster_sizes_std": float(cluster_sizes.std()),
            "key_clustered": {
                "mean_within_batch_cosine": float(np.mean(kc_cosines)),
                "mean_clusters_per_batch": float(np.mean(kc_diversity)),
            },
            "key_dispersed": {
                "mean_within_batch_cosine": float(np.mean(kd_cosines)),
                "mean_clusters_per_batch": float(np.mean(kd_diversity)),
            },
            "cosine_ratio": float(np.mean(kc_cosines) / max(np.mean(kd_cosines), 1e-10)),
        },
    }
    return report


# ─── Main ────────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(
        description="Generate matched ordering datasets for ZsRE"
    )
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--stream_length", type=int, default=5000)
    parser.add_argument("--batch_size", type=int, default=100)
    parser.add_argument("--n_clusters", type=int, default=50,
                        help="Number of spherical k-means clusters for key orderings")
    parser.add_argument("--keys_path", type=str, default=None,
                        help="Path to precomputed ZsRE key vectors (.npz)")
    parser.add_argument("--data_dir", type=str, default=None,
                        help="Directory containing zsre_mend_eval.json")
    parser.add_argument("--output_dir", type=str, default="results/matched_ordering_zsre",
                        help="Base output directory")
    parser.add_argument("--fixed_batch", action="store_true",
                        help="Generate fixed-batch orderings (constant batch membership, "
                             "permuted batch sequence)")
    parser.add_argument("--base_ordering", type=str, default="key_clustered",
                        help="Existing ordering to derive batch membership from")
    parser.add_argument("--n_random_perms", type=int, default=3,
                        help="Number of random batch permutations (fixed_batch mode)")
    args = parser.parse_args()

    # Resolve data directory
    if args.data_dir:
        data_dir = Path(args.data_dir)
    else:
        candidates = [
            PROJECT_ROOT / "data" / "dsets",
            PROJECT_ROOT / "vendor" / "AlphaEdit" / "data",
            *([] if not os.environ.get("DSETS_ROOT") else [Path(os.environ["DSETS_ROOT"])]),
        ]
        data_dir = None
        for c in candidates:
            if (c / "zsre_mend_eval.json").exists():
                data_dir = c
                break
        if data_dir is None:
            print("ERROR: Cannot find zsre_mend_eval.json")
            print("  Tried:", [str(c) for c in candidates])
            print("  Run: bash scripts/download_datasets.sh")
            sys.exit(1)

    # Resolve keys path
    if args.keys_path:
        keys_path = Path(args.keys_path)
    else:
        keys_path = PROJECT_ROOT / "results" / "key_vectors" / "zsre" / f"keys_seed{args.seed}_layer6.npz"
    if not keys_path.is_absolute():
        keys_path = PROJECT_ROOT / keys_path

    out_base = Path(args.output_dir)
    if not out_base.is_absolute():
        out_base = PROJECT_ROOT / args.output_dir
    ord_dir = out_base / "orderings"
    diag_dir = out_base / "diagnostics"
    ord_dir.mkdir(parents=True, exist_ok=True)
    diag_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*70}")
    print("ZsRE Matched Ordering Generator")
    print(f"  Seed:          {args.seed}")
    print(f"  Stream length: {args.stream_length}")
    print(f"  Batch size:    {args.batch_size}")
    print(f"  Key clusters:  {args.n_clusters}")
    print(f"  Keys:          {keys_path}")
    print(f"  Data dir:      {data_dir}")
    print(f"  Output:        {out_base}")
    print(f"{'='*70}")

    # ── Step 1: Load data and select records ──────────────────────────────────

    print(f"\n  Loading ZsRE dataset...")
    all_data = load_zsre(data_dir)
    print(f"  Total ZsRE records: {len(all_data)}")

    print(f"\n  Selecting {args.stream_length} unique, non-conflicting records...")
    rng = random.Random(args.seed)
    records = select_clean_pool(all_data, rng, args.stream_length)
    print(f"  Selected: {len(records)} records")
    print(f"  Unique subjects: {len(set(r['requested_rewrite']['subject'] for r in records))}")

    # Count unique derived relations
    relations = set(r["requested_rewrite"]["relation_id"] for r in records)
    print(f"  Unique derived relations: {len(relations)}")

    # ── Step 2: Semantic orderings ────────────────────────────────────────────

    print(f"\n  Generating semantic orderings...")
    rng_clust = random.Random(args.seed + 1000)
    rng_disp = random.Random(args.seed + 2000)
    clustered = create_clustered_ordering(records, rng_clust)
    dispersed = create_dispersed_ordering(records, rng_disp)
    print(f"    Clustered: done")
    print(f"    Dispersed: done")

    # ── Step 3: Key-geometry orderings ────────────────────────────────────────

    print(f"\n  Loading keys from {keys_path.name}...")
    if not keys_path.exists():
        print(f"  ERROR: Keys file not found at {keys_path}")
        print(f"  Run key extraction first:")
        print(f"    bash scripts/run_zsre_key_extraction.sh {args.seed}")
        sys.exit(1)

    npz = np.load(keys_path)
    all_keys = npz["keys"]
    all_case_ids = npz["case_ids"].tolist()
    print(f"  Full key matrix: {all_keys.shape}")

    # Filter keys to selected records
    key_idx_by_id = {cid: i for i, cid in enumerate(all_case_ids)}
    record_case_ids = [r["case_id"] for r in records]
    missing = [cid for cid in record_case_ids if cid not in key_idx_by_id]
    if missing:
        print(f"  WARNING: {len(missing)} records have no precomputed key — dropping them")
        records = [r for r in records if r["case_id"] in key_idx_by_id]
        record_case_ids = [r["case_id"] for r in records]
        rng_clust = random.Random(args.seed + 1000)
        rng_disp = random.Random(args.seed + 2000)
        clustered = create_clustered_ordering(records, rng_clust)
        dispersed = create_dispersed_ordering(records, rng_disp)

    key_indices = [key_idx_by_id[cid] for cid in record_case_ids]
    keys = all_keys[key_indices]
    case_id_to_idx = {records[i]["case_id"]: i for i in range(len(records))}
    print(f"  Selected keys: {keys.shape}")

    # ── Fixed-batch mode ─────────────────────────────────────────────────
    if args.fixed_batch:
        from generate_orderings import (
            batches_from_ordering, compute_batch_centroids,
            order_batches_high_exposure, order_batches_low_exposure,
            order_batches_random, validate_fixed_batch_orderings,
        )
        print(f"\n  === FIXED-BATCH MODE (zsRE) ===")

        base_path = ord_dir / f"{args.base_ordering}_seed{args.seed}.json"
        if base_path.exists():
            print(f"  Using batch decomposition from: {args.base_ordering}")
            with open(base_path) as f:
                base_ordering = json.load(f)
            batches = batches_from_ordering(base_ordering, args.batch_size)
        else:
            print(f"  Base ordering not found, using random assignment")
            rng_fb = random.Random(args.seed + 5000)
            from generate_orderings import assign_fixed_batches
            batches = assign_fixed_batches(records, args.batch_size, rng_fb)
        print(f"  {len(batches)} fixed batches of {args.batch_size}")

        centroids = compute_batch_centroids(batches, keys, case_id_to_idx)
        sim = centroids @ centroids.T
        np.fill_diagonal(sim, 0)
        print(f"  Inter-batch cosine: mean={sim[sim!=0].mean():.4f}")

        high_exp = order_batches_high_exposure(batches, centroids, random.Random(args.seed + 6000))
        low_exp = order_batches_low_exposure(batches, centroids, random.Random(args.seed + 7000))
        rand_perms = order_batches_random(batches, args.n_random_perms, random.Random(args.seed + 8000))

        all_orderings = {"fb_high_exposure": high_exp, "fb_low_exposure": low_exp}
        for i, rp in enumerate(rand_perms):
            all_orderings[f"fb_random{i}"] = rp

        fb_report = validate_fixed_batch_orderings(
            batches, all_orderings, keys, case_id_to_idx, args.batch_size)
        assert fb_report["batch_membership_preserved"], "Batch membership NOT preserved!"

        wb = list(fb_report["orderings"].values())[0]["mean_within_batch_cosine"]
        print(f"    Within-batch cosine: {wb:.6f} (identical)")
        hi_fe = fb_report["orderings"]["fb_high_exposure"]["mean_future_exposure"]
        lo_fe = fb_report["orderings"]["fb_low_exposure"]["mean_future_exposure"]
        print(f"    High exposure: {hi_fe:.4f}, Low: {lo_fe:.4f}, Ratio: {hi_fe/max(lo_fe,1e-10):.2f}x")

        for name, ordering in all_orderings.items():
            path = ord_dir / f"{name}_seed{args.seed}.json"
            with open(path, "w") as f:
                json.dump(ordering, f)
            print(f"    {path.name}")

        import json as _json
        ba = {"seed": args.seed, "batch_size": args.batch_size, "n_batches": len(batches),
              "batches": [[r["case_id"] for r in b] for b in batches]}
        ba_path = diag_dir / f"fixed_batch_assignment_seed{args.seed}.json"
        with open(ba_path, "w") as f:
            _json.dump(ba, f, indent=2)

        fb_report["seed"] = args.seed
        rpt_path = diag_dir / f"fixed_batch_report_seed{args.seed}.json"
        with open(rpt_path, "w") as f:
            _json.dump(fb_report, f, indent=2)

        print(f"\n  Fixed-batch zsRE orderings saved.")
        return

    print(f"\n  Running spherical k-means (k={args.n_clusters})...")
    assignments = spherical_kmeans(keys, args.n_clusters, max_iter=100, seed=args.seed)

    cluster_sizes = np.bincount(assignments)
    print(f"  Clusters: {len(cluster_sizes)} non-empty")
    print(f"  Cluster sizes: mean={cluster_sizes.mean():.1f}, "
          f"min={cluster_sizes.min()}, max={cluster_sizes.max()}, "
          f"std={cluster_sizes.std():.1f}")

    rng_kc = random.Random(args.seed + 3000)
    rng_kd = random.Random(args.seed + 4000)
    print(f"\n  Generating key-clustered ordering...")
    key_clustered = create_key_clustered_ordering(records, assignments, rng_kc)
    print(f"  Generating key-dispersed ordering...")
    key_dispersed = create_key_dispersed_ordering(records, assignments, rng_kd)

    # ── Step 4: Validate ──────────────────────────────────────────────────────

    print(f"\n  Validating all orderings...")
    report = validate_and_compute_metrics(
        records, clustered, dispersed, key_clustered, key_dispersed,
        keys, assignments, args.batch_size,
    )

    print(f"\n  Semantic metrics:")
    print(f"    Clustered concentration: {report['semantic']['clustered_mean_concentration']:.4f}")
    print(f"    Dispersed concentration: {report['semantic']['dispersed_mean_concentration']:.4f}")
    print(f"    Ratio: {report['semantic']['concentration_ratio']:.1f}x")

    print(f"\n  Key-geometry metrics:")
    kg = report["key_geometry"]
    print(f"    Key-clustered: {kg['key_clustered']['mean_within_batch_cosine']:.4f} "
          f"mean cosine ({kg['key_clustered']['mean_clusters_per_batch']:.1f} clusters/batch)")
    print(f"    Key-dispersed: {kg['key_dispersed']['mean_within_batch_cosine']:.4f} "
          f"mean cosine ({kg['key_dispersed']['mean_clusters_per_batch']:.1f} clusters/batch)")
    print(f"    Cosine ratio: {kg['cosine_ratio']:.2f}x")

    # ── Step 5: Save ──────────────────────────────────────────────────────────

    print(f"\n  Saving outputs...")

    for name, ordering in [("clustered", clustered), ("dispersed", dispersed),
                           ("key_clustered", key_clustered), ("key_dispersed", key_dispersed)]:
        path = ord_dir / f"{name}_seed{args.seed}.json"
        with open(path, "w") as f:
            json.dump(ordering, f)
        print(f"    {path.name}")

    report["seed"] = args.seed
    report_path = diag_dir / f"ordering_report_seed{args.seed}.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"    {report_path.name}")

    # ── Done ──────────────────────────────────────────────────────────────────

    print(f"\n{'='*70}")
    print("Done. Run experiments with:")
    print(f"  bash scripts/run_zsre_ordering_experiment.sh {args.seed} AlphaEdit key_clustered")
    print(f"  bash scripts/run_zsre_ordering_experiment.sh {args.seed} AlphaEdit key_dispersed")
    print(f"  bash scripts/run_zsre_ordering_experiment.sh {args.seed} MEMIT-Seq-lp1.0-ld0.0-cache0 key_clustered")
    print(f"  bash scripts/run_zsre_ordering_experiment.sh {args.seed} MEMIT-Seq-lp1.0-ld0.0-cache0 key_dispersed")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()

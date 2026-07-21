#!/usr/bin/env python3
"""
Matched Ordering Key Geometry Diagnostics — GPU required.

Verifies that relation-based ordering actually translates to MEMIT key-space
geometry differences. Computes keys for all 5000 edits (once, since facts
are identical) then evaluates geometry under both orderings.

Diagnostics computed:
  1. Within-batch key cosine similarity (clustered should be higher)
  2. Adjacent-batch key cosine similarity (clustered should be higher)
  3. Effective rank within each batch
  4. Effective rank of first-t-edit prefix cache
  5. Condition number of prefix caches over time
  6. Key norm by 1K cohort (should be matched)
  7. Global key-set spectrum (should be identical)
  8. Maximum similarity to subsequent edits (future-key exposure)
  9. Spectral concentration over time

Usage:
    uv run python analysis/matched_ordering_key_geometry.py \
        --seed 42 --layer 6

    # Quick test with 500 edits:
    uv run python analysis/matched_ordering_key_geometry.py \
        --seed 42 --max_cases 500
"""

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src" / "util"))
sys.path.insert(0, str(PROJECT_ROOT / "src" / "mechanism"))


# ─── Key Extraction (reuses compute_keys.py logic) ──────────────────────────


def extract_all_keys(
    model, tokenizer, records: list, layer: int, device: str = "cuda"
) -> np.ndarray:
    """Extract key vectors for all records. Returns (N, hidden_dim) array."""
    from compute_keys import KeyExtractor

    extractor = KeyExtractor(model, tokenizer, layer)
    keys = []
    failed = 0
    t0 = time.time()

    for i, record in enumerate(records):
        rw = record["requested_rewrite"]
        prompt = rw["prompt"]
        subject = rw["subject"]

        key = extractor.extract_key(prompt, subject)
        if key is not None:
            keys.append(key)
        else:
            # Use zero vector as placeholder (will be masked in analysis)
            keys.append(np.zeros_like(keys[-1]) if keys else None)
            failed += 1

        if (i + 1) % 500 == 0:
            elapsed = time.time() - t0
            rate = (i + 1) / elapsed
            print(f"    [{i+1}/{len(records)}] {rate:.1f} keys/sec ({failed} failed)")

    # Remove any leading None entries
    keys = [k for k in keys if k is not None]
    elapsed = time.time() - t0
    print(f"    Extracted {len(keys)} keys in {elapsed:.1f}s ({failed} failed)")
    return np.stack(keys, axis=0)


# ─── Geometry Computations ──────────────────────────────────────────────────


def within_batch_cosine(keys: np.ndarray, ordering: list, batch_size: int) -> list:
    """Mean pairwise cosine similarity within each batch."""
    case_id_to_idx = {r["case_id"]: i for i, r in enumerate(ordering)}
    n_batches = len(ordering) // batch_size
    similarities = []

    for b in range(n_batches):
        batch_records = ordering[b * batch_size: (b + 1) * batch_size]
        indices = [case_id_to_idx[r["case_id"]] for r in batch_records
                   if r["case_id"] in case_id_to_idx]
        if len(indices) < 2:
            similarities.append(0.0)
            continue

        batch_keys = keys[indices]
        # Normalize
        norms = np.linalg.norm(batch_keys, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-8)
        normed = batch_keys / norms
        # Cosine matrix
        cos_matrix = normed @ normed.T
        # Mean of upper triangle (excluding diagonal)
        n = len(indices)
        mask = np.triu(np.ones((n, n), dtype=bool), k=1)
        mean_cos = cos_matrix[mask].mean()
        similarities.append(float(mean_cos))

    return similarities


def adjacent_batch_cosine(keys: np.ndarray, ordering: list, batch_size: int) -> list:
    """Mean cosine similarity between adjacent batches."""
    case_id_to_idx = {r["case_id"]: i for i, r in enumerate(ordering)}
    n_batches = len(ordering) // batch_size
    adj_sims = []

    for b in range(n_batches - 1):
        batch_a = ordering[b * batch_size: (b + 1) * batch_size]
        batch_b = ordering[(b + 1) * batch_size: (b + 2) * batch_size]

        idx_a = [case_id_to_idx[r["case_id"]] for r in batch_a if r["case_id"] in case_id_to_idx]
        idx_b = [case_id_to_idx[r["case_id"]] for r in batch_b if r["case_id"] in case_id_to_idx]

        if not idx_a or not idx_b:
            adj_sims.append(0.0)
            continue

        keys_a = keys[idx_a]
        keys_b = keys[idx_b]

        # Normalize
        na = keys_a / np.maximum(np.linalg.norm(keys_a, axis=1, keepdims=True), 1e-8)
        nb = keys_b / np.maximum(np.linalg.norm(keys_b, axis=1, keepdims=True), 1e-8)

        # Cross-batch cosine
        cross = na @ nb.T
        adj_sims.append(float(cross.mean()))

    return adj_sims


def prefix_cache_spectrum(keys: np.ndarray, ordering: list, checkpoints: list) -> dict:
    """Compute cache spectrum at checkpoints (effective rank, condition, top_sv_share)."""
    case_id_to_idx = {r["case_id"]: i for i, r in enumerate(ordering)}
    results = {}

    for t in checkpoints:
        prefix_records = ordering[:t]
        indices = [case_id_to_idx[r["case_id"]] for r in prefix_records
                   if r["case_id"] in case_id_to_idx]
        if not indices:
            continue

        prefix_keys = keys[indices]  # (t, d)
        # Cache = K @ K.T (same as AlphaEdit's cache_c accumulation)
        cache = prefix_keys.T @ prefix_keys  # (d, d)

        # Eigendecomposition
        eigvals = np.linalg.eigvalsh(cache)
        eigvals = np.sort(eigvals)[::-1]
        eigvals = np.maximum(eigvals, 0)
        svs = np.sqrt(eigvals)

        # Metrics
        svs_pos = svs[svs > 1e-10]
        numerical_rank = int((svs > 1e-5).sum())

        if len(svs_pos) > 0:
            probs = svs_pos / svs_pos.sum()
            effective_rank = float(np.exp(-np.sum(probs * np.log(probs + 1e-10))))
            top_sv_share = float(svs_pos[0] / svs_pos.sum())
            condition = float(svs_pos[0] / svs_pos[-1]) if len(svs_pos) >= 2 else float("inf")
        else:
            effective_rank = 0.0
            top_sv_share = 1.0
            condition = float("inf")

        results[t] = {
            "numerical_rank": numerical_rank,
            "effective_rank": round(effective_rank, 2),
            "top_sv_share": round(top_sv_share, 6),
            "condition": round(condition, 2) if condition != float("inf") else "inf",
        }

    return results


def batch_effective_rank(keys: np.ndarray, ordering: list, batch_size: int) -> list:
    """Effective rank of keys within each batch."""
    case_id_to_idx = {r["case_id"]: i for i, r in enumerate(ordering)}
    n_batches = len(ordering) // batch_size
    ranks = []

    for b in range(n_batches):
        batch_records = ordering[b * batch_size: (b + 1) * batch_size]
        indices = [case_id_to_idx[r["case_id"]] for r in batch_records
                   if r["case_id"] in case_id_to_idx]
        if len(indices) < 2:
            ranks.append(0.0)
            continue

        batch_keys = keys[indices]
        # SVD of the batch key matrix
        svs = np.linalg.svd(batch_keys, compute_uv=False)
        svs_pos = svs[svs > 1e-10]
        if len(svs_pos) > 0:
            probs = svs_pos / svs_pos.sum()
            er = float(np.exp(-np.sum(probs * np.log(probs + 1e-10))))
        else:
            er = 0.0
        ranks.append(er)

    return ranks


def key_norms_by_cohort(keys: np.ndarray, ordering: list, cohort_size: int = 1000) -> list:
    """Mean key norm per cohort."""
    case_id_to_idx = {r["case_id"]: i for i, r in enumerate(ordering)}
    n_cohorts = len(ordering) // cohort_size
    norms = []

    for c in range(n_cohorts):
        cohort = ordering[c * cohort_size: (c + 1) * cohort_size]
        indices = [case_id_to_idx[r["case_id"]] for r in cohort if r["case_id"] in case_id_to_idx]
        cohort_keys = keys[indices]
        mean_norm = float(np.linalg.norm(cohort_keys, axis=1).mean())
        norms.append(mean_norm)

    return norms


def future_key_exposure(keys: np.ndarray, ordering: list, batch_size: int, lookahead: int = 10) -> list:
    """For each batch, max cosine sim to any key in the next `lookahead` batches."""
    case_id_to_idx = {r["case_id"]: i for i, r in enumerate(ordering)}
    n_batches = len(ordering) // batch_size
    exposures = []

    for b in range(n_batches):
        batch_records = ordering[b * batch_size: (b + 1) * batch_size]
        future_start = (b + 1) * batch_size
        future_end = min((b + 1 + lookahead) * batch_size, len(ordering))

        if future_end <= future_start:
            exposures.append(0.0)
            continue

        future_records = ordering[future_start:future_end]
        idx_curr = [case_id_to_idx[r["case_id"]] for r in batch_records if r["case_id"] in case_id_to_idx]
        idx_future = [case_id_to_idx[r["case_id"]] for r in future_records if r["case_id"] in case_id_to_idx]

        if not idx_curr or not idx_future:
            exposures.append(0.0)
            continue

        kc = keys[idx_curr]
        kf = keys[idx_future]
        nc = kc / np.maximum(np.linalg.norm(kc, axis=1, keepdims=True), 1e-8)
        nf = kf / np.maximum(np.linalg.norm(kf, axis=1, keepdims=True), 1e-8)

        # Max cosine of any current key to any future key
        cross = nc @ nf.T
        max_sim = float(cross.max())
        exposures.append(max_sim)

    return exposures


# ─── Main ───────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(
        description="Key-space geometry diagnostics for matched ordering"
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--layer", type=int, default=6,
                        help="Layer for key extraction (default: 6, middle of AlphaEdit edit layers)")
    parser.add_argument("--stream_dir", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--max_cases", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=100)
    parser.add_argument("--model", type=str,
                        default=os.environ.get("MODEL_NAME", "meta-llama/Meta-Llama-3-8B-Instruct"))
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    # Resolve paths
    if args.stream_dir:
        stream_dir = Path(args.stream_dir)
    else:
        stream_dir = PROJECT_ROOT / "results" / "matched_ordering"

    if args.output_dir:
        out_dir = Path(args.output_dir)
    else:
        out_dir = stream_dir / "key_geometry"
    out_dir.mkdir(parents=True, exist_ok=True)

    clust_path = stream_dir / f"clustered_seed{args.seed}.json"
    disp_path = stream_dir / f"dispersed_seed{args.seed}.json"

    if not clust_path.exists() or not disp_path.exists():
        print(f"ERROR: Stream files not found in {stream_dir}")
        sys.exit(1)

    # Load streams
    print(f"\n{'='*70}")
    print("Matched Ordering Key Geometry Diagnostics")
    print(f"  Seed: {args.seed}, Layer: {args.layer}")
    print(f"  Model: {args.model}")
    print(f"{'='*70}")

    print("\n  Loading streams...")
    with open(clust_path) as f:
        clustered = json.load(f)
    with open(disp_path) as f:
        dispersed = json.load(f)

    if args.max_cases:
        clustered = clustered[:args.max_cases]
        dispersed = dispersed[:args.max_cases]

    print(f"  Records: {len(clustered)}")

    # Load model
    print(f"\n  Loading model: {args.model}")
    from model_download import resolve_model_path
    model_id = resolve_model_path(args.model)

    from transformers import AutoModelForCausalLM, AutoTokenizer
    token = os.environ.get("HF_TOKEN")
    tokenizer = AutoTokenizer.from_pretrained(model_id, token=token)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_id, token=token, torch_dtype=torch.float16, device_map=args.device
    )
    model.eval()
    print(f"  Model loaded on {args.device}")

    # Extract keys (only once — same facts in both orderings)
    # Use dispersed ordering as the canonical ordering for key extraction
    # (order doesn't matter for individual key computation)
    print(f"\n  Extracting keys (layer {args.layer})...")
    all_keys = extract_all_keys(model, tokenizer, dispersed, args.layer, args.device)
    print(f"  Keys shape: {all_keys.shape}")

    # Free model memory
    del model
    torch.cuda.empty_cache()

    # ─── Compute diagnostics ─────────────────────────────────────────────────

    print(f"\n  Computing diagnostics...")
    batch_size = args.batch_size
    n_records = len(clustered)

    # 1. Within-batch cosine similarity
    print("    Within-batch cosine similarity...")
    clust_within = within_batch_cosine(all_keys, clustered, batch_size)
    disp_within = within_batch_cosine(all_keys, dispersed, batch_size)

    # 2. Adjacent-batch cosine
    print("    Adjacent-batch cosine similarity...")
    clust_adjacent = adjacent_batch_cosine(all_keys, clustered, batch_size)
    disp_adjacent = adjacent_batch_cosine(all_keys, dispersed, batch_size)

    # 3. Batch effective rank
    print("    Per-batch effective rank...")
    clust_batch_er = batch_effective_rank(all_keys, clustered, batch_size)
    disp_batch_er = batch_effective_rank(all_keys, dispersed, batch_size)

    # 4. Prefix cache spectrum at checkpoints
    checkpoints = [t for t in range(1000, n_records + 1, 1000)]
    print(f"    Prefix cache spectrum at {len(checkpoints)} checkpoints...")
    clust_prefix = prefix_cache_spectrum(all_keys, clustered, checkpoints)
    disp_prefix = prefix_cache_spectrum(all_keys, dispersed, checkpoints)

    # 5. Key norms by cohort
    print("    Key norms by cohort...")
    clust_norms = key_norms_by_cohort(all_keys, clustered)
    disp_norms = key_norms_by_cohort(all_keys, dispersed)

    # 6. Global spectrum (same facts → should be identical)
    print("    Global key-set spectrum...")
    global_svs = np.linalg.svd(all_keys, compute_uv=False)
    global_svs_pos = global_svs[global_svs > 1e-10]
    global_probs = global_svs_pos / global_svs_pos.sum()
    global_er = float(np.exp(-np.sum(global_probs * np.log(global_probs + 1e-10))))
    global_numerical_rank = int((global_svs > 1e-5).sum())
    global_top_sv_share = float(global_svs_pos[0] / global_svs_pos.sum())

    # 7. Future-key exposure
    print("    Future-key exposure (lookahead=10 batches)...")
    clust_future = future_key_exposure(all_keys, clustered, batch_size, lookahead=10)
    disp_future = future_key_exposure(all_keys, dispersed, batch_size, lookahead=10)

    # ─── Summary ─────────────────────────────────────────────────────────────

    print(f"\n{'='*70}")
    print("KEY GEOMETRY RESULTS")
    print(f"{'='*70}")

    print(f"\n  {'Diagnostic':<35} {'Clustered':>12} {'Dispersed':>12} {'Ratio':>8}")
    print(f"  {'─'*70}")

    mc = np.mean(clust_within)
    md = np.mean(disp_within)
    print(f"  {'Within-batch cosine (mean)':<35} {mc:>12.4f} {md:>12.4f} {mc/max(md,1e-10):>8.2f}x")

    mc = np.mean(clust_adjacent)
    md = np.mean(disp_adjacent)
    print(f"  {'Adjacent-batch cosine (mean)':<35} {mc:>12.4f} {md:>12.4f} {mc/max(md,1e-10):>8.2f}x")

    mc = np.mean(clust_batch_er)
    md = np.mean(disp_batch_er)
    print(f"  {'Batch effective rank (mean)':<35} {mc:>12.2f} {md:>12.2f} {mc/max(md,1e-10):>8.2f}x")

    mc = np.mean(clust_future)
    md = np.mean(disp_future)
    print(f"  {'Future-key exposure (mean max)':<35} {mc:>12.4f} {md:>12.4f} {mc/max(md,1e-10):>8.2f}x")

    mc = np.mean(clust_norms)
    md = np.mean(disp_norms)
    print(f"  {'Key norm (mean across cohorts)':<35} {mc:>12.2f} {md:>12.2f} {mc/max(md,1e-10):>8.2f}x")

    print(f"\n  Global spectrum (identical facts → should match):")
    print(f"    Effective rank:    {global_er:.2f}")
    print(f"    Numerical rank:    {global_numerical_rank}")
    print(f"    Top SV share:      {global_top_sv_share:.6f}")

    print(f"\n  Prefix cache condition number over time:")
    print(f"    {'Edits':<8} {'Clust EffRank':>14} {'Disp EffRank':>14} {'Clust Cond':>12} {'Disp Cond':>12}")
    for t in checkpoints:
        if t in clust_prefix and t in disp_prefix:
            cp = clust_prefix[t]
            dp = disp_prefix[t]
            cc = cp['condition'] if cp['condition'] != 'inf' else '∞'
            dc = dp['condition'] if dp['condition'] != 'inf' else '∞'
            print(f"    {t:<8} {cp['effective_rank']:>14.2f} {dp['effective_rank']:>14.2f} "
                  f"{str(cc):>12} {str(dc):>12}")

    # ─── Assessment ──────────────────────────────────────────────────────────

    within_ratio = np.mean(clust_within) / max(np.mean(disp_within), 1e-10)
    print(f"\n  {'='*70}")
    print("  ASSESSMENT:")
    if within_ratio > 1.5:
        print(f"    ✓ Within-batch cosine ratio = {within_ratio:.2f}x — relation grouping "
              f"produces substantial key-space clustering.")
    elif within_ratio > 1.1:
        print(f"    ~ Within-batch cosine ratio = {within_ratio:.2f}x — moderate key-space "
              f"effect. Relation grouping has some key-space footprint.")
    else:
        print(f"    ⚠ Within-batch cosine ratio = {within_ratio:.2f}x — relation grouping "
              f"does NOT produce key-space clustering. The ordering manipulation "
              f"may not create meaningful geometric differences for the editor.")

    norm_range = max(clust_norms) - min(clust_norms)
    if norm_range > 5.0:
        print(f"    ⚠ Key norm varies substantially across clustered cohorts "
              f"(range={norm_range:.1f}). Age-conditioned difficulty confound possible.")
    else:
        print(f"    ✓ Key norms balanced across cohorts (range={norm_range:.1f}).")
    print(f"  {'='*70}")

    # ─── Save ────────────────────────────────────────────────────────────────

    results = {
        "seed": args.seed,
        "layer": args.layer,
        "n_records": n_records,
        "batch_size": batch_size,
        "global_spectrum": {
            "effective_rank": global_er,
            "numerical_rank": global_numerical_rank,
            "top_sv_share": global_top_sv_share,
        },
        "within_batch_cosine": {
            "clustered_mean": float(np.mean(clust_within)),
            "dispersed_mean": float(np.mean(disp_within)),
            "ratio": float(within_ratio),
            "clustered": clust_within,
            "dispersed": disp_within,
        },
        "adjacent_batch_cosine": {
            "clustered_mean": float(np.mean(clust_adjacent)),
            "dispersed_mean": float(np.mean(disp_adjacent)),
            "clustered": clust_adjacent,
            "dispersed": disp_adjacent,
        },
        "batch_effective_rank": {
            "clustered_mean": float(np.mean(clust_batch_er)),
            "dispersed_mean": float(np.mean(disp_batch_er)),
            "clustered": clust_batch_er,
            "dispersed": disp_batch_er,
        },
        "prefix_cache_spectrum": {
            "clustered": {str(k): v for k, v in clust_prefix.items()},
            "dispersed": {str(k): v for k, v in disp_prefix.items()},
        },
        "key_norms_by_cohort": {
            "clustered": clust_norms,
            "dispersed": disp_norms,
        },
        "future_key_exposure": {
            "clustered_mean": float(np.mean(clust_future)),
            "dispersed_mean": float(np.mean(disp_future)),
            "clustered": clust_future,
            "dispersed": disp_future,
        },
    }

    out_path = out_dir / f"key_geometry_seed{args.seed}_layer{args.layer}.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  Saved: {out_path}")

    # Save keys for reuse
    keys_path = out_dir / f"keys_seed{args.seed}_layer{args.layer}.npz"
    case_ids = np.array([r["case_id"] for r in dispersed[:len(all_keys)]], dtype=np.int32)
    np.savez_compressed(keys_path, keys=all_keys, case_ids=case_ids, layer=np.array(args.layer))
    print(f"  Keys saved: {keys_path}")


if __name__ == "__main__":
    main()

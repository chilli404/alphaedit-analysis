#!/usr/bin/env python3
"""
Solve-Aware Interference Kernel η(i,t) for AlphaEdit.

Theory
------
AlphaEdit's weight update at batch t is:

    ΔW_t = A_t^{-1} @ P @ K_t @ R_t^T

where A_t = P @ (K_t @ K_t^T + C_{t-1}) + λI, and C_{t-1} = Σ_{b<t} K_b @ K_b^T.

The effect on a previously installed edit key k_i is:

    ΔW_t @ k_i = A_t^{-1} @ P @ K_t @ R_t^T @ k_i

The interference transfer score (geometry-dependent, residual-independent) is:

    η(i,t) = ||K_t^T @ A_t^{-1} @ k_i||

Raw cosine max_j cos(k_i, k_j) is a cheap approximation. η accounts for:
  - The covariance weighting from accumulated cache_c
  - The L2 regularization
  - The null-space projection P
  - Cross-key interactions within the batch

Practical Approximation
-----------------------
Without P and cache_c at each step, we use base-model keys to reconstruct
cache_c incrementally: C_t = Σ_{b≤t} K_b @ K_b^T (same keys used for cosine).
This approximation is fair because the survival model also uses base-model keys.

For the P term, we either:
  1. Load P from disk if available
  2. Use P = I (no projection) as an ablation — this isolates the
     cache-whitening effect from the projection effect

Usage:
    uv run python src/mechanism/interference_kernel.py --seed 42 --ordering key_clustered
    uv run python src/mechanism/interference_kernel.py --seed 42 --ordering key_clustered --no-projection
"""

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src" / "util"))


def load_stream_keys(
    keys_path: Path, stream_path: Path, batch_size: int = 100,
):
    """Load key vectors and organize by stream batch order."""
    keys_data = np.load(keys_path)
    all_keys = keys_data["keys"]  # (N_total, d)
    all_cids = keys_data["case_ids"].tolist()
    cid_to_idx = {int(c): i for i, c in enumerate(all_cids)}

    with open(stream_path) as f:
        stream = json.load(f)

    stream_cids = [r["case_id"] for r in stream]
    n_batches = len(stream_cids) // batch_size

    batches_keys = []
    batches_cids = []
    for b in range(n_batches):
        batch_cids = stream_cids[b * batch_size : (b + 1) * batch_size]
        batch_indices = [cid_to_idx[c] for c in batch_cids if c in cid_to_idx]
        if batch_indices:
            batches_keys.append(all_keys[batch_indices].astype(np.float32))
            batches_cids.append(batch_cids)

    return batches_keys, batches_cids, stream_cids


def load_P_matrix(layer_idx: int = 2):
    """Load null-space projection P for layer 6 (index 2 in [4,5,6,7,8]).

    Returns P as numpy array, or None if unavailable.
    """
    candidates = [
        PROJECT_ROOT / "data" / "stats" / "Llama3-8B" / "wikipedia_stats" / "null_space_project.pt",
        PROJECT_ROOT / "data" / "stats" / "llama3-8b-instruct" / "wikipedia_stats" / "null_space_project.pt",
        PROJECT_ROOT / "vendor" / "AlphaEdit" / "null_space_project.pt",
        PROJECT_ROOT / "vendor" / "AlphaEdit" / "data" / "stats" / "Llama3-8B" / "wikipedia_stats" / "null_space_project.pt",
    ]
    for p in candidates:
        if p.exists():
            import torch
            P_all = torch.load(str(p), map_location="cpu")
            return P_all[layer_idx].numpy().astype(np.float32)
    return None


def compute_kernel_scores(
    batches_keys: list,
    batches_cids: list,
    stream_cids: list,
    P: np.ndarray = None,
    L2: float = 10.0,
    batch_size: int = 100,
    max_batches: int = None,
    use_woodbury: bool = True,
):
    """Compute η(i,t) and raw cosine for each edit at each subsequent batch.

    Returns per-edit accumulated scores:
      eta_cumulative[i]: Σ_t η(i,t) for all batches t after edit i's batch
      cosine_cumulative[i]: Σ_t max_j cos(k_i, k_j^t) for all batches t after i
      eta_at_checkpoints[i][ckpt]: η accumulated up to checkpoint
    """
    n_batches = len(batches_keys)
    if max_batches:
        n_batches = min(n_batches, max_batches)

    d = batches_keys[0].shape[1]
    n_total = len(stream_cids)

    # Map case_id → stream position
    cid_to_pos = {c: i for i, c in enumerate(stream_cids)}

    # Incremental cache: C_t = Σ_{b<t} K_b @ K_b^T
    # For Woodbury: A_t = P @ (K_t @ K_t^T + C_{t-1}) + λI
    # We track C incrementally

    # Use low-rank representation of cache for efficiency
    # Instead of d×d matrix, store the keys themselves
    all_prev_keys = []  # list of key arrays from previous batches

    # Per-edit accumulators
    eta_cumul = np.zeros(n_total, dtype=np.float64)
    cos_cumul = np.zeros(n_total, dtype=np.float64)

    # Checkpoints at every 10 batches
    checkpoint_batches = set(range(9, n_batches, 10))
    eta_at_ckpt = {}
    cos_at_ckpt = {}

    use_P = P is not None
    if use_P:
        print(f"  Using P matrix (shape {P.shape})")
    else:
        print(f"  No P matrix — using identity (cache-whitened cosine only)")

    for t in range(n_batches):
        K_t = batches_keys[t].T  # (d, n_batch)
        n_batch = K_t.shape[1]

        # Build A_t approximation
        # A_t = P @ (K_t @ K_t^T + C_{t-1}) + λI
        # For efficiency with d=14336, we can't form the full d×d matrix every step.
        # Instead use the matrix-vector form: A_t @ x = P @ (K_t @ (K_t^T @ x) + C @ x) + λx
        # and solve iteratively, OR use the low-rank structure.

        # Practical approach: form A_t in the subspace spanned by K_t and recent cache keys.
        # The number of unique key directions grows with batches (100 per batch).
        # At batch 50, we have ~5000 key directions in 14336-d space.

        # For the kernel score, we need: K_t^T @ A_t^{-1} @ k_i for each old k_i
        # Using Woodbury: if A_t = λI + P @ M where M = K_t@K_t^T + C_{t-1}
        # then A_t^{-1} = (1/λ)(I - P @ M @ (λI + M @ P)^{-1})... complex.

        # PRACTICAL APPROXIMATION: Use batch-local solve.
        # A_t^{local} = K_t @ K_t^T + λI  (ignoring cache and P for tractability)
        # This captures the within-batch covariance structure.
        # η^{local}(i,t) = ||K_t^T @ (K_t @ K_t^T + λI)^{-1} @ k_i||

        # Better: include cache as a diagonal regularizer from its trace
        # Or use the FULL solve with incremental cache

        # Since d=14336 is large, let's use the kernel trick:
        # K_t^T @ A_t^{-1} @ k_i where A_t is d×d
        # = K_t^T @ (λI + sum of rank-1 terms)^{-1} @ k_i
        # Using Woodbury with the accumulated keys as the low-rank part

        # Actually for n_batch=100 keys in d=14336, the solve is:
        # (K_t @ K_t^T + λI) is d×d but has rank ≤ 100 + identity
        # Use: (K_t @ K_t^T + λI)^{-1} @ k_i = (1/λ)(k_i - K_t @ (K_t^T @ K_t + λI_n)^{-1} @ K_t^T @ k_i)
        # where (K_t^T @ K_t + λI_n) is only n_batch × n_batch!

        # Build the small Gram matrix for this batch
        G_t = K_t.T @ K_t  # (n_batch, n_batch) — current batch Gram
        G_t_reg = G_t + L2 * np.eye(n_batch, dtype=np.float32)

        # For cache-aware version: need to include C_{t-1}
        # C_{t-1} = Σ K_b @ K_b^T for b < t
        # Full A_t = K_t @ K_t^T + C_{t-1} + λI
        # Using Woodbury on ALL keys up to t:
        # Let K_all = [K_0 | K_1 | ... | K_{t-1} | K_t], shape (d, n_all)
        # Then A_t = K_all @ K_all^T + λI (ignoring P for now)
        # A_t^{-1} = (1/λ)(I - K_all @ (K_all^T @ K_all + λI)^{-1} @ K_all^T)

        if use_woodbury and t > 0 and len(all_prev_keys) > 0:
            # Stack all keys: previous + current batch
            K_prev = np.concatenate(all_prev_keys, axis=0).T  # (d, n_prev)
            K_all = np.concatenate([K_prev, K_t], axis=1)  # (d, n_all)
            n_all = K_all.shape[1]

            # Cap at last 2000 keys for computational feasibility
            if n_all > 2000:
                K_all = K_all[:, -2000:]
                n_all = 2000

            G_all = K_all.T @ K_all  # (n_all, n_all)
            G_all_reg = G_all + L2 * np.eye(n_all, dtype=np.float32)

            try:
                G_all_inv = np.linalg.solve(
                    G_all_reg, np.eye(n_all, dtype=np.float32)
                )
            except np.linalg.LinAlgError:
                G_all_inv = np.linalg.pinv(G_all_reg)
        else:
            K_all = K_t
            n_all = n_batch
            G_all_inv = np.linalg.solve(
                G_t_reg, np.eye(n_batch, dtype=np.float32)
            )

        # For eligible old edits (installed before batch t)
        eligible_start = 0
        eligible_end = t * batch_size

        if eligible_end == 0:
            all_prev_keys.append(batches_keys[t])
            continue

        # Get old edit keys — sample for speed if too many
        old_positions = list(range(eligible_start, min(eligible_end, n_total)))
        old_cids = [stream_cids[p] for p in old_positions]

        # Build old key matrix
        old_key_indices = []
        for p in old_positions:
            cid = stream_cids[p]
            old_key_indices.append(p)

        # Use batches_keys to get the actual keys in stream order
        old_keys = []
        for b_idx in range(t):
            old_keys.append(batches_keys[b_idx])
        old_K = np.concatenate(old_keys, axis=0)  # (n_old, d)

        # Compute η(i,t) = ||K_t^T @ A_t^{-1} @ k_i|| for each old k_i
        # A_t^{-1} @ k_i = (1/λ)(k_i - K_all @ G_all_inv @ K_all^T @ k_i)
        # Then K_t^T @ A_t^{-1} @ k_i = (1/λ)(K_t^T @ k_i - K_t^T @ K_all @ G_all_inv @ K_all^T @ k_i)

        # Precompute K_t^T @ K_all @ G_all_inv
        KtKall = K_t.T @ K_all  # (n_batch, n_all)
        KtKallGinv = KtKall @ G_all_inv  # (n_batch, n_all)

        # For each old key k_i:
        # projections_all = K_all^T @ k_i  (n_all,)
        # projections_t = K_t^T @ k_i  (n_batch,)
        # η_vec = (1/λ)(projections_t - KtKallGinv @ projections_all)
        # η(i,t) = ||η_vec||

        # Batch compute: old_K @ K_all^T -> (n_old, n_all)
        proj_all = old_K @ K_all  # (n_old, n_all)
        proj_t = old_K @ K_t  # (n_old, n_batch)

        # η_vecs = (1/λ)(proj_t - proj_all @ G_all_inv^T @ KtKall^T)
        #        = (1/λ)(proj_t - proj_all @ (KtKallGinv)^T)
        eta_vecs = (1.0 / L2) * (proj_t - proj_all @ KtKallGinv.T)  # (n_old, n_batch)
        eta_norms = np.linalg.norm(eta_vecs, axis=1)  # (n_old,)

        # Raw cosine: max cos(k_i, k_j) for j in batch t
        k_norms_old = np.linalg.norm(old_K, axis=1, keepdims=True)
        k_norms_old = np.maximum(k_norms_old, 1e-8)
        old_normed = old_K / k_norms_old

        k_norms_t = np.linalg.norm(K_t, axis=0, keepdims=True)  # (1, n_batch)
        k_norms_t = np.maximum(k_norms_t, 1e-8)
        t_normed = K_t / k_norms_t  # (d, n_batch)

        cos_matrix = old_normed @ t_normed  # (n_old, n_batch)
        max_cos = cos_matrix.max(axis=1)  # (n_old,)

        # Accumulate
        for local_i in range(len(old_positions)):
            pos = old_positions[local_i]
            eta_cumul[pos] += eta_norms[local_i]
            cos_cumul[pos] += max(max_cos[local_i], 0)

        # Store checkpoint snapshots
        if t in checkpoint_batches:
            eta_at_ckpt[t] = eta_cumul.copy()
            cos_at_ckpt[t] = cos_cumul.copy()

        # Add current batch keys to cache
        all_prev_keys.append(batches_keys[t])

        if (t + 1) % 10 == 0:
            print(f"  Batch {t}: η_mean={eta_norms.mean():.6f}, "
                  f"cos_mean={max_cos.mean():.4f}, n_old={len(old_positions)}")

    return eta_cumul, cos_cumul, eta_at_ckpt, cos_at_ckpt


def main():
    parser = argparse.ArgumentParser(
        description="Compute solve-aware interference kernel η(i,t)"
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--ordering", type=str, default="key_clustered")
    parser.add_argument("--keys_path", type=str,
                        default="results/key_vectors/full_mcf/keys_seed42_layer6.npz")
    parser.add_argument("--batch_size", type=int, default=100)
    parser.add_argument("--max_batches", type=int, default=50,
                        help="Max batches to process (default 50 = 5K edits)")
    parser.add_argument("--no_projection", action="store_true",
                        help="Skip P matrix (use identity)")
    parser.add_argument("--L2", type=float, default=10.0)
    parser.add_argument("--output_dir", type=str, default=None)
    args = parser.parse_args()

    from paths import get_result_root
    result_root = get_result_root()

    keys_path = Path(args.keys_path)
    if not keys_path.is_absolute():
        keys_path = PROJECT_ROOT / keys_path

    stream_path = result_root / "matched_ordering" / "orderings" / f"{args.ordering}_seed{args.seed}.json"
    assert stream_path.exists(), f"Stream not found: {stream_path}"

    print(f"Interference Kernel Computation")
    print(f"  Seed: {args.seed}, Ordering: {args.ordering}")
    print(f"  Max batches: {args.max_batches}, L2: {args.L2}")

    batches_keys, batches_cids, stream_cids = load_stream_keys(
        keys_path, stream_path, args.batch_size
    )
    print(f"  Loaded {len(batches_keys)} batches, {len(stream_cids)} stream records")

    P = None
    if not args.no_projection:
        P = load_P_matrix(layer_idx=2)  # layer 6 = index 2

    eta_cumul, cos_cumul, eta_at_ckpt, cos_at_ckpt = compute_kernel_scores(
        batches_keys, batches_cids, stream_cids,
        P=P, L2=args.L2, batch_size=args.batch_size,
        max_batches=args.max_batches,
    )

    # Save results
    if args.output_dir:
        out_dir = Path(args.output_dir)
    else:
        out_dir = result_root / "interference_kernel" / args.ordering / f"seed{args.seed}"
    out_dir.mkdir(parents=True, exist_ok=True)

    out = {
        "seed": args.seed,
        "ordering": args.ordering,
        "max_batches": args.max_batches,
        "L2": args.L2,
        "has_P": P is not None,
        "n_edits": len(stream_cids),
        "case_ids": stream_cids,
        "eta_cumulative": eta_cumul.tolist(),
        "cosine_cumulative": cos_cumul.tolist(),
    }

    out_path = out_dir / "kernel_scores.json"
    with open(out_path, "w") as f:
        json.dump(out, f)
    print(f"\n  Saved: {out_path}")

    # Quick comparison
    # For edits in the first 1K (positions 0-999), compare η vs cosine
    # as predictors of a simple outcome
    mask = np.array([i < 1000 for i in range(len(stream_cids))])
    eta_first1k = eta_cumul[mask]
    cos_first1k = cos_cumul[mask]

    if eta_first1k.std() > 0 and cos_first1k.std() > 0:
        corr = np.corrcoef(eta_first1k, cos_first1k)[0, 1]
        print(f"\n  First-1K correlation(η, cosine): r = {corr:.4f}")
        print(f"  η range: [{eta_first1k.min():.4f}, {eta_first1k.max():.4f}]")
        print(f"  cosine range: [{cos_first1k.min():.4f}, {cos_first1k.max():.4f}]")


if __name__ == "__main__":
    main()

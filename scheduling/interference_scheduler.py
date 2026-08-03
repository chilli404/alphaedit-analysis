"""Interference-aware edit scheduling.

Three ordering constructors that minimize cross-batch key-vector interference:
  - greedy_minmax: Greedily schedule the edit with lowest max-future-cosine
  - cluster_topo: Topology-aware cluster scheduling (isolated clusters first)
  - random: Seeded uniform permutation (control arm)

All methods return a permutation of range(N) indices into the record pool.

Usage:
    from scheduling.interference_scheduler import build_ordering

    keys = ...  # (N, D) L2-normalized float32
    perm = build_ordering(keys, method="greedy_minmax", batch_size=100, seed=42)
"""

import sys
from pathlib import Path

import numpy as np

# Allow importing spherical_kmeans from generate_orderings
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT / "src" / "datasets"))


def build_ordering(
    keys: np.ndarray,
    method: str,
    batch_size: int = 100,
    seed: int = 42,
    n_clusters: int = 50,
    verbose: bool = True,
) -> list:
    """Build an optimized ordering of N edits.

    Args:
        keys: (N, D) float32 array. Must be L2-normalized.
        method: Scheduling algorithm — "greedy_minmax", "cluster_topo", or "random".
        batch_size: Number of edits per batch (affects cluster_topo grouping).
        seed: Random seed for determinism.
        n_clusters: Number of spherical k-means clusters (cluster_topo only).
        verbose: Print progress.

    Returns:
        List of N indices — a permutation of range(N).
    """
    N = keys.shape[0]
    if method == "greedy_minmax":
        if verbose:
            print(f"  [scheduler] Computing {N}x{N} cosine matrix...")
        cos_matrix = keys @ keys.T
        np.fill_diagonal(cos_matrix, -np.inf)
        if verbose:
            print(f"  [scheduler] Running greedy_minmax (N={N})...")
        ordering = _greedy_minmax(cos_matrix, seed=seed, verbose=verbose)
    elif method == "cluster_topo":
        if verbose:
            print(f"  [scheduler] Running cluster_topo (N={N}, k={n_clusters})...")
        ordering = _cluster_topo(keys, n_clusters=n_clusters, batch_size=batch_size, seed=seed)
    elif method == "random":
        if verbose:
            print(f"  [scheduler] Random shuffle (N={N}, seed={seed})...")
        ordering = _random_shuffle(N, seed=seed)
    else:
        raise ValueError(f"Unknown scheduling method: {method!r}. "
                         f"Choose from: greedy_minmax, cluster_topo, random")

    # Validate output
    assert len(ordering) == N, f"Ordering length {len(ordering)} != N={N}"
    assert set(ordering) == set(range(N)), "Ordering is not a valid permutation"
    return ordering


def _greedy_minmax(
    cos_matrix: np.ndarray,
    seed: int = 42,
    verbose: bool = True,
) -> list:
    """Greedy argmin-of-max-future-cosine scheduling.

    For each step, picks the unscheduled edit whose max cosine to other
    unscheduled edits is minimal. Uses lazy refresh: only recomputes m[j]
    when the element that was j's argmax gets scheduled.

    Args:
        cos_matrix: (N, N) pairwise cosine similarity with diagonal = -inf.
        seed: For deterministic tie-breaking.
        verbose: Print progress every 1000 steps.

    Returns:
        Permutation of range(N).
    """
    N = cos_matrix.shape[0]
    rng = np.random.default_rng(seed + 5000)

    # State arrays
    scheduled = np.zeros(N, dtype=bool)
    m = np.full(N, np.inf, dtype=np.float64)        # max cos to unscheduled
    argmax_idx = np.full(N, -1, dtype=np.int64)     # which j achieves the max

    # Initialize: for each i, find max over all other j
    for i in range(N):
        row = cos_matrix[i].copy()
        row[i] = -np.inf
        argmax_idx[i] = row.argmax()
        m[i] = row[argmax_idx[i]]

    ordering = []
    n_refreshes = 0

    for step in range(N):
        if verbose and step % 1000 == 0 and step > 0:
            print(f"    step {step}/{N} (refreshes so far: {n_refreshes})")

        # Pick i* = argmin of m over unscheduled
        # Mask scheduled entries with inf
        m_masked = np.where(scheduled, np.inf, m)
        i_star = int(m_masked.argmin())

        # Schedule i*
        ordering.append(i_star)
        scheduled[i_star] = True
        m[i_star] = np.inf

        # Update: for any unscheduled j whose argmax was i_star, recompute
        needs_refresh = (~scheduled) & (argmax_idx == i_star)
        refresh_indices = np.where(needs_refresh)[0]
        n_refreshes += len(refresh_indices)

        for j in refresh_indices:
            # Recompute m[j] over remaining unscheduled (excluding j itself)
            row = cos_matrix[j].copy()
            row[scheduled] = -np.inf
            row[j] = -np.inf
            best = row.argmax()
            argmax_idx[j] = best
            m[j] = row[best]

        # Also: any unscheduled j that had cos[j, i_star] in their row
        # but whose argmax wasn't i_star — their m value is still valid
        # (removing i_star can only decrease or maintain max, never increase)
        # BUT if argmax_idx[j] != i_star, then m[j] is still correct.
        # This is the key insight of lazy refresh.

    return ordering


def _cluster_topo(
    keys: np.ndarray,
    n_clusters: int = 50,
    batch_size: int = 100,
    seed: int = 42,
) -> list:
    """Topology-aware cluster ordering.

    Algorithm:
      1. Spherical k-means clustering (reuse from generate_orderings.py)
      2. Compute centroid cosine similarity matrix (k x k)
      3. isolation[c] = max_{c' != c} S[c, c'] (low = geometrically isolated)
      4. Sort clusters by isolation ascending (isolated clusters first)
      5. Move top-quartile most-connected clusters to end, grouped by
         nearest-centroid adjacency
      6. Within each cluster, keep original corpus order
      7. Concatenate into final ordering
    """
    from generate_orderings import spherical_kmeans

    N = keys.shape[0]
    assignments = spherical_kmeans(keys, n_clusters, max_iter=100, seed=seed)

    # Compute centroids
    actual_clusters = int(assignments.max()) + 1
    centroids = np.zeros((actual_clusters, keys.shape[1]), dtype=np.float32)
    for c in range(actual_clusters):
        mask = assignments == c
        if mask.sum() > 0:
            centroid = keys[mask].mean(axis=0)
            norm = np.linalg.norm(centroid)
            if norm > 1e-8:
                centroids[c] = centroid / norm
            else:
                centroids[c] = centroid

    # Centroid similarity matrix
    S = centroids @ centroids.T
    np.fill_diagonal(S, -np.inf)

    # Isolation: max similarity to any other cluster (low = isolated)
    isolation = S.max(axis=1)

    # Sort clusters by isolation ascending (most isolated first)
    cluster_order = np.argsort(isolation)

    # Top quartile most-connected clusters: move to end, group by adjacency
    n_top_quartile = actual_clusters // 4
    isolated_clusters = cluster_order[:-n_top_quartile] if n_top_quartile > 0 else cluster_order
    connected_clusters = cluster_order[-n_top_quartile:] if n_top_quartile > 0 else np.array([], dtype=int)

    # Group connected clusters by nearest-centroid adjacency (greedy nearest-neighbor chain)
    if len(connected_clusters) > 1:
        ordered_connected = _greedy_nearest_chain(connected_clusters, S)
    else:
        ordered_connected = connected_clusters.tolist()

    # Final cluster sequence: isolated first, then connected
    final_cluster_order = list(isolated_clusters) + ordered_connected

    # Build ordering: within each cluster, keep original corpus order (index order)
    ordering = []
    for c in final_cluster_order:
        members = np.where(assignments == c)[0]
        # Keep original index order within cluster
        ordering.extend(sorted(members.tolist()))

    return ordering


def _greedy_nearest_chain(clusters: np.ndarray, S: np.ndarray) -> list:
    """Order a set of cluster indices by greedy nearest-neighbor chain.

    Starting from the first cluster in the input, always pick the nearest
    unvisited cluster (by centroid cosine similarity).
    """
    remaining = set(clusters.tolist())
    current = clusters[0]
    chain = [current]
    remaining.remove(current)

    while remaining:
        best_sim = -np.inf
        best_next = None
        for candidate in remaining:
            sim = S[current, candidate]
            if sim > best_sim:
                best_sim = sim
                best_next = candidate
        chain.append(best_next)
        remaining.remove(best_next)
        current = best_next

    return chain


def _random_shuffle(n: int, seed: int = 42) -> list:
    """Seeded uniform random permutation."""
    rng = np.random.default_rng(seed + 6000)
    return rng.permutation(n).tolist()

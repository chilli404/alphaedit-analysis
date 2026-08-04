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
    kappa_max: float = 40.0,
    verbose: bool = True,
) -> list:
    """Build an optimized ordering of N edits.

    Args:
        keys: (N, D) array. Must be L2-normalized.
        method: Scheduling algorithm — "greedy_minmax", "greedy_constrained",
                "cluster_topo", or "random".
        batch_size: Number of edits per batch.
        seed: Random seed for determinism.
        n_clusters: Number of spherical k-means clusters (cluster_topo only).
        kappa_max: Per-batch Gram condition number cap (greedy_constrained only).
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
    elif method == "greedy_constrained":
        if verbose:
            print(f"  [scheduler] Computing {N}x{N} cosine matrix...")
        cos_matrix = keys @ keys.T
        np.fill_diagonal(cos_matrix, -np.inf)
        if verbose:
            print(f"  [scheduler] Running greedy_constrained (N={N}, "
                  f"batch_size={batch_size}, κ_max={kappa_max})...")
        ordering = _greedy_constrained(
            cos_matrix, keys, batch_size=batch_size,
            kappa_max=kappa_max, seed=seed, verbose=verbose,
        )
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
                         f"Choose from: greedy_minmax, greedy_constrained, cluster_topo, random")

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


def _greedy_constrained(
    cos_matrix: np.ndarray,
    keys: np.ndarray,
    batch_size: int = 100,
    kappa_max: float = 40.0,
    seed: int = 42,
    verbose: bool = True,
) -> list:
    """Greedy argmin-of-max-future-cosine with per-batch Gram κ constraint.

    Same objective as greedy_minmax (minimize max cosine to remaining unscheduled)
    but rejects any candidate whose addition would push the current batch's
    cosine-normalized Gram condition number above kappa_max.

    When all top candidates violate κ, accepts the lowest-exposure candidate
    regardless (fallback — logged but not fatal).

    Args:
        cos_matrix: (N, N) pairwise cosine similarity with diagonal = -inf.
        keys: (N, D) L2-normalized key vectors (for Gram κ computation).
        batch_size: Edits per batch.
        kappa_max: Maximum allowed per-batch Gram condition number.
        seed: For deterministic tie-breaking.
        verbose: Print progress.

    Returns:
        Permutation of range(N).
    """
    N = cos_matrix.shape[0]
    rng = np.random.default_rng(seed + 7000)

    # Same lazy-refresh state as greedy_minmax
    scheduled = np.zeros(N, dtype=bool)
    m = np.full(N, np.inf, dtype=np.float64)
    argmax_idx = np.full(N, -1, dtype=np.int64)

    for i in range(N):
        row = cos_matrix[i].copy()
        row[i] = -np.inf
        argmax_idx[i] = row.argmax()
        m[i] = row[argmax_idx[i]]

    ordering = []
    n_refreshes = 0
    n_kappa_rejects = 0
    n_kappa_fallbacks = 0
    batch_kappas = []  # final κ of each completed batch
    MAX_SEARCH = 30  # max candidates to try before fallback

    n_batches = (N + batch_size - 1) // batch_size

    for batch_idx in range(n_batches):
        batch_keys_list = []  # keys of members added to this batch so far
        batch_gram = None  # running normalized Gram (updated incrementally)

        actual_batch_size = min(batch_size, N - len(ordering))

        for slot in range(actual_batch_size):
            # Find the best candidate(s) by exposure
            m_masked = np.where(scheduled, np.inf, m)

            selected = None
            n_tried = 0

            # Temporary mask: candidates rejected for this slot (restore after)
            slot_rejects = []

            while selected is None and n_tried < MAX_SEARCH:
                i_star = int(m_masked.argmin())
                if m_masked[i_star] == np.inf:
                    break  # no more candidates

                # κ check: skip for first few members (κ undefined for < 3)
                if len(batch_keys_list) >= 3:
                    # Compute κ with candidate added
                    cand_key = keys[i_star]
                    cand_norm = np.linalg.norm(cand_key)
                    if cand_norm > 1e-8:
                        cand_normed = cand_key / cand_norm
                    else:
                        cand_normed = cand_key

                    # Incremental Gram: add row/column for new member
                    n_cur = len(batch_keys_list)
                    new_gram = np.empty((n_cur + 1, n_cur + 1), dtype=np.float64)
                    new_gram[:n_cur, :n_cur] = batch_gram
                    # New row/col: cosine of candidate with each existing member
                    for k_idx in range(n_cur):
                        existing_normed = batch_keys_list[k_idx]
                        dot = float(np.dot(existing_normed, cand_normed))
                        new_gram[n_cur, k_idx] = dot
                        new_gram[k_idx, n_cur] = dot
                    new_gram[n_cur, n_cur] = 1.0

                    eigvals = np.linalg.eigvalsh(new_gram)
                    eigvals_pos = eigvals[eigvals > 1e-10]
                    if len(eigvals_pos) >= 2:
                        kappa = float(eigvals_pos[-1] / eigvals_pos[0])
                    else:
                        kappa = 1.0

                    if kappa > kappa_max:
                        # Reject: temporarily mask this candidate for this slot
                        n_kappa_rejects += 1
                        n_tried += 1
                        slot_rejects.append(i_star)
                        m_masked[i_star] = np.inf
                        continue

                    # Passes κ — accept and update Gram
                    batch_gram = new_gram
                    batch_keys_list.append(cand_normed)
                    selected = i_star
                else:
                    # Not enough members for κ check — accept unconditionally
                    cand_key = keys[i_star]
                    cand_norm = np.linalg.norm(cand_key)
                    cand_normed = cand_key / cand_norm if cand_norm > 1e-8 else cand_key
                    batch_keys_list.append(cand_normed)

                    # Initialize or grow Gram
                    n_cur = len(batch_keys_list)
                    if n_cur == 1:
                        batch_gram = np.array([[1.0]])
                    else:
                        new_gram = np.empty((n_cur, n_cur), dtype=np.float64)
                        new_gram[:n_cur-1, :n_cur-1] = batch_gram
                        for k_idx in range(n_cur - 1):
                            dot = float(np.dot(batch_keys_list[k_idx], cand_normed))
                            new_gram[n_cur-1, k_idx] = dot
                            new_gram[k_idx, n_cur-1] = dot
                        new_gram[n_cur-1, n_cur-1] = 1.0
                        batch_gram = new_gram

                    selected = i_star

            # Fallback: if MAX_SEARCH exhausted, take the original best
            if selected is None:
                # Restore rejected candidates and just take the lowest-exposure one
                m_masked_restore = np.where(scheduled, np.inf, m)
                i_star = int(m_masked_restore.argmin())
                if m_masked_restore[i_star] < np.inf:
                    selected = i_star
                    n_kappa_fallbacks += 1

                    # Update Gram anyway (we're over κ, but must track state)
                    cand_key = keys[selected]
                    cand_norm = np.linalg.norm(cand_key)
                    cand_normed = cand_key / cand_norm if cand_norm > 1e-8 else cand_key
                    n_cur = len(batch_keys_list)
                    new_gram = np.empty((n_cur + 1, n_cur + 1), dtype=np.float64)
                    new_gram[:n_cur, :n_cur] = batch_gram
                    for k_idx in range(n_cur):
                        dot = float(np.dot(batch_keys_list[k_idx], cand_normed))
                        new_gram[n_cur, k_idx] = dot
                        new_gram[k_idx, n_cur] = dot
                    new_gram[n_cur, n_cur] = 1.0
                    batch_gram = new_gram
                    batch_keys_list.append(cand_normed)

            if selected is None:
                break  # shouldn't happen unless all edits are scheduled

            # Schedule selected
            ordering.append(int(selected))
            scheduled[selected] = True
            m[selected] = np.inf

            # Lazy refresh (same logic as unconstrained)
            needs_refresh = (~scheduled) & (argmax_idx == selected)
            refresh_indices = np.where(needs_refresh)[0]
            n_refreshes += len(refresh_indices)

            for j in refresh_indices:
                row = cos_matrix[j].copy()
                row[scheduled] = -np.inf
                row[j] = -np.inf
                best = row.argmax()
                argmax_idx[j] = best
                m[j] = row[best]

        # Record final batch κ
        if batch_gram is not None and batch_gram.shape[0] >= 2:
            eigvals = np.linalg.eigvalsh(batch_gram)
            eigvals_pos = eigvals[eigvals > 1e-10]
            if len(eigvals_pos) >= 2:
                batch_kappas.append(float(eigvals_pos[-1] / eigvals_pos[0]))
            else:
                batch_kappas.append(1.0)
        else:
            batch_kappas.append(1.0)

        if verbose and (batch_idx + 1) % 10 == 0:
            recent_kappas = batch_kappas[-10:]
            print(f"    batch {batch_idx+1}/{n_batches}: "
                  f"κ_max_recent={max(recent_kappas):.1f}, "
                  f"rejects={n_kappa_rejects}, fallbacks={n_kappa_fallbacks}")

    if verbose:
        if batch_kappas:
            print(f"  [constrained] Done. κ stats: "
                  f"max={max(batch_kappas):.1f}, "
                  f"mean={np.mean(batch_kappas):.1f}, "
                  f"median={np.median(batch_kappas):.1f}")
            violations = sum(1 for k in batch_kappas if k > kappa_max)
            print(f"  [constrained] Batches > κ_max: {violations}/{len(batch_kappas)} "
                  f"({n_kappa_rejects} total candidate rejections, "
                  f"{n_kappa_fallbacks} fallbacks)")

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

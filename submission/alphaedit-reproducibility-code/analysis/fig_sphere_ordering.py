"""
Visualize spherical k-means cluster structure and temporal ordering traversal.

Four visualization approaches:
1. Geodesic MDS: Project cluster centroids via MDS on angular distances, show temporal paths
2. Polar/circular layout: Arrange clusters around unit circle by angular proximity, show batch arcs
3. Inter-cluster cosine matrix with temporal overlay
4. Cross-batch exposure: Shows WHY clustered is better — per-edit max cosine to future keys

Usage:
    uv run python -m analysis.fig_sphere_ordering
"""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.collections import LineCollection
from matplotlib.colors import Normalize
import numpy as np
from scipy.spatial.distance import squareform, pdist

from analysis.style import setup_style, save_figure, PAPER_OUTPUT

# ─── Constants ────────────────────────────────────────────────────────────────

SEED = 42
N_CLUSTERS = 50
BATCH_SIZE = 100
KEY_PATH = Path("results/matched_ordering/key_geometry/keys_seed42_layer6.npz")
ORDERING_DIR = Path("results/matched_ordering/orderings")

# Colors
CLUSTERED_COLOR = "#2196F3"   # Blue
DISPERSED_COLOR = "#E91E63"   # Pink
CENTROID_COLOR = "#333333"
BATCH_CMAP = "viridis"


# ─── Data Loading ─────────────────────────────────────────────────────────────

def load_keys():
    """Load L2-normalized key vectors."""
    data = np.load(KEY_PATH)
    keys = data["keys"]
    case_ids = data["case_ids"]
    # L2-normalize
    norms = np.linalg.norm(keys, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-8)
    keys = keys / norms
    return keys, case_ids


def spherical_kmeans(keys, n_clusters, max_iter=100, seed=42):
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
            break

        for c in range(n_clusters):
            mask = assignments == c
            if mask.sum() > 0:
                centroids[c] = normed[mask].mean(axis=0)
                cn = np.linalg.norm(centroids[c])
                if cn > 1e-8:
                    centroids[c] /= cn

    return assignments, centroids


def load_ordering(ordering_name, seed=42):
    """Load ordering JSON and return list of case_ids in stream order."""
    path = ORDERING_DIR / f"{ordering_name}_seed{seed}.json"
    records = json.loads(path.read_text())
    return [r["case_id"] for r in records]


def get_stream_cluster_sequence(ordering_case_ids, case_ids, assignments):
    """Map ordering case_ids to cluster assignments in stream order."""
    case_to_cluster = dict(zip(case_ids.tolist(), assignments.tolist()))
    clusters_in_order = []
    for cid in ordering_case_ids:
        if cid in case_to_cluster:
            clusters_in_order.append(case_to_cluster[cid])
        # skip case_ids not in our key set
    return np.array(clusters_in_order)


# ─── Visualization 1: Geodesic MDS ───────────────────────────────────────────

def plot_geodesic_mds(centroids, clustered_seq, dispersed_seq, ax_clust, ax_disp):
    """Project centroids via MDS on angular distances, show temporal traversal paths."""
    n_clusters = len(centroids)

    # Compute pairwise angular distances
    cos_sim = centroids @ centroids.T
    cos_sim = np.clip(cos_sim, -1, 1)
    angular_dist = np.arccos(cos_sim)
    np.fill_diagonal(angular_dist, 0)

    # MDS embedding
    from sklearn.manifold import MDS
    mds = MDS(n_components=2, dissimilarity="precomputed", random_state=SEED, normalized_stress="auto")
    coords = mds.fit_transform(angular_dist)

    # Normalize to unit circle
    max_r = np.max(np.linalg.norm(coords, axis=1))
    coords = coords / max_r * 0.85

    for ax, seq, title, color in [
        (ax_clust, clustered_seq, "Key Clustered", CLUSTERED_COLOR),
        (ax_disp, dispersed_seq, "Key Dispersed", DISPERSED_COLOR),
    ]:
        # Draw unit circle
        theta = np.linspace(0, 2 * np.pi, 100)
        ax.plot(np.cos(theta), np.sin(theta), "k-", lw=0.5, alpha=0.3)

        # Draw centroids
        ax.scatter(coords[:, 0], coords[:, 1], s=60, c=CENTROID_COLOR,
                   alpha=0.4, zorder=2, edgecolors="white", linewidths=0.5)

        # Draw temporal path for first 5 batches
        n_show_batches = 5
        batch_seq = seq[: n_show_batches * BATCH_SIZE]
        cmap = plt.get_cmap(BATCH_CMAP)

        for b in range(n_show_batches):
            batch_clusters = batch_seq[b * BATCH_SIZE : (b + 1) * BATCH_SIZE]
            # Get unique clusters visited in this batch (in order of first appearance)
            seen = []
            for c in batch_clusters:
                if c not in seen:
                    seen.append(c)

            # Draw path through cluster centroids
            batch_coords = coords[seen]
            alpha = 0.7 - b * 0.1
            lw = 2.0 - b * 0.2
            batch_color = cmap(b / n_show_batches)
            ax.plot(batch_coords[:, 0], batch_coords[:, 1],
                    "-", color=batch_color, alpha=alpha, lw=lw, zorder=3)
            # Mark start of each batch
            ax.scatter(batch_coords[0, 0], batch_coords[0, 1],
                       s=30, c=[batch_color], marker="o", zorder=4, edgecolors="k", linewidths=0.5)

        # Label clusters with IDs
        for i in range(n_clusters):
            ax.annotate(str(i), coords[i], fontsize=5, ha="center", va="center",
                        color="#666666", alpha=0.6)

        ax.set_xlim(-1.1, 1.1)
        ax.set_ylim(-1.1, 1.1)
        ax.set_aspect("equal")
        ax.set_title(title, fontsize=11)
        ax.axis("off")


# ─── Visualization 2: Polar/Circular Layout ──────────────────────────────────

def compute_circular_order(centroids):
    """Order clusters around a circle by angular proximity (greedy TSP)."""
    n = len(centroids)
    cos_sim = centroids @ centroids.T

    # Greedy nearest-neighbor tour
    visited = [0]
    remaining = set(range(1, n))
    while remaining:
        last = visited[-1]
        # Find nearest unvisited
        best = max(remaining, key=lambda x: cos_sim[last, x])
        visited.append(best)
        remaining.remove(best)
    return visited


def plot_polar_layout(centroids, clustered_seq, dispersed_seq, ax_clust, ax_disp):
    """Arrange clusters around unit circle by angular proximity, show batch arcs."""
    n_clusters = len(centroids)
    circular_order = compute_circular_order(centroids)

    # Map cluster_id -> angular position on circle
    cluster_to_angle = {}
    for pos, cluster_id in enumerate(circular_order):
        cluster_to_angle[cluster_id] = 2 * np.pi * pos / n_clusters

    for ax, seq, title, color in [
        (ax_clust, clustered_seq, "Key Clustered", CLUSTERED_COLOR),
        (ax_disp, dispersed_seq, "Key Dispersed", DISPERSED_COLOR),
    ]:
        # Draw unit circle
        theta = np.linspace(0, 2 * np.pi, 200)
        ax.plot(np.cos(theta), np.sin(theta), "k-", lw=0.8, alpha=0.3)

        # Draw cluster positions on circle
        angles = np.array([cluster_to_angle[i] for i in range(n_clusters)])
        cx = np.cos(angles)
        cy = np.sin(angles)
        ax.scatter(cx, cy, s=40, c=CENTROID_COLOR, alpha=0.5, zorder=3,
                   edgecolors="white", linewidths=0.5)

        # Draw batch arcs for first 10 batches
        n_show_batches = 10
        cmap = plt.get_cmap(BATCH_CMAP)

        for b in range(n_show_batches):
            batch_clusters = seq[b * BATCH_SIZE : (b + 1) * BATCH_SIZE]
            unique_clusters = list(set(batch_clusters.tolist()))

            # Get angles of clusters in this batch
            batch_angles = sorted([cluster_to_angle[c] for c in unique_clusters])

            batch_color = cmap(b / n_show_batches)
            radius = 0.75 - b * 0.04

            # Draw dots at each cluster position (on inner ring)
            for c in unique_clusters:
                a = cluster_to_angle[c]
                ax.scatter(radius * np.cos(a), radius * np.sin(a),
                           s=12, c=[batch_color], alpha=0.7, zorder=4)

            # Draw arc connecting the angular span
            if len(batch_angles) > 1:
                # Compute angular span
                arc_angles = np.linspace(min(batch_angles), max(batch_angles), 50)
                ax.plot(radius * np.cos(arc_angles), radius * np.sin(arc_angles),
                        "-", color=batch_color, alpha=0.5, lw=1.5)

        # Cluster labels
        for i in range(n_clusters):
            a = cluster_to_angle[i]
            label_r = 1.08
            ax.annotate(str(i), (label_r * np.cos(a), label_r * np.sin(a)),
                        fontsize=5, ha="center", va="center", color="#666666")

        ax.set_xlim(-1.35, 1.35)
        ax.set_ylim(-1.35, 1.35)
        ax.set_aspect("equal")
        ax.set_title(title, fontsize=11)
        ax.axis("off")


# ─── Visualization 3: Cosine Matrix + Temporal Overlay ────────────────────────

def plot_cosine_matrix(centroids, clustered_seq, dispersed_seq, ax_clust, ax_disp, ax_mat):
    """Show inter-cluster cosine matrix and batch co-occurrence patterns."""
    n_clusters = len(centroids)

    # Compute pairwise cosine similarity between centroids
    cos_sim = centroids @ centroids.T

    # Reorder by circular tour for better visual structure
    circular_order = compute_circular_order(centroids)
    reordered = cos_sim[np.ix_(circular_order, circular_order)]

    # Plot cosine similarity matrix
    im = ax_mat.imshow(reordered, cmap="RdBu_r", vmin=-0.2, vmax=0.8, aspect="equal")
    ax_mat.set_title("Inter-Cluster Cosine Similarity", fontsize=10)
    ax_mat.set_xlabel("Cluster (angular order)")
    ax_mat.set_ylabel("Cluster (angular order)")
    plt.colorbar(im, ax=ax_mat, fraction=0.046, pad=0.04)

    # For each ordering, compute batch co-occurrence matrix
    for ax, seq, title, color in [
        (ax_clust, clustered_seq, "Clustered: Batch Co-occurrence", CLUSTERED_COLOR),
        (ax_disp, dispersed_seq, "Dispersed: Batch Co-occurrence", DISPERSED_COLOR),
    ]:
        n_batches = len(seq) // BATCH_SIZE
        cooccur = np.zeros((n_clusters, n_clusters))

        for b in range(min(n_batches, 50)):  # First 50 batches
            batch_clusters = seq[b * BATCH_SIZE : (b + 1) * BATCH_SIZE]
            unique = list(set(batch_clusters.tolist()))
            for i in unique:
                for j in unique:
                    if i != j:
                        cooccur[i, j] += 1

        # Normalize
        if cooccur.max() > 0:
            cooccur = cooccur / cooccur.max()

        # Reorder to match circular order
        reordered_co = cooccur[np.ix_(circular_order, circular_order)]

        im2 = ax.imshow(reordered_co, cmap="YlOrRd", vmin=0, vmax=1, aspect="equal")
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("Cluster (angular order)")
        ax.set_ylabel("Cluster (angular order)")


# ─── Visualization 4: Cross-Batch Exposure ───────────────────────────────────

def compute_cross_batch_exposure(keys, stream_case_ids, case_ids, batch_size=100):
    """For each edit, compute max cosine similarity to all keys edited in subsequent batches.

    Returns array of shape (n_edits,) where each value is the max cosine
    between that edit's key and any key edited AFTER it (in later batches).
    """
    # Build case_id -> key index mapping
    case_to_idx = {int(cid): i for i, cid in enumerate(case_ids)}

    # Get key indices in stream order
    stream_key_indices = []
    for cid in stream_case_ids:
        if cid in case_to_idx:
            stream_key_indices.append(case_to_idx[cid])
    stream_key_indices = np.array(stream_key_indices)

    n_edits = len(stream_key_indices)
    n_batches = n_edits // batch_size

    # Precompute batch boundaries
    max_cos_per_edit = np.zeros(n_edits)

    for b in range(n_batches - 1):  # Skip last batch (no future)
        batch_start = b * batch_size
        batch_end = (b + 1) * batch_size
        future_start = batch_end

        # Keys from this batch
        batch_keys = keys[stream_key_indices[batch_start:batch_end]]  # (batch_size, D)
        # Keys from all future batches
        future_keys = keys[stream_key_indices[future_start:]]  # (n_future, D)

        # Cosine similarities: (batch_size, n_future)
        cos_sims = batch_keys @ future_keys.T

        # Max cosine to any future key, for each edit in this batch
        max_cos_per_edit[batch_start:batch_end] = cos_sims.max(axis=1)

    return max_cos_per_edit


def plot_cross_batch_exposure(keys, case_ids, clustered_ids, dispersed_ids, axes):
    """Plot cross-batch exposure comparison showing why clustered is better.

    Three panels:
    - Left: Running mean of max-cosine-to-future per edit index
    - Middle: Histogram of max-cosine-to-future values
    - Right: Fraction of edits with max_cos > threshold over time
    """
    ax_line, ax_hist, ax_frac = axes

    print("  Computing cross-batch exposure (clustered)...")
    clust_exposure = compute_cross_batch_exposure(keys, clustered_ids, case_ids)
    print("  Computing cross-batch exposure (dispersed)...")
    disp_exposure = compute_cross_batch_exposure(keys, dispersed_ids, case_ids)

    n_edits = min(len(clust_exposure), len(disp_exposure))
    clust_exposure = clust_exposure[:n_edits]
    disp_exposure = disp_exposure[:n_edits]

    # Panel 1: Running mean of max-cosine-to-future
    window = 200
    clust_smooth = np.convolve(clust_exposure, np.ones(window)/window, mode="valid")
    disp_smooth = np.convolve(disp_exposure, np.ones(window)/window, mode="valid")
    x_axis = np.arange(len(clust_smooth))

    ax_line.plot(x_axis, disp_smooth, color=DISPERSED_COLOR, lw=1.5, label="Dispersed", alpha=0.9)
    ax_line.plot(x_axis, clust_smooth, color=CLUSTERED_COLOR, lw=1.5, label="Clustered", alpha=0.9)
    ax_line.set_xlabel("Edit Index")
    ax_line.set_ylabel("Max Cosine to Future Keys")
    ax_line.set_title("Cross-Batch Exposure\n(running mean, window=200)")
    ax_line.legend(frameon=False)
    ax_line.set_ylim(0, 1.0)

    # Add annotation showing the gap
    mid = len(clust_smooth) // 2
    gap = disp_smooth[mid] - clust_smooth[mid]
    ax_line.annotate(
        f"Gap: {gap:.3f}",
        xy=(mid, (disp_smooth[mid] + clust_smooth[mid]) / 2),
        fontsize=8, ha="center", color="#333333",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="#cccccc", alpha=0.8)
    )

    # Panel 2: Histogram
    # Only use edits not in last batch (which have zero exposure)
    clust_nonzero = clust_exposure[clust_exposure > 0]
    disp_nonzero = disp_exposure[disp_exposure > 0]

    bins = np.linspace(0, 1, 40)
    ax_hist.hist(disp_nonzero, bins=bins, alpha=0.6, color=DISPERSED_COLOR,
                 label=f"Dispersed (mean={disp_nonzero.mean():.3f})", density=True)
    ax_hist.hist(clust_nonzero, bins=bins, alpha=0.6, color=CLUSTERED_COLOR,
                 label=f"Clustered (mean={clust_nonzero.mean():.3f})", density=True)
    ax_hist.set_xlabel("Max Cosine to Future Keys")
    ax_hist.set_ylabel("Density")
    ax_hist.set_title("Distribution of\nCross-Batch Exposure")
    ax_hist.legend(frameon=False, fontsize=8)

    # Panel 3: Fraction above threshold over batch index
    thresholds = [0.3, 0.5, 0.7]
    batch_size = BATCH_SIZE
    n_batches = n_edits // batch_size

    for thresh in thresholds:
        clust_fracs = []
        disp_fracs = []
        for b in range(n_batches - 1):
            start = b * batch_size
            end = (b + 1) * batch_size
            clust_fracs.append((clust_exposure[start:end] > thresh).mean())
            disp_fracs.append((disp_exposure[start:end] > thresh).mean())

        ls = "-" if thresh == 0.3 else ("--" if thresh == 0.5 else ":")
        ax_frac.plot(range(len(disp_fracs)), disp_fracs, color=DISPERSED_COLOR,
                     ls=ls, lw=1.2, alpha=0.8, label=f"Disp cos>{thresh}")
        ax_frac.plot(range(len(clust_fracs)), clust_fracs, color=CLUSTERED_COLOR,
                     ls=ls, lw=1.2, alpha=0.8, label=f"Clust cos>{thresh}")

    ax_frac.set_xlabel("Batch Index")
    ax_frac.set_ylabel("Fraction of Edits Exposed")
    ax_frac.set_title("Fraction with High Future\nCosine Exposure per Batch")
    ax_frac.legend(frameon=False, fontsize=7, ncol=2)
    ax_frac.set_ylim(0, 1.05)


# ─── Main Generation ──────────────────────────────────────────────────────────

def generate():
    """Generate all three visualization variants."""
    setup_style()

    print("Loading keys and computing clusters...")
    keys, case_ids = load_keys()
    assignments, centroids = spherical_kmeans(keys, N_CLUSTERS, seed=SEED)

    print(f"  {N_CLUSTERS} clusters, {len(keys)} keys")
    print(f"  Cluster sizes: min={np.bincount(assignments).min()}, "
          f"max={np.bincount(assignments).max()}, "
          f"mean={np.bincount(assignments).mean():.0f}")

    print("Loading orderings...")
    clustered_ids = load_ordering("key_clustered", SEED)
    dispersed_ids = load_ordering("key_dispersed", SEED)

    clustered_seq = get_stream_cluster_sequence(clustered_ids, case_ids, assignments)
    dispersed_seq = get_stream_cluster_sequence(dispersed_ids, case_ids, assignments)

    print(f"  Clustered stream: {len(clustered_seq)} edits mapped to clusters")
    print(f"  Dispersed stream: {len(dispersed_seq)} edits mapped to clusters")

    # Compute summary stats
    n_batches_show = min(50, len(clustered_seq) // BATCH_SIZE)
    clust_clusters_per_batch = []
    disp_clusters_per_batch = []
    for b in range(n_batches_show):
        cb = clustered_seq[b * BATCH_SIZE : (b + 1) * BATCH_SIZE]
        db = dispersed_seq[b * BATCH_SIZE : (b + 1) * BATCH_SIZE]
        clust_clusters_per_batch.append(len(set(cb.tolist())))
        disp_clusters_per_batch.append(len(set(db.tolist())))

    print(f"  Clusters/batch — clustered: {np.mean(clust_clusters_per_batch):.1f}, "
          f"dispersed: {np.mean(disp_clusters_per_batch):.1f}")

    # ─── Figure 1: Geodesic MDS ──────────────────────────────────────────────
    print("\nGenerating Option 1: Geodesic MDS...")
    fig1, (ax1a, ax1b) = plt.subplots(1, 2, figsize=(10, 5))
    plot_geodesic_mds(centroids, clustered_seq, dispersed_seq, ax1a, ax1b)
    fig1.suptitle("Cluster Traversal on Angular-Distance Sphere (MDS Projection)\n"
                  "Lines show temporal path through cluster centroids (first 5 batches)",
                  fontsize=11, y=0.98)
    fig1.tight_layout()
    save_figure(fig1, "sphere_option1_geodesic_mds")

    # ─── Figure 2: Polar/Circular Layout ─────────────────────────────────────
    print("Generating Option 2: Polar/Circular Layout...")
    fig2, (ax2a, ax2b) = plt.subplots(1, 2, figsize=(10, 5))
    plot_polar_layout(centroids, clustered_seq, dispersed_seq, ax2a, ax2b)
    fig2.suptitle("Cluster Positions on Unit Circle (Angular Proximity Order)\n"
                  f"Inner rings show batch cluster membership (first 10 batches, "
                  f"clustered: {np.mean(clust_clusters_per_batch):.0f} clusters/batch, "
                  f"dispersed: {np.mean(disp_clusters_per_batch):.0f} clusters/batch)",
                  fontsize=10, y=0.98)
    fig2.tight_layout()
    save_figure(fig2, "sphere_option2_polar")

    # ─── Figure 3: Cosine Matrix + Co-occurrence ─────────────────────────────
    print("Generating Option 3: Cosine Matrix + Temporal Overlay...")
    fig3, (ax3a, ax3b, ax3c) = plt.subplots(1, 3, figsize=(14, 4.5))
    plot_cosine_matrix(centroids, clustered_seq, dispersed_seq, ax3a, ax3b, ax3c)
    fig3.suptitle("Inter-Cluster Structure and Batch Co-occurrence Patterns",
                  fontsize=11, y=0.98)
    fig3.tight_layout()
    save_figure(fig3, "sphere_option3_cosine_matrix")

    # ─── Figure 4: Cross-Batch Exposure ──────────────────────────────────────
    print("\nGenerating Option 4: Cross-Batch Exposure...")
    fig4, (ax4a, ax4b, ax4c) = plt.subplots(1, 3, figsize=(14, 4.5))
    plot_cross_batch_exposure(keys, case_ids, clustered_ids, dispersed_ids,
                             (ax4a, ax4b, ax4c))
    fig4.suptitle("Cross-Batch Cosine Exposure: Why Clustered Ordering Preserves Better\n"
                  "Each edit's max cosine similarity to all keys edited in subsequent batches",
                  fontsize=10, y=1.0)
    fig4.tight_layout()
    save_figure(fig4, "sphere_option4_exposure")

    print("\nDone! All figures saved to results/figures/paper/")


if __name__ == "__main__":
    generate()

"""
Visualize spherical k-means cluster structure and temporal ordering traversal.

Three visualization approaches:
1. Geodesic MDS: Project cluster centroids via MDS on angular distances, show temporal paths
2. Polar/circular layout: Arrange clusters around unit circle by angular proximity, show batch arcs
3. Inter-cluster cosine matrix with temporal overlay

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

    print("\nDone! All figures saved to results/figures/paper/")


if __name__ == "__main__":
    generate()

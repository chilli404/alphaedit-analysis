"""Supplemental Figure 1B — Key-space geometry under concentration.

Visualizes the edit-key vector geometry difference between low-concentration
(dispersed) and high-concentration (clustered) streams via shared PCA/SVD.

Approach:
  1. Collect all 5000 edit-key vectors for both orderings.
  2. Fit a shared 2D projection (SVD on all combined keys).
  3. Project all keys into that shared basis.
  4. Color a subset of batches with distinct colors; rest in gray.
  5. In clustered: same-colored keys clump spatially.
     In dispersed: same-colored keys scatter uniformly.
  6. Annotate with quantitative metrics.

Usage:
    uv run python -m analysis.suppfig1b_key_geometry
    uv run python -m analysis.suppfig1b_key_geometry --output-dir results/figures/appendix
    uv run python -m analysis.suppfig1b_key_geometry --seed 42
"""

import argparse
import json
import warnings
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from analysis.style import setup_style, save_figure, PAPER_OUTPUT, RESULTS

# ─── Configuration ────────────────────────────────────────────────────────────

DEFAULT_SEED = 42
DEFAULT_LAYER = 6
BATCH_SIZE = 100

# Batches to highlight (spread across the stream for visual diversity)
HIGHLIGHT_BATCHES = [5, 20, 40, 60, 80, 95]
HIGHLIGHT_COLORS = ["#E91E63", "#2196F3", "#4CAF50", "#FF9800", "#9C27B0", "#00BCD4"]


# ─── Core Logic ──────────────────────────────────────────────────────────────


def load_stream_keys(seed: int, ordering: str, layer: int = DEFAULT_LAYER):
    """Load key vectors in stream order for a given ordering.

    Returns:
        keys_ordered: ndarray (N, D) float64
        positions: list of int — stream positions of the returned keys
    """
    npz_path = RESULTS / "matched_ordering" / "key_geometry" / f"keys_seed{seed}_layer{layer}.npz"
    npz = np.load(npz_path)
    all_keys = npz["keys"].astype(np.float64)
    case_ids = npz["case_ids"].tolist()
    key_index = {int(cid): i for i, cid in enumerate(case_ids)}

    stream_path = RESULTS / "matched_ordering" / "orderings" / f"{ordering}_seed{seed}.json"
    with open(stream_path) as f:
        stream = json.load(f)

    keys_ordered = []
    positions = []
    for pos, record in enumerate(stream):
        cid = record["case_id"]
        if cid in key_index:
            keys_ordered.append(all_keys[key_index[cid]])
            positions.append(pos)

    return np.array(keys_ordered, dtype=np.float64), positions


def compute_batch_concentration(keys: np.ndarray, positions: list,
                                batch_size: int = BATCH_SIZE) -> float:
    """Compute mean within-batch cosine similarity."""
    batch_cosines = []
    pos_arr = np.array(positions)
    for b_start in range(0, int(pos_arr.max()) + 1, batch_size):
        b_end = b_start + batch_size
        mask = (pos_arr >= b_start) & (pos_arr < b_end)
        batch_keys = keys[mask]
        if len(batch_keys) < 2:
            continue
        norms = np.linalg.norm(batch_keys, axis=1, keepdims=True)
        batch_normed = batch_keys / (norms + 1e-10)
        cos_matrix = batch_normed @ batch_normed.T
        n = len(batch_keys)
        triu_idx = np.triu_indices(n, k=1)
        batch_cosines.append(cos_matrix[triu_idx].mean())
    return np.mean(batch_cosines) if batch_cosines else 0.0


# ─── Figure Generation ────────────────────────────────────────────────────────


def generate(output_dir: Path = PAPER_OUTPUT, seed: int = DEFAULT_SEED):
    """Generate Supplemental Figure 1B."""
    warnings.filterwarnings("ignore", category=RuntimeWarning,
                            message=".*encountered in matmul.*")
    setup_style()

    # Load all keys for both orderings
    K_clust, pos_clust = load_stream_keys(seed, "key_clustered")
    K_disp, pos_disp = load_stream_keys(seed, "key_dispersed")

    print(f"  Keys used: clustered={len(K_clust)}, dispersed={len(K_disp)}")

    # ─── Shared SVD basis on all combined keys ───────────────────────────
    K_combined = np.vstack([K_disp, K_clust])
    K_mean = K_combined.mean(axis=0, keepdims=True)
    K_centered = K_combined - K_mean

    from scipy.sparse.linalg import svds
    U, S_top, Vt_top = svds(K_centered, k=2)
    idx_sort = np.argsort(S_top)[::-1]
    S_top = S_top[idx_sort]
    Vt_top = Vt_top[idx_sort]
    basis = Vt_top.T  # (D, 2)

    total_var = np.sum(K_centered ** 2)
    var_explained = S_top ** 2 / total_var

    # Project all keys
    Z_clust = (K_clust - K_mean) @ basis
    Z_disp = (K_disp - K_mean) @ basis

    # ─── Assign batch labels ─────────────────────────────────────────────
    pos_clust_arr = np.array(pos_clust)
    pos_disp_arr = np.array(pos_disp)
    batch_clust = pos_clust_arr // BATCH_SIZE
    batch_disp = pos_disp_arr // BATCH_SIZE

    # ─── Quantitative metrics ────────────────────────────────────────────
    wbc_clust = compute_batch_concentration(K_clust, pos_clust)
    wbc_disp = compute_batch_concentration(K_disp, pos_disp)

    # ─── Plot ──────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))

    panel_data = [
        (axes[0], Z_disp, batch_disp,
         "Low Concentration (Dispersed)", wbc_disp),
        (axes[1], Z_clust, batch_clust,
         "High Concentration (Clustered)", wbc_clust),
    ]

    for ax, Z, batches, title, wbc in panel_data:
        # Background: all non-highlighted keys in light gray
        highlight_mask = np.isin(batches, HIGHLIGHT_BATCHES)
        bg_mask = ~highlight_mask

        ax.scatter(Z[bg_mask, 0], Z[bg_mask, 1], s=3, alpha=0.12,
                   color="#AAAAAA", edgecolors="none", rasterized=True)

        # Highlighted batches in distinct colors
        for batch_id, color in zip(HIGHLIGHT_BATCHES, HIGHLIGHT_COLORS):
            bmask = batches == batch_id
            if bmask.sum() == 0:
                continue
            ax.scatter(Z[bmask, 0], Z[bmask, 1], s=20, alpha=0.9,
                       color=color, edgecolors="none", zorder=5)

        ax.set_xlabel(f"PC1 ({var_explained[0]:.1%} var.)")
        ax.set_ylabel(f"PC2 ({var_explained[1]:.1%} var.)")
        ax.set_title(title, fontsize=11)

    plt.tight_layout()
    save_figure(fig, "suppfig1b_key_geometry", output_dir)

    # Print summary
    print(f"  Within-batch cosine — dispersed: {wbc_disp:.4f}, "
          f"clustered: {wbc_clust:.4f}")
    print(f"  Ratio: {wbc_clust / wbc_disp:.2f}x")


def main():
    parser = argparse.ArgumentParser(
        description="Supplemental Figure 1B: Key-space geometry visualization")
    parser.add_argument("--output-dir", type=Path, default=PAPER_OUTPUT)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    args = parser.parse_args()
    generate(args.output_dir, seed=args.seed)


if __name__ == "__main__":
    main()

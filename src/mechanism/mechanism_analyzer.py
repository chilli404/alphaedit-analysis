#!/usr/bin/env python3
"""
Post-hoc mechanism analyzer for AlphaEdit collapse characterization.

Loads saved checkpoint weights + cache_c from existing runs and computes:
1. Key-space geometry: effective rank, stable rank, cosine similarities, Gram condition
2. Projection behavior: cache_c rank, condition number, consumption ratio
3. Weight-spectrum distortion: ||W_t - W_0|| / ||W_0||, SVD changes, subspace angles

Designed to run on GPU cluster with access to:
  - Saved checkpoints (model_weights.pt, cache_c.pt) at each 1000-edit boundary
  - Base model weights (from HuggingFace or local cache)

Output: JSONL with one record per (checkpoint, layer) containing all diagnostics.

Usage:
    # Analyze all checkpoints for seed 42
    python src/mechanism_analyzer.py \\
        --seed 42 \\
        --checkpoint_base /s3-data/continual-learning/alphaedit/checkpoints/failure_curve/AlphaEdit/seed42 \\
        --model_name NousResearch/Meta-Llama-3-8B-Instruct \\
        --hparams_fname Llama3-8B.json

    # Analyze specific batch indices only
    python src/mechanism_analyzer.py \\
        --seed 42 \\
        --checkpoint_base /path/to/checkpoints \\
        --batch_indices 30 40 50 60 70 80 90 100
"""

import argparse
import json
import math
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

# Project paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
ALPHAEDIT_ROOT = PROJECT_ROOT / "vendor" / "AlphaEdit"
SRC_DIR = PROJECT_ROOT / "src"

# NOTE: Do NOT add SRC_DIR to sys.path — src/datasets/ shadows HuggingFace 'datasets'
if str(SRC_DIR / "util") not in sys.path:
    sys.path.insert(0, str(SRC_DIR / "util"))



def load_hparams(hparams_fname: str, alg_name: str = "AlphaEdit"):
    """Load AlphaEdit hyperparameters (layers, module templates) from JSON directly."""
    hparams_dir = ALPHAEDIT_ROOT / "hparams" / alg_name
    hparams_path = hparams_dir / hparams_fname
    if not hparams_path.exists():
        raise FileNotFoundError(f"Hparams file not found: {hparams_path}")
    with open(hparams_path) as f:
        data = json.load(f)

    class HParams:
        pass

    hp = HParams()
    hp.layers = data["layers"]
    hp.rewrite_module_tmp = data["rewrite_module_tmp"]
    hp.model_name = data.get("model_name", "")
    hp.nullspace_threshold = data.get("nullspace_threshold", 2e-2)
    return hp


def load_base_model_weights(model_name: str, layers: list[int], device: str = "cpu"):
    """
    Load base model directly in the main process and extract edited layer weights.
    Model is cached from first download so subsequent loads are fast.
    """
    import os
    from transformers import AutoModelForCausalLM
    from model_download import resolve_model_path

    model_name = resolve_model_path(model_name)
    print(f"  Loading base model: {model_name}")
    token = os.environ.get("HF_TOKEN")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        token=token,
        torch_dtype=torch.float32,
    ).cuda()

    base_weights = {}
    for layer_idx in layers:
        param_name = f"model.layers.{layer_idx}.mlp.down_proj.weight"
        param = dict(model.named_parameters()).get(param_name)
        if param is not None:
            base_weights[layer_idx] = param.detach().cpu().float()
            print(f"    Layer {layer_idx}: {param_name} shape={param.shape}")
        else:
            print(f"    WARNING: Could not find weight for layer {layer_idx}")

    del model
    torch.cuda.empty_cache()
    print(f"  Loaded {len(base_weights)} base weight tensors")
    return base_weights


def load_checkpoint_weights(ckpt_dir: Path, device: str = "cpu"):
    """Load model weights and cache_c from a checkpoint directory."""
    weights_path = ckpt_dir / "model_weights.pt"
    cache_path = ckpt_dir / "cache_c.pt"
    metadata_path = ckpt_dir / "metadata.json"

    if not weights_path.exists():
        return None, None, None

    weights = torch.load(weights_path, map_location=device)
    cache_c = torch.load(cache_path, map_location=device) if cache_path.exists() else None

    metadata = None
    if metadata_path.exists():
        with open(metadata_path) as f:
            metadata = json.load(f)

    return weights, cache_c, metadata


# ─── Geometry Diagnostics ────────────────────────────────────────────────────

def compute_effective_rank(svs: torch.Tensor) -> float:
    """Entropy-based effective rank: exp(H(normalized singular values))."""
    svs_pos = svs[svs > 1e-10]
    if len(svs_pos) <= 1:
        return float(len(svs_pos))
    p = svs_pos / svs_pos.sum()
    entropy = -(p * torch.log(p)).sum().item()
    return math.exp(entropy)


def compute_stable_rank(matrix: torch.Tensor) -> float:
    """Stable rank: ||A||_F^2 / ||A||_2^2."""
    fro_sq = (matrix ** 2).sum().item()
    spectral = torch.linalg.norm(matrix, ord=2).item() ** 2
    if spectral < 1e-12:
        return 0.0
    return fro_sq / spectral


def compute_numerical_rank(svs: torch.Tensor, threshold: float = 1e-5) -> int:
    """Count singular values above threshold."""
    return int((svs > threshold).sum().item())


def compute_cosine_statistics(keys: torch.Tensor) -> dict:
    """
    Compute cosine similarity statistics for a set of key vectors.

    Args:
        keys: (n_keys, dim) matrix of key vectors

    Returns:
        dict with mean_cosine, max_cosine, nearest_neighbor_mean, off_diagonal_mean
    """
    if keys.shape[0] < 2:
        return {
            "mean_cosine": 0.0,
            "max_cosine": 0.0,
            "nearest_neighbor_mean": 0.0,
            "off_diagonal_mean": 0.0,
        }

    # Normalize
    norms = keys.norm(dim=1, keepdim=True).clamp(min=1e-8)
    normed = keys / norms

    # Gram matrix of cosines
    gram = normed @ normed.T  # (n, n)

    # Mask diagonal
    n = gram.shape[0]
    mask = ~torch.eye(n, dtype=torch.bool, device=gram.device)
    off_diag = gram[mask]

    # Nearest neighbor: max cosine for each row (excluding self)
    gram_masked = gram.clone()
    gram_masked.fill_diagonal_(-1.0)
    nn_cosines = gram_masked.max(dim=1).values

    return {
        "mean_cosine": float(off_diag.mean().item()),
        "max_cosine": float(off_diag.max().item()),
        "nearest_neighbor_mean": float(nn_cosines.mean().item()),
        "off_diagonal_std": float(off_diag.std().item()),
    }


def compute_gram_condition(keys: torch.Tensor) -> float:
    """Condition number of the Gram matrix K @ K^T."""
    if keys.shape[0] < 2:
        return 1.0
    gram = keys @ keys.T
    svs = torch.linalg.svdvals(gram)
    svs_pos = svs[svs > 1e-10]
    if len(svs_pos) < 2:
        return float("inf")
    return float((svs_pos[0] / svs_pos[-1]).item())


def compute_principal_angles(subspace_a: torch.Tensor, subspace_b: torch.Tensor, k: int = 10) -> list[float]:
    """
    Compute principal angles between two subspaces (from their basis matrices).

    Args:
        subspace_a: (dim, rank_a) orthonormal basis
        subspace_b: (dim, rank_b) orthonormal basis
        k: number of angles to return

    Returns:
        List of k principal angles in radians
    """
    # Compute SVD of A^T @ B to get cosines of principal angles
    cos_angles = torch.linalg.svdvals(subspace_a.T @ subspace_b)
    cos_angles = cos_angles.clamp(-1.0, 1.0)
    angles = torch.acos(cos_angles)[:k]
    return angles.tolist()


# ─── Weight Spectrum Diagnostics ─────────────────────────────────────────────

def compute_weight_spectrum_distortion(
    W_0: torch.Tensor,
    W_t: torch.Tensor,
    r_values: list[int] = [16, 32, 64, 128],
) -> dict:
    """
    Compute weight-spectrum distortion metrics between base and edited weights.

    Args:
        W_0: Base (pre-edit) weight matrix
        W_t: Current (post-edit) weight matrix
        r_values: Rank values for subspace energy computation

    Returns:
        Dict with all distortion metrics
    """
    # Move to GPU for fast SVD (4096x14336 fits easily in GPU memory)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    W_0 = W_0.to(device).float()
    W_t = W_t.to(device).float()
    delta_W = W_t - W_0

    # Basic norms
    W0_fro = W_0.norm().item()
    Wt_fro = W_t.norm().item()
    delta_fro = delta_W.norm().item()
    relative_perturbation = delta_fro / W0_fro if W0_fro > 0 else 0.0

    # Spectral norm of update
    spectral_norm_update = torch.linalg.norm(delta_W, ord=2).item()

    # SVD of W_0 and W_t (on GPU this is seconds instead of minutes)
    U0, S0, Vh0 = torch.linalg.svd(W_0, full_matrices=False)
    Ut, St, Vht = torch.linalg.svd(W_t, full_matrices=False)

    # Top singular value changes
    n_sv = min(20, len(S0), len(St))
    sv_changes = (St[:n_sv] - S0[:n_sv]).tolist()
    sv_relative_changes = ((St[:n_sv] - S0[:n_sv]) / S0[:n_sv].clamp(min=1e-8)).tolist()

    # Stable rank
    stable_rank_0 = compute_stable_rank(W_0)
    stable_rank_t = compute_stable_rank(W_t)

    # Singular value entropy
    def sv_entropy(S):
        S_pos = S[S > 1e-10]
        if len(S_pos) <= 1:
            return 0.0
        p = S_pos / S_pos.sum()
        return -(p * torch.log(p)).sum().item()

    entropy_0 = sv_entropy(S0)
    entropy_t = sv_entropy(St)

    # Fraction of update energy in top-r original singular subspace
    # Project delta_W onto the top-r right singular vectors of W_0
    energy_in_subspace = {}
    total_update_energy = delta_fro ** 2
    for r in r_values:
        if r > Vh0.shape[0]:
            continue
        # Top-r right singular subspace of W_0
        Vh0_r = Vh0[:r, :]  # (r, dim)
        # Project: delta_W @ Vh0_r^T gives the component in that subspace
        projected = delta_W @ Vh0_r.T  # (out_dim, r)
        energy_in_r = (projected ** 2).sum().item()
        fraction = energy_in_r / total_update_energy if total_update_energy > 0 else 0.0
        energy_in_subspace[f"r{r}"] = round(fraction, 6)

    # Principal angles between top-r subspaces of W_0 and W_t
    r_for_angles = min(32, U0.shape[1], Ut.shape[1])
    angles = compute_principal_angles(U0[:, :r_for_angles], Ut[:, :r_for_angles], k=5)

    # Free GPU memory from SVD intermediates
    del W_0, W_t, delta_W, U0, S0, Vh0, Ut, St, Vht
    torch.cuda.empty_cache()

    return {
        "relative_perturbation": round(relative_perturbation, 6),
        "spectral_norm_update": round(spectral_norm_update, 4),
        "W0_frobenius": round(W0_fro, 4),
        "delta_frobenius": round(delta_fro, 4),
        "stable_rank_base": round(stable_rank_0, 2),
        "stable_rank_current": round(stable_rank_t, 2),
        "sv_entropy_base": round(entropy_0, 4),
        "sv_entropy_current": round(entropy_t, 4),
        "top5_sv_base": S0[:5].tolist(),
        "top5_sv_current": St[:5].tolist(),
        "top10_sv_change": sv_changes[:10],
        "top5_sv_relative_change": sv_relative_changes[:5],
        "update_energy_in_original_subspace": energy_in_subspace,
        "principal_angles_top32": [round(a, 6) for a in angles],
    }


# ─── Cache/Projection Diagnostics ───────────────────────────────────────────

def compute_cache_diagnostics(cache_c_layer: torch.Tensor, threshold: float = 1e-5) -> dict:
    """Compute geometry diagnostics on the accumulated covariance cache."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cache = cache_c_layer.to(device).float()
    print(f"        [cache diag] device={cache.device}, shape={cache.shape}", flush=True)

    if cache.abs().max() < 1e-12:
        return {
            "cache_numerical_rank": 0,
            "cache_effective_rank": 0.0,
            "cache_stable_rank": 0.0,
            "cache_condition": 1.0,
            "cache_trace": 0.0,
            "cache_top5_svs": [],
            "cache_1pct_sv": 0.0,
            "cache_5pct_sv": 0.0,
            "cache_10pct_sv": 0.0,
        }

    # cache_c is symmetric PSD (K @ K^T), so use eigvalsh instead of svdvals.
    # eigvalsh exploits symmetry and is O(n^2.37) vs O(n^3) for full SVD.
    eigvals = torch.linalg.eigvalsh(cache)  # ascending order
    svs = eigvals.flip(0).clamp(min=0).sqrt()  # convert to singular values, descending

    numerical_rank = compute_numerical_rank(svs, threshold)
    effective_rank = compute_effective_rank(svs)
    # Stable rank from already-computed singular values (avoid recomputing spectral norm)
    fro_sq = (svs ** 2).sum().item()
    spectral_sq = (svs[0].item()) ** 2 if len(svs) > 0 else 1e-12
    stable_rank = fro_sq / spectral_sq if spectral_sq > 1e-12 else 0.0

    # Condition number
    svs_pos = svs[svs > 1e-10]
    condition = float((svs_pos[0] / svs_pos[-1]).item()) if len(svs_pos) >= 2 else 1.0

    # Percentile singular values
    n_sv = len(svs)
    pct_1 = svs[max(0, int(0.01 * n_sv))].item() if n_sv > 0 else 0.0
    pct_5 = svs[max(0, int(0.05 * n_sv))].item() if n_sv > 0 else 0.0
    pct_10 = svs[max(0, int(0.10 * n_sv))].item() if n_sv > 0 else 0.0

    # Top eigenvalue share
    top_sv_share = (svs[0].item() / svs.sum().item()) if svs.sum() > 0 else 0.0

    return {
        "cache_numerical_rank": numerical_rank,
        "cache_effective_rank": round(effective_rank, 2),
        "cache_stable_rank": round(stable_rank, 2),
        "cache_condition": round(condition, 2),
        "cache_trace": round(cache.trace().item(), 4),
        "cache_top5_svs": svs[:5].tolist(),
        "cache_1pct_sv": round(pct_1, 6),
        "cache_5pct_sv": round(pct_5, 6),
        "cache_10pct_sv": round(pct_10, 6),
        "cache_top_sv_share": round(top_sv_share, 6),
    }


# ─── Main Analysis Loop ─────────────────────────────────────────────────────

def analyze_checkpoint(
    batch_idx: int,
    ckpt_dir: Path,
    base_weights: dict,
    hparams,
    nullspace_threshold: float,
) -> list[dict]:
    """
    Analyze a single checkpoint. Returns list of per-layer diagnostic records.
    """
    print(f"    Loading checkpoint...", end=" ", flush=True)
    weights, cache_c, metadata = load_checkpoint_weights(ckpt_dir)
    if weights is None:
        print(f"MISSING")
        return []
    print(f"loaded ({len(weights)} weights, cache={'yes' if cache_c is not None else 'no'})", flush=True)

    total_edits = (batch_idx + 1) * 100  # batch_size=100

    records = []
    for i, layer_idx in enumerate(hparams.layers):
        print(f"    layer {layer_idx} ({i+1}/{len(hparams.layers)})...", flush=True)
        record = {
            "batch_idx": batch_idx,
            "total_edits": total_edits,
            "layer_idx": layer_idx,
            "layer_position": i,
        }

        # --- Weight-spectrum distortion ---
        W_0 = base_weights.get(layer_idx)
        W_t = None

        # Find the weight for this layer in the checkpoint
        # Checkpoint weights may use different key naming
        for param_name, param_tensor in weights.items():
            if f"layers.{layer_idx}" in param_name and "down_proj" in param_name:
                W_t = param_tensor.float()
                break

        if W_0 is not None and W_t is not None:
            spectrum = compute_weight_spectrum_distortion(W_0, W_t)
            record["weight_spectrum"] = spectrum
            print(f"      spectrum done", flush=True)
        else:
            record["weight_spectrum"] = None

        # Free GPU memory before cache SVD
        torch.cuda.empty_cache()

        # --- Cache/projection diagnostics ---
        if cache_c is not None and i < len(cache_c):
            print(f"      cache svd (shape={cache_c[i].shape})...", flush=True)
            cache_diag = compute_cache_diagnostics(cache_c[i], nullspace_threshold)
            record["cache"] = cache_diag
            print(f"      cache done", flush=True)

            # Key-space geometry from cache_c
            # cache_c accumulates K @ K^T, so its SVD gives information about key space
            # But for direct key geometry we'd need the raw keys — approximate from cache
            record["cache"]["hidden_dim"] = int(cache_c[i].shape[0])
        else:
            record["cache"] = None

        records.append(record)

    return records


def main():
    parser = argparse.ArgumentParser(description="Post-hoc mechanism analysis on saved checkpoints")
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--checkpoint_base", type=str, required=True,
                        help="Base directory containing batch_N/ checkpoint subdirs")
    parser.add_argument("--model_name", type=str,
                        default="NousResearch/Meta-Llama-3-8B-Instruct")
    parser.add_argument("--hparams_fname", type=str, default="Llama3-8B.json")
    parser.add_argument("--batch_indices", type=int, nargs="*", default=None,
                        help="Specific batch indices to analyze (default: all found)")
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--skip_base_model", action="store_true",
                        help="Skip base model loading (only compute cache diagnostics)")
    args = parser.parse_args()

    ckpt_base = Path(args.checkpoint_base)
    if not ckpt_base.exists():
        print(f"ERROR: Checkpoint base not found: {ckpt_base}")
        sys.exit(1)

    # Output
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = PROJECT_ROOT / "results" / "mechanism_analysis" / f"seed{args.seed}" / "AlphaEdit"
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    output_path = output_dir / f"mechanism_seed{args.seed}_{timestamp}.jsonl"

    print("=" * 70)
    print("Post-Hoc Mechanism Analyzer")
    print(f"  Seed:          {args.seed}")
    print(f"  Checkpoints:   {ckpt_base}")
    print(f"  Model:         {args.model_name}")
    print(f"  Output:        {output_path}")
    print("=" * 70)

    # Load hparams
    print("\nLoading hparams...")
    hparams = load_hparams(args.hparams_fname)
    print(f"  Edited layers: {hparams.layers}")
    print(f"  Nullspace threshold: {hparams.nullspace_threshold}")

    # Discover checkpoints
    if args.batch_indices:
        batch_indices = args.batch_indices
    else:
        batch_indices = []
        for d in sorted(ckpt_base.iterdir()):
            if d.is_dir() and d.name.startswith("batch_"):
                try:
                    idx = int(d.name.replace("batch_", ""))
                    batch_indices.append(idx)
                except ValueError:
                    pass
        batch_indices.sort()

    print(f"\n  Found {len(batch_indices)} checkpoints: {batch_indices}")

    # Load base model weights (for spectrum distortion analysis)
    base_weights = {}
    if not args.skip_base_model:
        print("\nLoading base model weights...")
        base_weights = load_base_model_weights(args.model_name, hparams.layers)
    else:
        print("\n  Skipping base model (--skip_base_model)")

    # Analyze each checkpoint
    print("\nAnalyzing checkpoints...")
    total_records = 0

    with open(output_path, "w") as f:
        for batch_idx in batch_indices:
            ckpt_dir = ckpt_base / f"batch_{batch_idx}"
            if not ckpt_dir.exists():
                print(f"  batch_{batch_idx}: not found, skipping")
                continue

            print(f"  batch_{batch_idx} ({(batch_idx + 1) * 100} edits)...")
            records = analyze_checkpoint(
                batch_idx, ckpt_dir, base_weights, hparams,
                hparams.nullspace_threshold,
            )

            for record in records:
                record["seed"] = args.seed
                record["algorithm"] = "AlphaEdit"
                record["model"] = args.model_name
                f.write(json.dumps(record, default=lambda x: x.tolist() if hasattr(x, 'tolist') else x) + "\n")
                total_records += 1

    print(f"\n{'=' * 70}")
    print(f"Analysis complete: {total_records} records written to {output_path}")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()

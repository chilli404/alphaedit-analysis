#!/usr/bin/env python3
"""
Post-hoc retained-energy analysis for the projection capacity sweep.

For each threshold in the GPT-J projection sweep, computes:
  1. Retained edit energy:   ||P @ K_edit||_F² / ||K_edit||_F²
  2. Retained update energy: ||ΔW @ P||_F² / ||ΔW||_F²

These metrics test whether efficacy tracks the fraction of edit/update
energy that lives in the feasible subspace, rather than just raw rank.
If so, this provides a more mechanistically meaningful capacity variable
than r/d alone.

Requires:
  - Projection sweep checkpoints (model_weights.pt at each threshold)
  - null_space_project.pt (full [6, d, d] P matrix at default threshold)
  - Covariance stats .npz files (for SVD-based P reconstruction at other thresholds)
  - Key vectors OR base model (for ΔW computation)

Usage:
    uv run python src/experiments/retained_energy_analysis.py --seed 42
    uv run python src/experiments/retained_energy_analysis.py --seed 42 --batch 49
    uv run python src/experiments/retained_energy_analysis.py --seed 42 --thresholds 0.005 0.01 0.02 0.05 0.1 0.2 0.5
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import torch

_SRC_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_SRC_DIR / "util"))

from paths import get_project_root, get_checkpoint_root, get_result_root

PROJECT_ROOT = get_project_root()

GPTJ_LAYERS = [3, 4, 5, 6, 7, 8]
GPTJ_WEIGHT_KEYS = [f"transformer.h.{l}.mlp.fc_out.weight" for l in GPTJ_LAYERS]
GPTJ_STATS_DIR = None
_stats_candidates = [
    PROJECT_ROOT / "data" / "stats" / "gpt-j-6b" / "wikipedia_stats",
    PROJECT_ROOT / "vendor" / "AlphaEdit" / "data" / "stats" / "EleutherAI_gpt-j-6B" / "wikipedia_stats",
    PROJECT_ROOT / "vendor" / "AlphaEdit" / "data" / "stats" / "gpt-j-6b" / "wikipedia_stats",
    Path("/s3-data/continual-learning/alphaedit/stats/gpt-j-6b"),
]
for _candidate in _stats_candidates:
    if _candidate.exists():
        GPTJ_STATS_DIR = _candidate
        break
if GPTJ_STATS_DIR is None:
    GPTJ_STATS_DIR = _stats_candidates[0]  # fallback for error message
GPTJ_HIDDEN = 4096
GPTJ_INTERMEDIATE = 16384  # fc_out: [4096, 16384] so keys are 16384-dim


def load_p_matrix(threshold: float) -> torch.Tensor:
    """Load the full [6, d, d] P matrix, reconstructing from SVD if needed.

    At the default threshold (0.02), loads null_space_project.pt directly.
    At other thresholds, reconstructs from covariance stats using SVD.
    """
    cache_path = GPTJ_STATS_DIR / f"null_space_project_t{threshold}.pt"
    if cache_path.exists():
        return torch.load(cache_path, map_location="cpu", weights_only=False)

    default_path = GPTJ_STATS_DIR / "null_space_project.pt"
    if threshold == 0.02 and default_path.exists():
        return torch.load(default_path, map_location="cpu", weights_only=False)

    print(f"  Reconstructing P at threshold={threshold} from covariance stats...")
    P_layers = []
    for layer_idx, layer_num in enumerate(GPTJ_LAYERS):
        stats_file = GPTJ_STATS_DIR / f"transformer.h.{layer_num}.mlp.fc_out_float32_mom2_100000.npz"
        if not stats_file.exists():
            raise FileNotFoundError(f"Covariance stats not found: {stats_file}")

        data = np.load(stats_file)
        mom2 = data["mom2"]  # [d, d]

        U, S, _ = np.linalg.svd(mom2, full_matrices=True)
        S_max = S[0]
        mask = S < (threshold * S_max)
        rank = mask.sum()

        U_null = U[:, mask]  # columns corresponding to small eigenvalues
        P = U_null @ U_null.T  # projection onto null space
        P_layers.append(torch.from_numpy(P).float())
        print(f"    Layer {layer_num}: rank(P) = {rank}/{len(S)} "
              f"(r/d = {rank/len(S)*100:.1f}%)")

    P_full = torch.stack(P_layers)  # [6, d, d]
    torch.save(P_full, cache_path)
    print(f"  Saved P cache: {cache_path}")
    return P_full


def load_checkpoint_weights(ckpt_dir: Path, batch_idx: int) -> Optional[dict]:
    """Load model weights from a batch checkpoint."""
    batch_dir = ckpt_dir / f"batch_{batch_idx}"
    weights_file = batch_dir / "model_weights.pt"
    if not weights_file.exists():
        return None
    return torch.load(weights_file, map_location="cpu", weights_only=False)


def load_base_weights(model_name: str = "EleutherAI/gpt-j-6b") -> dict:
    """Load base model weights for the edited layers only.

    Tries local cache first (S3 FUSE mount), then HuggingFace.
    """
    from transformers import AutoModelForCausalLM

    s3_path = Path("/s3-data/continual-learning/models/gpt-j-6b")
    if s3_path.exists():
        model_path = str(s3_path)
    else:
        model_path = model_name

    print(f"  Loading base model from {model_path}...")
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float32,
        low_cpu_mem_usage=True,
    )

    base_weights = {}
    for layer_num in GPTJ_LAYERS:
        key = f"transformer.h.{layer_num}.mlp.fc_out.weight"
        base_weights[key] = model.state_dict()[key].cpu().clone()

    del model
    return base_weights


def compute_energy_ratios(
    P: torch.Tensor,
    delta_W: torch.Tensor,
    K_edit: Optional[torch.Tensor] = None,
) -> dict:
    """Compute projection-space energy ratios for one layer.

    Args:
        P: [d, d] projection matrix (null-space projector)
        delta_W: [d_out, d_in] weight update (ΔW = W_edited - W_base)
        K_edit: [d_in, n_keys] edit keys (optional)

    Returns:
        dict with energy ratios
    """
    results = {}

    dW_norm_sq = (delta_W ** 2).sum().item()
    results["delta_W_norm"] = dW_norm_sq ** 0.5

    # Key projection ratio ρ_K: ||P @ K_edit||_F² / ||K_edit||_F²
    # Measures how much of the incoming edit key space lies within P's range.
    # This varies by threshold and indicates whether the edit directions are
    # representable in the constrained subspace.
    if K_edit is not None:
        K_norm_sq = (K_edit ** 2).sum().item()
        if K_norm_sq > 0:
            P_K = P @ K_edit  # [d, n_keys]
            P_K_norm_sq = (P_K ** 2).sum().item()
            results["key_projection_ratio"] = P_K_norm_sq / K_norm_sq
        else:
            results["key_projection_ratio"] = float("nan")
        results["K_edit_norm"] = K_norm_sq ** 0.5

    return results


def load_key_vectors(seed: int, model_name: str = "EleutherAI/gpt-j-6b") -> Optional[np.ndarray]:
    """Load precomputed key vectors if available."""
    if "gpt-j" in model_name.lower():
        candidates = [
            get_result_root() / "key_vectors" / "gptj_full_mcf" / f"keys_seed{seed}_layer5.npz",
            get_result_root() / "key_vectors" / "gptj_full_mcf" / "keys_seed2024_layer5.npz",
            get_result_root() / "key_vectors" / "gptj_full_mcf" / "keys_seed42_layer5.npz",
        ]
    else:
        candidates = [
            get_result_root() / "key_vectors" / "full_mcf" / f"keys_seed{seed}_layer6.npz",
            get_result_root() / "key_vectors" / f"seed{seed}" / f"keys_seed{seed}.npz",
        ]
    for path in candidates:
        if path.exists():
            data = np.load(path)
            print(f"  Loaded keys from {path.name}: shape={data['keys'].shape}")
            return data["keys"]
    return None


def run_analysis(
    seed: int,
    thresholds: list[float],
    batch_idx: int,
    model_name: str = "EleutherAI/gpt-j-6b",
):
    """Run retained energy analysis across all thresholds."""

    print(f"\n{'='*70}")
    print("Retained Energy Analysis — Projection Capacity Sweep")
    print(f"  Seed: {seed}")
    print(f"  Batch: {batch_idx}")
    print(f"  Thresholds: {thresholds}")
    print(f"{'='*70}\n")

    ckpt_root = get_checkpoint_root()

    # Load base weights (needed to compute ΔW)
    base_weights = load_base_weights(model_name)

    # Load key vectors (if available)
    all_keys = load_key_vectors(seed, model_name)

    results = []

    for threshold in thresholds:
        print(f"\n--- Threshold: {threshold} ---")

        # Resolve checkpoint directory for this threshold
        alg_tag = f"AlphaEdit-t{threshold}"
        ckpt_dir = ckpt_root / "failure_curve" / "gpt-j-6b" / alg_tag / f"seed{seed}"

        if not ckpt_dir.exists():
            # Try without model tag (legacy)
            ckpt_dir = ckpt_root / "failure_curve" / alg_tag / f"seed{seed}"

        if not ckpt_dir.exists():
            print(f"  SKIP: checkpoint dir not found at {ckpt_dir}")
            continue

        # Load checkpoint weights
        edited_weights = load_checkpoint_weights(ckpt_dir, batch_idx)
        if edited_weights is None:
            print(f"  SKIP: no checkpoint at batch {batch_idx}")
            continue

        # Load P matrix for this threshold
        P_full = load_p_matrix(threshold)

        # Get rank info from P
        threshold_result = {
            "threshold": threshold,
            "batch_idx": batch_idx,
            "total_edits": (batch_idx + 1) * 100,
            "layers": {},
        }

        for layer_idx, layer_num in enumerate(GPTJ_LAYERS):
            weight_key = f"transformer.h.{layer_num}.mlp.fc_out.weight"

            if weight_key not in edited_weights:
                continue
            if weight_key not in base_weights:
                continue

            W_edited = edited_weights[weight_key].float()
            W_base = base_weights[weight_key].float()
            delta_W = W_edited - W_base
            P = P_full[layer_idx]

            # Compute rank of P (trace of idempotent matrix = rank)
            rank = int(torch.trace(P).round().item())
            d = P.shape[0]
            r_over_d = rank / d

            # Get edit keys for this layer's batch if available
            K_edit = None
            if all_keys is not None and layer_num == 5:
                # Use keys from the batch range [batch_start, batch_end)
                batch_start = batch_idx * 100
                batch_end = min((batch_idx + 1) * 100, len(all_keys))
                if batch_end <= len(all_keys):
                    K_edit = torch.from_numpy(all_keys[batch_start:batch_end].T).float()

            energy = compute_energy_ratios(P, delta_W, K_edit)
            energy["rank"] = rank
            energy["total_dims"] = d
            energy["r_over_d"] = r_over_d

            threshold_result["layers"][str(layer_num)] = energy
            rho_k_str = f", ρ_K={energy['key_projection_ratio']:.4f}" if 'key_projection_ratio' in energy else ""
            print(f"  Layer {layer_num}: rank={rank}/{d} ({r_over_d*100:.1f}%){rho_k_str}")

        # Compute layer-averaged metrics
        layer_values = list(threshold_result["layers"].values())
        if layer_values:
            avg_rank = np.mean([v["r_over_d"] for v in layer_values])
            threshold_result["avg_r_over_d"] = float(avg_rank)

            key_ratios = [v["key_projection_ratio"] for v in layer_values
                         if "key_projection_ratio" in v and not np.isnan(v["key_projection_ratio"])]
            if key_ratios:
                threshold_result["avg_key_projection_ratio"] = float(np.mean(key_ratios))

        results.append(threshold_result)

    # Save results
    out_dir = get_result_root() / "retained_energy_analysis" / f"seed{seed}"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"energy_batch{batch_idx}.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  Results saved: {out_path}")

    # Print correlation summary
    if len(results) >= 3:
        r_over_d = [r["avg_r_over_d"] for r in results if "avg_r_over_d" in r]
        key_ratios = [r["avg_key_projection_ratio"] for r in results if "avg_key_projection_ratio" in r]
        if len(r_over_d) == len(key_ratios) and len(r_over_d) >= 3:
            from scipy.stats import spearmanr
            rho, p = spearmanr(r_over_d, key_ratios)
            print(f"\n  Spearman(r/d, ρ_K) = {rho:.3f} (p={p:.4f})")
            print(f"  Interpretation: {'ρ_K tracks rank monotonically' if rho > 0.8 else 'ρ_K provides information beyond raw rank'}")

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Post-hoc retained energy analysis for projection capacity sweep"
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch", type=int, default=49,
                        help="Batch index to analyze (default 49 = 5000 edits)")
    parser.add_argument("--thresholds", type=float, nargs="+",
                        default=[0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5],
                        help="Nullspace thresholds to analyze")
    parser.add_argument("--model_name", type=str, default="EleutherAI/gpt-j-6b")
    parser.add_argument("--all_batches", action="store_true",
                        help="Run analysis at all available checkpoint batches")
    args = parser.parse_args()

    if args.all_batches:
        # Find all batch indices available for the first threshold
        ckpt_root = get_checkpoint_root()
        first_threshold = args.thresholds[0]
        alg_tag = f"AlphaEdit-t{first_threshold}"
        ckpt_dir = ckpt_root / "failure_curve" / "gpt-j-6b" / alg_tag / f"seed{args.seed}"
        if ckpt_dir.exists():
            batches = sorted(
                int(d.name.split("_")[1])
                for d in ckpt_dir.glob("batch_*")
                if d.is_dir() and (d / "model_weights.pt").exists()
            )
            print(f"  Found {len(batches)} checkpoints: {batches}")
            for batch_idx in batches:
                run_analysis(args.seed, args.thresholds, batch_idx, args.model_name)
        else:
            print(f"ERROR: No checkpoints found at {ckpt_dir}")
            sys.exit(1)
    else:
        run_analysis(args.seed, args.thresholds, args.batch, args.model_name)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Polynomial-Kernel Memory Diagnostic — Stage 2: Gram Matrix Analysis

Pure CPU analysis (no model loading, no source injection). Loads extracted keys
from Stage 1 (.pt file), constructs Gram matrices under linear and degree-2
polynomial kernels, and computes capacity/separation metrics.

Scientific Question:
    Are AlphaEdit/MEMIT failure modes consistent with a linear key-space
    capacity bottleneck? Would a polynomial kernel create more linearly
    independent key directions (as predicted by ATLAS-style capacity theory)?

Gram Matrices:
    G_linear = K^T @ K                    (standard inner product)
    G_poly2  = (1 + K^T @ K)^2            (element-wise, degree-2 polynomial kernel)

Metrics per Gram matrix:
    eff_rank      - exp(-Σ p_i log p_i), spectral entropy
    stable_rank   - (Σλ)² / Σ(λ²), participation ratio
    num_rank      - #{λ_i / λ_max > threshold}
    mean_offdiag  - mean(|C[i,j]|) for i≠j, C = cosine-normalized G
    max_nn_sim    - max nearest-neighbor cosine similarity
    mean_nn_sim   - mean nearest-neighbor cosine similarity
    condition_num - λ_max / λ_min_nonzero (capped at 1e12)

Analysis Windows:
    per_batch   - Keys within a single batch
    cumulative  - All keys up to batch t
    sliding     - Last W batches
    by_coupling - Keys grouped by coupling type metadata

Usage:
    python src/polykernel_diagnostic.py \
        --keys_file results/polykernel_diagnostic/keys_AlphaEdit_seed42.pt \
        --rank_threshold 1e-5 \
        --window_size 5
"""

import argparse
import json
import math
import sys
from pathlib import Path

import torch


def compute_gram_metrics(K: torch.Tensor, rank_threshold: float = 1e-5) -> dict:
    """
    Compute Gram matrix metrics for a set of keys.

    Args:
        K: (d_model, n_keys) — columns are key vectors, float32
        rank_threshold: eigenvalue ratio threshold for num_rank

    Returns:
        dict with linear and poly2 metrics
    """
    n_keys = K.shape[1]
    if n_keys < 2:
        return {"linear": {}, "poly2": {}, "n_keys": n_keys}

    # Gram matrices
    G_linear = K.T @ K  # (n_keys, n_keys)
    G_poly2 = (1.0 + G_linear).pow(2)  # element-wise degree-2 polynomial kernel

    linear_metrics = _gram_metrics(G_linear, rank_threshold)
    poly2_metrics = _gram_metrics(G_poly2, rank_threshold)

    # Compute ratios (poly2 / linear) for key metrics
    ratio = {}
    for key in ["eff_rank", "stable_rank", "num_rank", "mean_nn_sim", "mean_offdiag"]:
        l_val = linear_metrics.get(key, 0)
        p_val = poly2_metrics.get(key, 0)
        if l_val > 0:
            ratio[key] = p_val / l_val
        else:
            ratio[key] = float("inf") if p_val > 0 else 1.0

    return {
        "linear": linear_metrics,
        "poly2": poly2_metrics,
        "ratio": ratio,
        "n_keys": n_keys,
    }


def _gram_metrics(G: torch.Tensor, rank_threshold: float) -> dict:
    """Compute all metrics for a single Gram matrix."""
    n = G.shape[0]

    # Eigendecomposition (G is symmetric PSD)
    eigenvalues = torch.linalg.eigvalsh(G)  # ascending order
    eigenvalues = eigenvalues.flip(0)  # descending order
    eigenvalues = eigenvalues.clamp(min=0)  # numerical PSD enforcement

    lambda_max = eigenvalues[0].item()
    if lambda_max < 1e-12:
        return {
            "eff_rank": 0.0,
            "stable_rank": 0.0,
            "num_rank": 0,
            "mean_offdiag": 0.0,
            "max_nn_sim": 0.0,
            "mean_nn_sim": 0.0,
            "condition_num": 0.0,
        }

    # Effective rank (spectral entropy)
    p = eigenvalues / eigenvalues.sum()
    p_positive = p[p > 1e-30]
    eff_rank = torch.exp(-torch.sum(p_positive * torch.log(p_positive))).item()

    # Stable rank (participation ratio)
    trace = eigenvalues.sum().item()
    frob_sq = (eigenvalues ** 2).sum().item()
    stable_rank = (trace ** 2) / frob_sq if frob_sq > 0 else 0.0

    # Numerical rank
    num_rank = int((eigenvalues / lambda_max > rank_threshold).sum().item())

    # Cosine similarity matrix (normalize G to correlation matrix)
    diag = G.diag().clamp(min=1e-12)
    diag_inv_sqrt = 1.0 / diag.sqrt()
    C = G * diag_inv_sqrt.unsqueeze(0) * diag_inv_sqrt.unsqueeze(1)

    # Off-diagonal statistics
    mask = ~torch.eye(n, dtype=torch.bool)
    offdiag = C[mask]
    mean_offdiag = offdiag.abs().mean().item()

    # Nearest-neighbor similarities
    C_nn = C.clone()
    C_nn.fill_diagonal_(float("-inf"))
    nn_sims = C_nn.max(dim=1).values
    max_nn_sim = nn_sims.max().item()
    mean_nn_sim = nn_sims.mean().item()

    # Condition number
    nonzero_eigs = eigenvalues[eigenvalues / lambda_max > rank_threshold]
    if len(nonzero_eigs) > 1:
        condition_num = min(nonzero_eigs[0].item() / nonzero_eigs[-1].item(), 1e12)
    else:
        condition_num = 1.0

    return {
        "eff_rank": round(eff_rank, 4),
        "stable_rank": round(stable_rank, 4),
        "num_rank": num_rank,
        "mean_offdiag": round(mean_offdiag, 6),
        "max_nn_sim": round(max_nn_sim, 6),
        "mean_nn_sim": round(mean_nn_sim, 6),
        "condition_num": round(condition_num, 2),
    }


def analyze_per_batch(keys: dict, batch_tags: list, layers: list, rank_threshold: float) -> list:
    """Analyze keys within each individual batch."""
    results = []
    # Determine keys per batch from batch_tags
    for tag in batch_tags:
        batch_idx = tag["batch_idx"]
        n_keys_per_batch = len(tag["case_ids"])

        batch_result = {"batch_idx": batch_idx, "layers": {}}
        for layer in layers:
            layer_keys = keys[layer]  # (d_model, total_keys)
            start = batch_idx * n_keys_per_batch
            end = start + n_keys_per_batch
            if end > layer_keys.shape[1]:
                break
            K_batch = layer_keys[:, start:end].float()
            batch_result["layers"][str(layer)] = compute_gram_metrics(K_batch, rank_threshold)

        results.append(batch_result)
    return results


def analyze_cumulative(keys: dict, batch_tags: list, layers: list, rank_threshold: float) -> list:
    """Analyze all keys up to batch t (cumulative growth)."""
    results = []
    if not batch_tags:
        return results

    n_keys_per_batch = len(batch_tags[0]["case_ids"])

    for t, tag in enumerate(batch_tags):
        batch_idx = tag["batch_idx"]
        end = (t + 1) * n_keys_per_batch

        batch_result = {"batch_idx": batch_idx, "cumulative_keys": end, "layers": {}}
        for layer in layers:
            layer_keys = keys[layer]
            if end > layer_keys.shape[1]:
                end = layer_keys.shape[1]
            K_cum = layer_keys[:, :end].float()
            batch_result["layers"][str(layer)] = compute_gram_metrics(K_cum, rank_threshold)

        results.append(batch_result)
    return results


def analyze_sliding(keys: dict, batch_tags: list, layers: list, rank_threshold: float, window_size: int) -> list:
    """Analyze keys from the last W batches (sliding window)."""
    results = []
    if not batch_tags:
        return results

    n_keys_per_batch = len(batch_tags[0]["case_ids"])

    for t, tag in enumerate(batch_tags):
        batch_idx = tag["batch_idx"]
        w_start = max(0, t + 1 - window_size)
        start = w_start * n_keys_per_batch
        end = (t + 1) * n_keys_per_batch

        batch_result = {
            "batch_idx": batch_idx,
            "window_batches": t + 1 - w_start,
            "window_keys": end - start,
            "layers": {},
        }
        for layer in layers:
            layer_keys = keys[layer]
            if end > layer_keys.shape[1]:
                end = layer_keys.shape[1]
            K_win = layer_keys[:, start:end].float()
            batch_result["layers"][str(layer)] = compute_gram_metrics(K_win, rank_threshold)

        results.append(batch_result)
    return results


def analyze_by_coupling_type(keys: dict, batch_tags: list, layers: list, rank_threshold: float) -> dict:
    """Group keys by coupling type and analyze each group."""
    # Build per-type key indices
    type_indices = {}
    key_offset = 0
    for tag in batch_tags:
        n_keys = len(tag["case_ids"])
        for i, meta in enumerate(tag.get("coupling_meta", [])):
            ctype = meta.get("coupling_type_name", "unknown") if meta else "unknown"
            if ctype not in type_indices:
                type_indices[ctype] = []
            type_indices[ctype].append(key_offset + i)
        key_offset += n_keys

    if not type_indices or all(t == "unknown" for t in type_indices):
        return {}

    results = {}
    for ctype, indices in type_indices.items():
        if len(indices) < 2:
            continue
        idx_tensor = torch.tensor(indices, dtype=torch.long)
        type_result = {"n_keys": len(indices), "layers": {}}
        for layer in layers:
            layer_keys = keys[layer]
            valid_idx = idx_tensor[idx_tensor < layer_keys.shape[1]]
            if len(valid_idx) < 2:
                continue
            K_type = layer_keys[:, valid_idx].float()
            type_result["layers"][str(layer)] = compute_gram_metrics(K_type, rank_threshold)
        results[ctype] = type_result

    return results


def analyze_cross_group_separation(keys: dict, batch_tags: list, layers: list, rank_threshold: float) -> dict:
    """Compare within-type vs between-type similarity."""
    # Build per-type key indices
    type_indices = {}
    key_offset = 0
    for tag in batch_tags:
        n_keys = len(tag["case_ids"])
        for i, meta in enumerate(tag.get("coupling_meta", [])):
            ctype = meta.get("coupling_type_name", "unknown") if meta else "unknown"
            if ctype not in type_indices:
                type_indices[ctype] = []
            type_indices[ctype].append(key_offset + i)
        key_offset += n_keys

    # Need at least 2 groups with >= 2 keys each
    valid_types = {t: idx for t, idx in type_indices.items() if len(idx) >= 2 and t != "unknown"}
    if len(valid_types) < 2:
        return {}

    results = {}
    for layer in layers:
        layer_keys = keys[layer].float()  # (d_model, total)
        n_total = layer_keys.shape[1]

        within_sims = []
        between_sims = []

        type_names = list(valid_types.keys())
        for i, t1 in enumerate(type_names):
            idx1 = torch.tensor([x for x in valid_types[t1] if x < n_total])
            if len(idx1) < 2:
                continue
            K1 = layer_keys[:, idx1]
            # Normalize columns
            K1_norm = K1 / K1.norm(dim=0, keepdim=True).clamp(min=1e-12)

            # Within-group: pairwise cosine sim
            C_within = K1_norm.T @ K1_norm
            mask_within = ~torch.eye(len(idx1), dtype=torch.bool)
            within_sims.extend(C_within[mask_within].tolist())

            # Between-group
            for j in range(i + 1, len(type_names)):
                t2 = type_names[j]
                idx2 = torch.tensor([x for x in valid_types[t2] if x < n_total])
                if len(idx2) < 2:
                    continue
                K2 = layer_keys[:, idx2]
                K2_norm = K2 / K2.norm(dim=0, keepdim=True).clamp(min=1e-12)
                C_between = K1_norm.T @ K2_norm
                between_sims.extend(C_between.flatten().tolist())

        if within_sims and between_sims:
            results[str(layer)] = {
                "mean_within_sim": round(sum(within_sims) / len(within_sims), 6),
                "mean_between_sim": round(sum(between_sims) / len(between_sims), 6),
                "separation_ratio": round(
                    (sum(within_sims) / len(within_sims)) /
                    max(abs(sum(between_sims) / len(between_sims)), 1e-10),
                    4
                ),
                "n_within_pairs": len(within_sims),
                "n_between_pairs": len(between_sims),
            }

    return results


def generate_summary(per_batch: list, cumulative: list, by_coupling: dict) -> dict:
    """Generate interpretive summary from results."""
    # Check if poly2 consistently improves effective rank in later batches
    if not cumulative:
        return {"conclusion": "B", "evidence": "No data to analyze."}

    # Look at the last few cumulative entries
    late_entries = cumulative[-3:] if len(cumulative) >= 3 else cumulative
    eff_rank_ratios = []
    nn_sim_improvements = []

    for entry in late_entries:
        for layer_data in entry.get("layers", {}).values():
            ratio = layer_data.get("ratio", {})
            if "eff_rank" in ratio:
                eff_rank_ratios.append(ratio["eff_rank"])
            linear_nn = layer_data.get("linear", {}).get("mean_nn_sim", 0)
            poly2_nn = layer_data.get("poly2", {}).get("mean_nn_sim", 0)
            if linear_nn > 0:
                nn_sim_improvements.append(poly2_nn < linear_nn)

    if not eff_rank_ratios:
        return {"conclusion": "B", "evidence": "Insufficient data for conclusion."}

    mean_eff_ratio = sum(eff_rank_ratios) / len(eff_rank_ratios)
    nn_improve_frac = sum(nn_sim_improvements) / max(len(nn_sim_improvements), 1)

    # Check for coupling-type effects
    coupling_effect = bool(by_coupling) and len(by_coupling) >= 2

    if mean_eff_ratio > 1.3 and nn_improve_frac > 0.5:
        conclusion = "A"
        evidence = (
            f"Linear bottleneck supported: poly2/linear eff_rank ratio = {mean_eff_ratio:.3f} "
            f"(>1.3) in late batches, and poly2 reduces mean_nn_sim in "
            f"{nn_improve_frac*100:.0f}% of layer/batch combos."
        )
    elif mean_eff_ratio < 1.05 or (mean_eff_ratio < 1.3 and nn_improve_frac < 0.3):
        conclusion = "B"
        evidence = (
            f"No evidence of linear bottleneck: poly2/linear eff_rank ratio = {mean_eff_ratio:.3f}, "
            f"nn_sim improvement fraction = {nn_improve_frac*100:.0f}%."
        )
    else:
        conclusion = "C"
        evidence = (
            f"Mixed evidence: poly2/linear eff_rank ratio = {mean_eff_ratio:.3f}, "
            f"nn_sim improvement fraction = {nn_improve_frac*100:.0f}%."
        )
        if coupling_effect:
            evidence += " Coupling-type grouping present but effect varies across groups."

    return {"conclusion": conclusion, "evidence": evidence}


def run(args: argparse.Namespace) -> None:
    """Load keys and run Gram matrix analysis."""
    keys_file = Path(args.keys_file)
    if not keys_file.exists():
        print(f"ERROR: Keys file not found: {keys_file}")
        sys.exit(1)

    print(f"Loading keys from {keys_file}...")
    data = torch.load(keys_file, map_location="cpu")

    keys = data["keys"]  # {layer_int: Tensor(d_model, n_keys)}
    batch_tags = data["batch_tags"]
    metadata = data["metadata"]

    layers = sorted(keys.keys())
    print(f"  Layers: {layers}")
    for layer in layers:
        print(f"    Layer {layer}: {keys[layer].shape}")
    print(f"  Batches: {len(batch_tags)}")

    rank_threshold = args.rank_threshold
    window_size = args.window_size

    print(f"\nAnalyzing (rank_threshold={rank_threshold}, window_size={window_size})...")

    # Run all analysis windows
    print("  Per-batch analysis...")
    per_batch = analyze_per_batch(keys, batch_tags, layers, rank_threshold)

    print("  Cumulative analysis...")
    cumulative = analyze_cumulative(keys, batch_tags, layers, rank_threshold)

    print("  Sliding window analysis...")
    sliding = analyze_sliding(keys, batch_tags, layers, rank_threshold, window_size)

    print("  Coupling-type analysis...")
    by_coupling = analyze_by_coupling_type(keys, batch_tags, layers, rank_threshold)

    print("  Cross-group separation...")
    cross_group = analyze_cross_group_separation(keys, batch_tags, layers, rank_threshold)

    # Summary
    print("  Generating summary...")
    summary = generate_summary(per_batch, cumulative, by_coupling)

    # Output
    output = {
        "metadata": metadata,
        "analysis_params": {
            "rank_threshold": rank_threshold,
            "window_size": window_size,
        },
        "per_batch": per_batch,
        "cumulative": cumulative,
        "sliding": sliding,
        "by_coupling_type": by_coupling,
        "cross_group_separation": cross_group,
        "summary": summary,
    }

    # Determine output path
    alg_name = metadata.get("alg_name", "unknown")
    seed = metadata.get("seed", 0)
    output_path = keys_file.parent / f"analysis_{alg_name}_seed{seed}.json"

    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)

    print(f"\n{'=' * 70}")
    print("Polynomial-Kernel Diagnostic Complete")
    print(f"  Output:     {output_path}")
    print(f"  Conclusion: {summary['conclusion']}")
    print(f"  Evidence:   {summary['evidence']}")
    print(f"{'=' * 70}")


def main():
    parser = argparse.ArgumentParser(
        description="Polynomial-kernel diagnostic: Gram matrix analysis of edit keys"
    )

    parser.add_argument("--keys_file", type=str, required=True,
                        help="Path to .pt file from polykernel_key_extractor.py")
    parser.add_argument("--rank_threshold", type=float, default=1e-5,
                        help="Eigenvalue ratio threshold for numerical rank (default: 1e-5)")
    parser.add_argument("--window_size", type=int, default=5,
                        help="Number of batches in sliding window (default: 5)")

    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()

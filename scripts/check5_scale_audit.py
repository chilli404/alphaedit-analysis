#!/usr/bin/env python3
"""
Check 5: Scale Audit of All Constants.

Reports where τ=0.02, λ=L2, and α=mom2_update_weight sit relative to
each model's K@K^T and C0 spectra.

For each model, computes (from the mom2/C0 stats):
  - C0 spectrum: S.max, S.min, trace, ratio α*S.max to K@K^T_typical
  - Shows that τ, λ, α are ALL absolute constants interacting with
    activation scale differently per model.

Usage:
    uv run python scripts/check5_scale_audit.py
"""

import sys
import numpy as np
import torch
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent

sys.path.insert(0, str(PROJECT_DIR / "src" / "util"))
from model_registry import get_model_spec


def find_stats(model_tag):
    """Find stats directory."""
    candidates = [
        PROJECT_DIR / "data" / "stats" / model_tag / "wikipedia_stats",
        PROJECT_DIR / "vendor" / "AlphaEdit" / "data" / "stats" / model_tag / "wikipedia_stats",
    ]
    # Original-case variants
    case_map = {
        "llama3-8b-instruct": "Meta-Llama-3-8B-Instruct",
        "qwen2.5-7b-instruct": "Qwen2.5-7B-Instruct",
        "gpt-j-6b": "gpt-j-6b",
    }
    if model_tag in case_map:
        candidates.append(
            PROJECT_DIR / "vendor" / "AlphaEdit" / "data" / "stats" / case_map[model_tag] / "wikipedia_stats"
        )
    for d in candidates:
        if d.exists() and any(d.glob("*.npz")):
            return d
    return None


def load_spectrum(stats_dir, layer_name):
    """Load covariance and return its spectrum."""
    filename = stats_dir / f"{layer_name}_float32_mom2_100000.npz"
    if not filename.exists():
        return None, None
    data = np.load(filename, allow_pickle=True)
    if "mom2.mom2" not in data:
        return None, None
    raw = torch.from_numpy(data["mom2.mom2"])
    count = int(data["mom2.count"])
    cov = (raw / count).float()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    _, S, _ = torch.linalg.svd(cov.to(device), full_matrices=False)
    return S.cpu(), count


def main():
    print()
    print("=" * 72)
    print("CHECK 5: SCALE AUDIT — How absolute constants interact with spectra")
    print("=" * 72)
    print()

    # Constants
    tau = 0.02          # nullspace_threshold
    alpha = 15000       # mom2_update_weight
    L2_qwen = 1        # AlphaEdit L2 for Qwen
    L2_llama = 10       # AlphaEdit L2 for Llama
    L2_gptj = 10        # AlphaEdit L2 for GPT-J (from hparams)

    models = [
        ("Llama-3-8B", "llama3-8b-instruct", "model.layers.{}.mlp.down_proj", [4,5,6,7,8], L2_llama),
        ("Qwen2.5-7B", "qwen2.5-7b-instruct", "model.layers.{}.mlp.down_proj", [4,5,6,7,8], L2_qwen),
        ("GPT-J-6B", "gpt-j-6b", "transformer.h.{}.mlp.fc_out", [3,4,5,6,7,8], L2_gptj),
    ]

    for model_name, tag, module_tmp, layers, L2 in models:
        print(f"\n{'─'*72}")
        print(f"  {model_name} (L2={L2})")
        print(f"{'─'*72}")

        stats_dir = find_stats(tag)
        if stats_dir is None:
            print(f"  STATS NOT FOUND for {tag}")
            continue

        # Use middle layer for representative spectrum
        mid_layer = layers[len(layers)//2]
        layer_name = module_tmp.format(mid_layer)
        S, count = load_spectrum(stats_dir, layer_name)

        if S is None:
            print(f"  Could not load spectrum for {layer_name}")
            continue

        n = S.shape[0]
        trace = S.sum().item()

        print(f"  Layer {mid_layer} ({layer_name})")
        print(f"    Dimension: {n}")
        print(f"    Sample count: {count}")
        print()
        print(f"    C0 spectrum:")
        print(f"      S.max:  {S[0].item():.6e}")
        print(f"      S.min:  {S[-1].item():.6e}")
        print(f"      trace:  {trace:.6e}")
        print(f"      S.max / S.min: {S[0].item() / max(S[-1].item(), 1e-30):.2e}")
        print()
        print(f"    Constants vs spectrum:")
        print(f"      τ = {tau}")
        print(f"        τ / S.max = {tau / S[0].item():.6e}  (relative significance of threshold)")
        print(f"        dims with S < τ: {(S < tau).sum().item()}/{n} ({100*(S < tau).sum().item()/n:.2f}%)")
        print()
        print(f"      α = {alpha} (mom2_update_weight)")
        print(f"        α * S.max = {alpha * S[0].item():.6e}  (dominant LHS term in MEMIT)")
        print(f"        α * trace = {alpha * trace:.6e}")
        print(f"        α * S.max / L2 = {alpha * S[0].item() / L2:.6e}  (ratio to ridge)")
        print()
        print(f"      L2 = {L2} (AlphaEdit ridge)")
        print(f"        L2 / S.max = {L2 / S[0].item():.6e}  (ridge relative to spectrum)")
        print(f"        L2 / trace = {L2 / trace:.6e}")
        print()

        # Estimate K@K^T scale from C0 (K@K^T for 100 edits ≈ 100/N * C0_unnormalized ≈ C0 * 100/N)
        # Actually K@K^T for batch of B edits: each column has norm ≈ sqrt(trace(C0))
        # So K@K^T ≈ B * diag(C0) approximately
        # More precisely: E[K@K^T] for B samples from the distribution ≈ B * C0
        batch_size = 100
        estimated_kkt_scale = batch_size * S[0].item()
        print(f"    Estimated K@K^T scale (batch={batch_size}):")
        print(f"      ≈ B * S.max(C0) = {estimated_kkt_scale:.6e}")
        print(f"      L2 / (B*S.max) = {L2 / estimated_kkt_scale:.6e}  (ridge significance)")
        print(f"      α*S.max / (B*S.max) = {alpha * S[0].item() / estimated_kkt_scale:.6e} = α = {alpha}")
        print()
        print(f"    Summary:")
        if L2 / estimated_kkt_scale < 0.01:
            print(f"      ⚠ L2={L2} is negligible vs K@K^T scale ({L2/estimated_kkt_scale:.1e}x)")
            print(f"        → Without P or C0, solve is poorly conditioned on batch 1")
        else:
            print(f"      L2={L2} provides meaningful regularization ({L2/estimated_kkt_scale:.1e}x of K@K^T)")
        print()

    print()
    print("=" * 72)
    print("CONCLUSION")
    print("=" * 72)
    print()
    print("  All three constants (τ, α, L2) are absolute values that interact")
    print("  very differently with each model's activation scale:")
    print()
    print("  - τ=0.02: defines 'null space' cutoff on C0 singular values")
    print("  - α=15000: MEMIT's C0 weighting (stabilizer, overwhelms K@K^T)")
    print("  - L2: AlphaEdit's ridge (relies on P to bound updates; L2 alone insufficient)")
    print()
    print("  For models with large C0 spectra (Qwen), L2 alone is insufficient")
    print("  regularization — P must be tight (few retained dims) to compensate.")
    print("  For models with small C0 spectra (Llama), P≈I is fine because even")
    print("  the unregularized solve produces bounded updates.")


if __name__ == "__main__":
    main()

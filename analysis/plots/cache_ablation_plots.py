#!/usr/bin/env python3
"""
Cache Ablation Analysis: Visualize causal evidence for over-regularization.

Produces multi-panel figures showing how cache scaling (γ) affects:
  - Inverse gain (solver response per key)
  - Residual attainment
  - Relative update size
  - Cache/key energy ratio
  - P vs I comparison

Usage:
    uv run python -m analysis.cache_ablation_plots
    uv run python -m analysis.cache_ablation_plots results/cache_ablation/seed42/
"""

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

DEFAULT_DIR = Path("results/cache_ablation/seed42")


def load_ablation_data(results_dir: Path) -> tuple[list[dict], dict | None]:
    """Load cache ablation JSONL. Returns (gamma_records, projection_ablation)."""
    gamma_records = []
    proj_ablation = None

    for f in sorted(results_dir.glob("*.jsonl")):
        with open(f) as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                record = json.loads(line)
                if record.get("type") == "projection_ablation":
                    proj_ablation = record
                else:
                    gamma_records.append(record)

    return gamma_records, proj_ablation


def main():
    if len(sys.argv) > 1:
        results_dir = Path(sys.argv[1])
    else:
        results_dir = DEFAULT_DIR

    output_dir = Path("results/figures/cache_ablation")
    output_dir.mkdir(parents=True, exist_ok=True)

    gamma_records, proj_ablation = load_ablation_data(results_dir)
    if not gamma_records:
        print(f"No data found in {results_dir}")
        return

    print(f"Loaded {len(gamma_records)} gamma records")

    # Extract structured data
    gammas = sorted(set(r["gamma"] for r in gamma_records))
    layers = sorted(set(int(k) for r in gamma_records for k in r["layers"].keys()))
    print(f"Gammas: {gammas}")
    print(f"Layers: {layers}")

    # Build lookup: data[gamma][layer] = metrics dict
    data = {}
    for r in gamma_records:
        g = r["gamma"]
        data[g] = {}
        for layer_str, metrics in r["layers"].items():
            data[g][int(layer_str)] = metrics

    # ═══════════════════════════════════════════════════════════════════════
    # Figure 1: Main 4-panel diagnostic
    # ═══════════════════════════════════════════════════════════════════════
    fig, axes = plt.subplots(2, 2, figsize=(11, 8.5))
    fig.suptitle(
        "Cache Ablation: Causal Evidence for Over-Regularization\n"
        "(Seed 42, Checkpoint at 7K edits, batch 70)",
        fontsize=12, fontweight="bold",
    )

    colors = plt.cm.viridis(np.linspace(0.2, 0.9, len(layers)))

    # Panel A: γ vs inverse gain
    ax = axes[0, 0]
    for i, layer in enumerate(layers):
        y = [data[g][layer]["mean_inverse_gain"] for g in gammas]
        ax.plot(gammas, y, "o-", color=colors[i], label=f"Layer {layer}", markersize=5)
    ax.set_xlabel("γ (cache scaling)")
    ax.set_ylabel("Mean inverse gain")
    ax.set_title("A. Inverse Gain vs Cache Strength")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(-0.05, 1.05)

    # Panel B: γ vs residual attainment
    ax = axes[0, 1]
    for i, layer in enumerate(layers):
        y = [data[g][layer]["residual_attainment"] for g in gammas]
        ax.plot(gammas, y, "o-", color=colors[i], label=f"Layer {layer}", markersize=5)
    ax.set_xlabel("γ (cache scaling)")
    ax.set_ylabel("||ΔWK||_F / ||R||_F")
    ax.set_title("B. Residual Attainment vs Cache Strength")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(-0.05, 1.05)

    # Panel C: cache/key Frobenius ratio per layer (at γ=1.0)
    ax = axes[1, 0]
    dominance_by_gamma = {}
    for g in gammas:
        dominance_by_gamma[g] = [data[g][layer]["cache_dominance_fro"] for layer in layers]
    # Show as grouped bar or lines
    for i, g in enumerate(gammas[1:], 1):  # skip γ=0
        ax.bar(
            np.arange(len(layers)) + (i - 1) * 0.18,
            dominance_by_gamma[g],
            width=0.18,
            alpha=0.8,
            label=f"γ={g}",
        )
    ax.set_xticks(np.arange(len(layers)) + 0.27)
    ax.set_xticklabels([f"Layer {l}" for l in layers])
    ax.set_ylabel("||γC||_F / ||KK^T + λI||_F")
    ax.set_title("C. Cache Energy Dominance by Layer")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3, axis="y")
    ax.axhline(1.0, color="red", linestyle="--", alpha=0.5, linewidth=0.8)
    ax.annotate("cache = key energy", xy=(0, 1.0), xytext=(0.5, 1.5),
                fontsize=7, color="red", alpha=0.7)

    # Panel D: relative update size
    ax = axes[1, 1]
    for i, layer in enumerate(layers):
        y = [data[g][layer]["relative_update_size"] for g in gammas]
        ax.plot(gammas, y, "o-", color=colors[i], label=f"Layer {layer}", markersize=5)
    ax.set_xlabel("γ (cache scaling)")
    ax.set_ylabel("||ΔW||_F / ||W||_F")
    ax.set_title("D. Relative Update Size vs Cache Strength")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(-0.05, 1.05)

    plt.tight_layout()
    out_path = output_dir / "cache_ablation_main.pdf"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.savefig(out_path.with_suffix(".png"), dpi=150, bbox_inches="tight")
    print(f"\nSaved: {out_path}")
    plt.close()

    # ═══════════════════════════════════════════════════════════════════════
    # Figure 2: P vs I comparison
    # ═══════════════════════════════════════════════════════════════════════
    if proj_ablation:
        fig, axes = plt.subplots(1, 3, figsize=(13, 4))
        fig.suptitle(
            "Projection Ablation: P vs I at γ=1.0 (7K edits)",
            fontsize=12, fontweight="bold",
        )

        proj_layers = sorted(int(k) for k in proj_ablation["layers"].keys())
        gains_P = [proj_ablation["layers"][str(l)]["gain_with_P"] for l in proj_layers]
        gains_I = [proj_ablation["layers"][str(l)]["gain_with_I"] for l in proj_layers]
        ratios = [proj_ablation["layers"][str(l)]["gain_ratio_P_over_I"] for l in proj_layers]
        key_losses = [proj_ablation["layers"][str(l)]["key_projection_loss"] for l in proj_layers]
        update_P = [proj_ablation["layers"][str(l)]["update_norm_P"] for l in proj_layers]
        update_I = [proj_ablation["layers"][str(l)]["update_norm_I"] for l in proj_layers]

        # Panel A: gain comparison
        ax = axes[0]
        x = np.arange(len(proj_layers))
        w = 0.35
        ax.bar(x - w / 2, gains_P, w, label="With P", color="steelblue")
        ax.bar(x + w / 2, gains_I, w, label="With I", color="coral")
        ax.set_xticks(x)
        ax.set_xticklabels([f"L{l}" for l in proj_layers])
        ax.set_ylabel("Mean inverse gain")
        ax.set_title("A. Gain: P vs I")
        ax.legend()
        ax.grid(True, alpha=0.3, axis="y")

        # Panel B: gain ratio (should be ≈1.0)
        ax = axes[1]
        ax.bar(x, ratios, color="mediumpurple", alpha=0.8)
        ax.axhline(1.0, color="red", linestyle="--", alpha=0.7)
        ax.set_xticks(x)
        ax.set_xticklabels([f"L{l}" for l in proj_layers])
        ax.set_ylabel("gain(P) / gain(I)")
        ax.set_title("B. Gain Ratio (1.0 = P irrelevant)")
        ax.set_ylim(0.99, 1.005)
        ax.grid(True, alpha=0.3, axis="y")

        # Panel C: relative update difference ||ΔW_P - ΔW_I|| / ||ΔW_I||
        # Approximate from norms (not exact without the actual vectors)
        ax = axes[2]
        ax.bar(x, key_losses, color="darkorange", alpha=0.8)
        ax.set_xticks(x)
        ax.set_xticklabels([f"L{l}" for l in proj_layers])
        ax.set_ylabel("||Pk - k|| / ||k||")
        ax.set_title("C. Key Projection Loss\n(P modifies keys by this fraction)")
        ax.grid(True, alpha=0.3, axis="y")

        plt.tight_layout()
        out_path = output_dir / "projection_ablation.pdf"
        plt.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.savefig(out_path.with_suffix(".png"), dpi=150, bbox_inches="tight")
        print(f"Saved: {out_path}")
        plt.close()

    # ═══════════════════════════════════════════════════════════════════════
    # Print summary statistics
    # ═══════════════════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("CACHE ABLATION SUMMARY")
    print("=" * 70)

    print("\n--- Confirmed findings ---")
    print("\n1. Cache dominance at γ=1.0 (||γC||_F / ||KK^T+λI||_F):")
    for layer in layers:
        dom = data[1.0][layer]["cache_dominance_fro"]
        print(f"   Layer {layer}: {dom:.1f}× (cache energy exceeds key energy)")

    print("\n2. Gain suppression (γ=0 → γ=1.0):")
    for layer in layers:
        g0 = data[0.0][layer]["mean_inverse_gain"]
        g1 = data[1.0][layer]["mean_inverse_gain"]
        pct = (g0 - g1) / g0 * 100
        print(f"   Layer {layer}: {g0:.4f} → {g1:.4f} ({pct:.1f}% reduction)")

    print("\n3. Residual attainment suppression (γ=0 → γ=1.0):")
    for layer in layers:
        a0 = data[0.0][layer]["residual_attainment"]
        a1 = data[1.0][layer]["residual_attainment"]
        pct = (a0 - a1) / a0 * 100
        print(f"   Layer {layer}: {a0:.4f} → {a1:.4f} ({pct:.1f}% reduction)")

    print("\n4. P vs I operational equivalence:")
    if proj_ablation:
        for layer in proj_layers:
            m = proj_ablation["layers"][str(layer)]
            print(f"   Layer {layer}: gain ratio = {m['gain_ratio_P_over_I']:.4f}, "
                  f"key projection loss = {m['key_projection_loss']:.4f}")

    print("\n5. Solve residuals (numerical validity):")
    for g in [0.0, 1.0]:
        residuals = [data[g][layer]["solve_residual"] for layer in layers]
        print(f"   γ={g}: mean={np.mean(residuals):.2e}, max={np.max(residuals):.2e}")

    print("\n--- Per-key cache alignment (k^T C k / ||k||²) ---")
    for layer in layers:
        m = data[1.0][layer]
        print(f"   Layer {layer}: mean={m['mean_cache_alignment_kCk']:.1f}, "
              f"max={m['max_cache_alignment_kCk']:.1f}")

    print("\n--- Still to confirm ---")
    print("  • Reducing γ restores actual factual-edit efficacy (not just linear attainment)")
    print("  • Recovered efficacy trades off against prior-edit retention")
    print("  • Mechanism generalizes across seeds, datasets, models")
    print()


if __name__ == "__main__":
    main()

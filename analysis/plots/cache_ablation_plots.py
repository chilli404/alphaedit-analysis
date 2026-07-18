#!/usr/bin/env python3
"""
Cache Ablation Analysis: Visualize causal evidence for over-regularization.

Produces multi-panel figures showing how cache scaling (gamma) affects:
  - Inverse gain (solver response per key)
  - Residual attainment
  - Relative update size
  - Cache/key energy ratio
  - P vs I comparison
  - Behavioral tradeoff (new efficacy vs retention)

Automatically discovers all seeds and batch checkpoints.

Usage:
    uv run python -m analysis.plots.cache_ablation_plots
"""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

CACHE_ABLATION_DIR = Path("results/cache_ablation")
BEHAVIORAL_DIR = Path("results/cache_ablation_behavioral")
OUTPUT_DIR = Path("results/figures/cache_ablation")


def load_ablation_data(results_dir: Path) -> dict[str, tuple[list[dict], dict | None]]:
    """Load cache ablation JSONL for all seeds/batches.

    Returns dict keyed by "seed{N}_batch{M}" -> (gamma_records, projection_ablation).
    """
    all_data = {}

    for seed_dir in sorted(results_dir.iterdir()):
        if not seed_dir.is_dir() or not seed_dir.name.startswith("seed"):
            continue

        for f in sorted(seed_dir.glob("cache_ablation_*.jsonl")):
            gamma_records = []
            proj_ablation = None

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

            if gamma_records:
                # Extract batch from filename or record
                batch = gamma_records[0].get("checkpoint_batch", "unknown")
                key = f"{seed_dir.name}_batch{batch}"
                all_data[key] = (gamma_records, proj_ablation)

    return all_data


def load_behavioral_data(results_dir: Path) -> dict[str, list[dict]]:
    """Load behavioral ablation JSONL for all seeds/batches.

    Returns dict keyed by "seed{N}_batch{M}" -> list of gamma records.
    """
    all_data = {}

    for seed_dir in sorted(results_dir.iterdir()):
        if not seed_dir.is_dir() or not seed_dir.name.startswith("seed"):
            continue

        for f in sorted(seed_dir.glob("behavioral_*.jsonl")):
            records = []
            with open(f) as fh:
                for line in fh:
                    line = line.strip()
                    if not line:
                        continue
                    records.append(json.loads(line))

            if records:
                batch = records[0].get("checkpoint_batch", "unknown")
                key = f"{seed_dir.name}_batch{batch}"
                all_data[key] = records

    return all_data


def plot_gamma_sweep(all_data: dict, output_dir: Path) -> None:
    """Plot gamma sweep diagnostics, one figure per seed/batch combination."""

    for run_key, (gamma_records, proj_ablation) in all_data.items():
        gammas = sorted(set(r["gamma"] for r in gamma_records))
        layers = sorted(set(int(k) for r in gamma_records for k in r["layers"].keys()))

        # Build lookup
        data = {}
        for r in gamma_records:
            g = r["gamma"]
            data[g] = {}
            for layer_str, metrics in r["layers"].items():
                data[g][int(layer_str)] = metrics

        seed = gamma_records[0].get("seed", "?")
        batch = gamma_records[0].get("checkpoint_batch", "?")
        total_edits = gamma_records[0].get("total_prior_edits", (batch + 1) * 100 if isinstance(batch, int) else "?")

        # ─── Figure 1: 4-panel diagnostic ───
        fig, axes = plt.subplots(2, 2, figsize=(11, 8.5))
        fig.suptitle(
            f"Cache Ablation: Causal Evidence for Over-Regularization\n"
            f"(Seed {seed}, batch {batch}, {total_edits} edits)",
            fontsize=12, fontweight="bold",
        )

        colors = plt.cm.viridis(np.linspace(0.2, 0.9, len(layers)))

        # Panel A: gamma vs inverse gain
        ax = axes[0, 0]
        for i, layer in enumerate(layers):
            y = [data[g][layer]["mean_inverse_gain"] for g in gammas]
            ax.plot(gammas, y, "o-", color=colors[i], label=f"Layer {layer}", markersize=5)
        ax.set_xlabel("\u03b3 (cache scaling)")
        ax.set_ylabel("Mean inverse gain")
        ax.set_title("A. Inverse Gain vs Cache Strength")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.set_xlim(-0.05, 1.05)

        # Panel B: gamma vs residual attainment
        ax = axes[0, 1]
        for i, layer in enumerate(layers):
            y = [data[g][layer]["residual_attainment"] for g in gammas]
            ax.plot(gammas, y, "o-", color=colors[i], label=f"Layer {layer}", markersize=5)
        ax.set_xlabel("\u03b3 (cache scaling)")
        ax.set_ylabel("||\u0394WK||_F / ||R||_F")
        ax.set_title("B. Residual Attainment vs Cache Strength")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.set_xlim(-0.05, 1.05)

        # Panel C: cache dominance per layer
        ax = axes[1, 0]
        for i, g in enumerate(gammas[1:], 1):  # skip gamma=0
            dominance = [data[g][layer]["cache_dominance_fro"] for layer in layers]
            ax.bar(
                np.arange(len(layers)) + (i - 1) * 0.18,
                dominance, width=0.18, alpha=0.8, label=f"\u03b3={g}",
            )
        ax.set_xticks(np.arange(len(layers)) + 0.27)
        ax.set_xticklabels([f"Layer {l}" for l in layers])
        ax.set_ylabel("||\u03b3C||_F / ||KK^T + \u03bbI||_F")
        ax.set_title("C. Cache Energy Dominance by Layer")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3, axis="y")
        ax.axhline(1.0, color="red", linestyle="--", alpha=0.5, linewidth=0.8)

        # Panel D: relative update size
        ax = axes[1, 1]
        for i, layer in enumerate(layers):
            y = [data[g][layer]["relative_update_size"] for g in gammas]
            ax.plot(gammas, y, "o-", color=colors[i], label=f"Layer {layer}", markersize=5)
        ax.set_xlabel("\u03b3 (cache scaling)")
        ax.set_ylabel("||\u0394W||_F / ||W||_F")
        ax.set_title("D. Relative Update Size vs Cache Strength")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.set_xlim(-0.05, 1.05)

        plt.tight_layout()
        out_path = output_dir / f"cache_ablation_{run_key}.png"
        plt.savefig(out_path, dpi=150, bbox_inches="tight")
        print(f"Saved: {out_path}")
        plt.close()

        # ─── Figure 2: P vs I comparison ───
        if proj_ablation:
            fig, axes2 = plt.subplots(1, 3, figsize=(13, 4))
            fig.suptitle(
                f"Projection Ablation: P vs I (Seed {seed}, {total_edits} edits)",
                fontsize=12, fontweight="bold",
            )

            proj_layers = sorted(int(k) for k in proj_ablation["layers"].keys())
            gains_P = [proj_ablation["layers"][str(l)]["gain_with_P"] for l in proj_layers]
            gains_I = [proj_ablation["layers"][str(l)]["gain_with_I"] for l in proj_layers]
            ratios = [proj_ablation["layers"][str(l)]["gain_ratio_P_over_I"] for l in proj_layers]
            key_losses = [proj_ablation["layers"][str(l)]["key_projection_loss"] for l in proj_layers]

            x = np.arange(len(proj_layers))
            w = 0.35

            ax = axes2[0]
            ax.bar(x - w / 2, gains_P, w, label="With P", color="steelblue")
            ax.bar(x + w / 2, gains_I, w, label="With I", color="coral")
            ax.set_xticks(x)
            ax.set_xticklabels([f"L{l}" for l in proj_layers])
            ax.set_ylabel("Mean inverse gain")
            ax.set_title("A. Gain: P vs I")
            ax.legend()
            ax.grid(True, alpha=0.3, axis="y")

            ax = axes2[1]
            ax.bar(x, ratios, color="mediumpurple", alpha=0.8)
            ax.axhline(1.0, color="red", linestyle="--", alpha=0.7)
            ax.set_xticks(x)
            ax.set_xticklabels([f"L{l}" for l in proj_layers])
            ax.set_ylabel("gain(P) / gain(I)")
            ax.set_title("B. Gain Ratio (1.0 = P irrelevant)")
            ax.set_ylim(0.99, 1.005)
            ax.grid(True, alpha=0.3, axis="y")

            ax = axes2[2]
            ax.bar(x, key_losses, color="darkorange", alpha=0.8)
            ax.set_xticks(x)
            ax.set_xticklabels([f"L{l}" for l in proj_layers])
            ax.set_ylabel("||Pk - k|| / ||k||")
            ax.set_title("C. Key Projection Loss")
            ax.grid(True, alpha=0.3, axis="y")

            plt.tight_layout()
            out_path = output_dir / f"projection_ablation_{run_key}.png"
            plt.savefig(out_path, dpi=150, bbox_inches="tight")
            print(f"Saved: {out_path}")
            plt.close()

    # ─── Print summary across all runs ───
    print("\n" + "=" * 70)
    print("CACHE ABLATION SUMMARY (all seeds/batches)")
    print("=" * 70)

    for run_key, (gamma_records, proj_ablation) in all_data.items():
        gammas = sorted(set(r["gamma"] for r in gamma_records))
        layers = sorted(set(int(k) for r in gamma_records for k in r["layers"].keys()))
        data = {}
        for r in gamma_records:
            g = r["gamma"]
            data[g] = {int(k): v for k, v in r["layers"].items()}

        print(f"\n--- {run_key} ---")
        print("  Gain suppression (gamma=0 -> gamma=1.0):")
        for layer in layers:
            g0 = data[0.0][layer]["mean_inverse_gain"]
            g1 = data[1.0][layer]["mean_inverse_gain"]
            pct = (g0 - g1) / g0 * 100
            print(f"    Layer {layer}: {g0:.4f} -> {g1:.4f} ({pct:.1f}% reduction)")

        print("  Cache dominance at gamma=1.0:")
        for layer in layers:
            dom = data[1.0][layer]["cache_dominance_fro"]
            print(f"    Layer {layer}: {dom:.1f}x")

        if proj_ablation:
            print("  P vs I gain ratio:")
            for layer_str, m in proj_ablation["layers"].items():
                print(f"    Layer {layer_str}: {m['gain_ratio_P_over_I']:.4f}")


def plot_behavioral(all_behavioral: dict, output_dir: Path) -> None:
    """Plot behavioral tradeoff: new efficacy vs retention across gamma values."""

    if not all_behavioral:
        print("No behavioral data found.")
        return

    # Composite figure: all seeds/batches
    n_runs = len(all_behavioral)
    fig, axes = plt.subplots(1, n_runs, figsize=(5 * n_runs, 4.5), squeeze=False)
    fig.suptitle(
        "Cache Ablation Behavioral: Stability-Plasticity Tradeoff",
        fontsize=12, fontweight="bold",
    )

    for idx, (run_key, records) in enumerate(sorted(all_behavioral.items())):
        ax = axes[0, idx]
        gammas = [r["gamma"] for r in records]
        new_eff = [r.get("new_efficacy", np.nan) for r in records]
        retention = [r.get("retention_efficacy", np.nan) for r in records]

        ax.plot(gammas, new_eff, "o-", color="#2196F3", linewidth=2, markersize=6, label="New-edit efficacy")
        ax.plot(gammas, retention, "s-", color="#4caf50", linewidth=2, markersize=6, label="Prior-edit retention")

        ax.set_xlabel("\u03b3 (cache scaling)")
        ax.set_ylabel("Score")
        ax.set_title(run_key.replace("_", " "))
        ax.legend(fontsize=8)
        ax.set_ylim(0, 1.05)
        ax.set_xlim(-0.05, 1.05)
        ax.grid(True, alpha=0.3)
        ax.axhline(0.5, color="gray", linestyle=":", alpha=0.4)

    plt.tight_layout()
    out_path = output_dir / "behavioral_tradeoff.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Saved: {out_path}")
    plt.close()

    # Print summary
    print("\n" + "=" * 70)
    print("BEHAVIORAL TRADEOFF SUMMARY")
    print("=" * 70)
    for run_key, records in sorted(all_behavioral.items()):
        print(f"\n--- {run_key} ---")
        print(f"  {'gamma':>6} | {'new_eff':>8} | {'retention':>9} | {'new_para':>8} | {'ret_para':>8}")
        print(f"  {'-'*6}-+-{'-'*8}-+-{'-'*9}-+-{'-'*8}-+-{'-'*8}")
        for r in records:
            print(f"  {r['gamma']:>6.2f} | "
                  f"{r.get('new_efficacy', 0):>8.3f} | "
                  f"{r.get('retention_efficacy', 0):>9.3f} | "
                  f"{r.get('new_paraphrase', 0):>8.3f} | "
                  f"{r.get('retention_paraphrase', 0):>8.3f}")


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Load linear-algebra diagnostics
    print("Loading cache ablation data...")
    all_data = load_ablation_data(CACHE_ABLATION_DIR)
    if all_data:
        print(f"Found runs: {list(all_data.keys())}")
        plot_gamma_sweep(all_data, OUTPUT_DIR)
    else:
        print(f"No cache ablation data in {CACHE_ABLATION_DIR}")

    # Load behavioral data
    print("\nLoading behavioral data...")
    all_behavioral = load_behavioral_data(BEHAVIORAL_DIR)
    if all_behavioral:
        print(f"Found runs: {list(all_behavioral.keys())}")
        plot_behavioral(all_behavioral, OUTPUT_DIR)
    else:
        print(f"No behavioral data in {BEHAVIORAL_DIR}")

    print("\n=== Done ===")


if __name__ == "__main__":
    main()

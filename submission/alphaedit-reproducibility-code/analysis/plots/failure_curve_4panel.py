#!/usr/bin/env python3
"""
Four-panel failure curve figure for AlphaEdit reproducibility study.

Generates the central visual argument for the paper:
    stable geometry and retention → crowding transition → AlphaEdit degradation

Panel A — Cumulative efficacy (AlphaEdit vs MEMIT over edit count)
Panel B — Retention by edit age (heatmap: cohort × checkpoint)
Panel C — Efficacy + probability-specificity together (locality vs retention crossover)
Panel D — AlphaEdit advantage gap (converging to zero)

Usage:
    uv run python -m analysis.failure_curve_4panel
    uv run python -m analysis.failure_curve_4panel --results_dir results/failure_curve_checkpointed
"""

import argparse
import json
import random
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import pandas as pd
from matplotlib.gridspec import GridSpec

# Paper-quality matplotlib settings
plt.rcParams.update({
    "font.size": 9,
    "axes.labelsize": 10,
    "axes.titlesize": 11,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 8,
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "font.family": "serif",
    "axes.spines.top": False,
    "axes.spines.right": False,
})

COLORS = {"AlphaEdit": "#2196F3", "MEMIT": "#FF9800"}
SEEDS = [42, 2024]
BATCH_SIZE = 100
SAMPLE_SIZE = 2000  # Per-checkpoint sample for speed


def find_run_dirs(base_dir: Path) -> list[dict]:
    """Scan checkpoint results and find all (seed, algorithm, edit_count, run_dir) tuples.

    Supports both flat (AlphaEdit/run_000/) and nested (wrapper/AlphaEdit/run_000/) layouts.
    """
    entries = []
    for seed_dir in sorted(base_dir.iterdir()):
        if not seed_dir.is_dir() or not seed_dir.name.startswith("seed"):
            continue
        seed = int(seed_dir.name.replace("seed", ""))

        for edits_dir in sorted(seed_dir.iterdir()):
            if not edits_dir.is_dir() or not edits_dir.name.endswith("edits"):
                continue
            total_edits = int(edits_dir.name.replace("edits", ""))

            # Collect algorithm dirs from both flat and nested layouts
            alg_dirs = []
            for subdir in edits_dir.iterdir():
                if not subdir.is_dir():
                    continue
                if subdir.name in ("AlphaEdit", "MEMIT"):
                    alg_dirs.append(subdir)
                else:
                    for alg_dir in subdir.iterdir():
                        if alg_dir.is_dir() and alg_dir.name in ("AlphaEdit", "MEMIT"):
                            alg_dirs.append(alg_dir)

            for alg_dir in alg_dirs:
                for run_dir in sorted(alg_dir.iterdir()):
                    if not run_dir.is_dir() or not run_dir.name.startswith("run_"):
                        continue
                    case_files = list(run_dir.glob("*_edits-case_*.json"))
                    if len(case_files) < 100:
                        continue
                    entries.append({
                        "seed": seed,
                        "algorithm": alg_dir.name,
                        "total_edits": total_edits,
                        "run_dir": run_dir,
                        "n_cases": len(case_files),
                    })
    return entries


def compute_metrics_for_entry(entry: dict) -> dict:
    """
    Load case files for one (seed, alg, edits) entry.
    Returns dict with efficacy, generalization, specificity (argmax and prob).
    """
    run_dir = entry["run_dir"]
    case_files = sorted(run_dir.glob("*_edits-case_*.json"))

    # Sample for speed
    rng = random.Random(42)
    sample = rng.sample(case_files, min(SAMPLE_SIZE, len(case_files)))

    eff_vals = []
    gen_vals = []
    spec_argmax_vals = []
    spec_prob_vals = []

    for cf in sample:
        with open(cf) as f:
            d = json.load(f)
        post = d["post"]

        rc = post.get("rewrite_prompts_correct", [])
        if rc:
            eff_vals.append(sum(rc) / len(rc))

        pc = post.get("paraphrase_prompts_correct", [])
        if pc:
            gen_vals.append(sum(pc) / len(pc))

        nc = post.get("neighborhood_prompts_correct", [])
        if nc:
            spec_argmax_vals.append(sum(nc) / len(nc))

        np_probs = post.get("neighborhood_prompts_probs", [])
        if np_probs:
            prob_correct = sum(1 for p in np_probs if p["target_true"] < p["target_new"])
            spec_prob_vals.append(prob_correct / len(np_probs))

    return {
        "seed": entry["seed"],
        "algorithm": entry["algorithm"],
        "total_edits": entry["total_edits"],
        "efficacy": np.mean(eff_vals) if eff_vals else np.nan,
        "generalization": np.mean(gen_vals) if gen_vals else np.nan,
        "spec_argmax": np.mean(spec_argmax_vals) if spec_argmax_vals else np.nan,
        "spec_prob": np.mean(spec_prob_vals) if spec_prob_vals else np.nan,
        "n_sampled": len(sample),
    }


def compute_cohort_heatmap(entries: list[dict], algorithm: str = "AlphaEdit") -> np.ndarray:
    """
    Build a cohort × checkpoint heatmap matrix for AlphaEdit.

    Rows = cohort bands (groups of 10 batches = 1000 edits)
    Columns = checkpoint edit counts
    Values = mean efficacy for that cohort at that checkpoint
    """
    alg_entries = [e for e in entries if e["algorithm"] == algorithm]
    checkpoints = sorted(set(e["total_edits"] for e in alg_entries))

    # Define cohort bands (1000-edit bands)
    max_edits = max(checkpoints)
    n_bands = max_edits // 1000
    band_labels = [f"{i}K–{i+1}K" for i in range(n_bands)]

    # Initialize heatmap: rows=bands, cols=checkpoints
    heatmap = np.full((n_bands, len(checkpoints)), np.nan)

    for col_idx, edits in enumerate(checkpoints):
        # Load case data for this checkpoint (sample across seeds)
        ckpt_entries = [e for e in alg_entries if e["total_edits"] == edits]

        cohort_efficacies = {}  # band_idx -> list of efficacy values

        for entry in ckpt_entries:
            run_dir = entry["run_dir"]
            case_files = sorted(run_dir.glob("*_edits-case_*.json"))

            # Sample for speed
            rng = random.Random(42)
            sample = rng.sample(case_files, min(SAMPLE_SIZE, len(case_files)))

            for cf in sample:
                with open(cf) as f:
                    d = json.load(f)
                case_id = d["case_id"]
                band_idx = (case_id // BATCH_SIZE) // 10  # Group into 1000-edit bands

                if band_idx >= n_bands:
                    continue
                # Only include bands that were edited BEFORE this checkpoint
                if (band_idx + 1) * 1000 > edits:
                    continue

                rc = d["post"].get("rewrite_prompts_correct", [])
                if rc:
                    eff = sum(rc) / len(rc)
                    if band_idx not in cohort_efficacies:
                        cohort_efficacies[band_idx] = []
                    cohort_efficacies[band_idx].append(eff)

        for band_idx, vals in cohort_efficacies.items():
            heatmap[band_idx, col_idx] = np.mean(vals)

    return heatmap, band_labels, checkpoints


def plot_4panel(
    metrics_df: pd.DataFrame,
    heatmap: np.ndarray,
    band_labels: list[str],
    checkpoints: list[int],
    output_dir: Path,
) -> None:
    """Generate the 4-panel figure."""

    fig = plt.figure(figsize=(12, 9))
    gs = GridSpec(2, 2, figure=fig, hspace=0.35, wspace=0.30)

    # ─── Panel A: Cumulative Efficacy ───────────────────────────────────────
    ax_a = fig.add_subplot(gs[0, 0])

    for alg in ["AlphaEdit", "MEMIT"]:
        alg_data = metrics_df[metrics_df["algorithm"] == alg]
        if alg_data.empty:
            continue
        color = COLORS[alg]

        # Per-seed thin lines
        for seed in alg_data["seed"].unique():
            seed_data = alg_data[alg_data["seed"] == seed].sort_values("total_edits")
            ax_a.plot(seed_data["total_edits"], seed_data["efficacy"],
                      "-", color=color, alpha=0.25, linewidth=1)

        # Mean thick line
        mean_data = (
            alg_data.groupby("total_edits")["efficacy"]
            .mean().reset_index().sort_values("total_edits")
        )
        ax_a.plot(mean_data["total_edits"], mean_data["efficacy"],
                  "-o", color=color, linewidth=2.2, markersize=5, label=alg)

    ax_a.set_xlabel("Total Sequential Edits")
    ax_a.set_ylabel("Efficacy (argmax)")
    ax_a.set_title("A. Cumulative Edit Efficacy")
    ax_a.legend(loc="upper right")
    ax_a.set_ylim(-0.02, 1.02)
    ax_a.axhline(y=0.5, color="gray", linestyle=":", alpha=0.3, linewidth=0.8)
    ax_a.set_xticks(sorted(metrics_df["total_edits"].unique()))
    ax_a.tick_params(axis="x", rotation=45)

    # Annotate transition zone
    ax_a.axvspan(5000, 7000, alpha=0.06, color="red", label="_nolegend_")
    ax_a.text(5800, 0.95, "transition\nzone", ha="center", fontsize=7,
              color="red", alpha=0.7, style="italic")

    # ─── Panel B: Cohort Retention Heatmap ──────────────────────────────────
    ax_b = fig.add_subplot(gs[0, 1])

    # Mask NaN values
    masked_heatmap = np.ma.masked_invalid(heatmap)

    cmap = plt.cm.RdYlGn
    cmap.set_bad(color="white")

    im = ax_b.imshow(
        masked_heatmap, aspect="auto", cmap=cmap, vmin=0, vmax=1,
        origin="lower", interpolation="nearest",
    )

    # Axis labels
    ax_b.set_xticks(range(len(checkpoints)))
    ax_b.set_xticklabels([f"{c//1000}K" for c in checkpoints], fontsize=7)
    ax_b.set_xlabel("Evaluation Checkpoint (total edits)")

    # Show subset of y-labels to avoid crowding
    n_bands = len(band_labels)
    y_tick_indices = list(range(0, n_bands, 2)) if n_bands > 6 else list(range(n_bands))
    ax_b.set_yticks(y_tick_indices)
    ax_b.set_yticklabels([band_labels[i] for i in y_tick_indices], fontsize=7)
    ax_b.set_ylabel("Edit Cohort (when edited)")
    ax_b.set_title("B. Cohort Retention (AlphaEdit)")

    # Colorbar
    cbar = fig.colorbar(im, ax=ax_b, fraction=0.046, pad=0.04)
    cbar.set_label("Efficacy", fontsize=8)
    cbar.ax.tick_params(labelsize=7)

    # ─── Panel C: Efficacy vs Probability-Specificity ───────────────────────
    ax_c = fig.add_subplot(gs[1, 0])

    # AlphaEdit only — show both curves
    ae_data = metrics_df[metrics_df["algorithm"] == "AlphaEdit"]
    ae_mean = ae_data.groupby("total_edits")[["efficacy", "spec_prob"]].mean().reset_index()
    ae_mean = ae_mean.sort_values("total_edits")

    ax_c.plot(ae_mean["total_edits"], ae_mean["efficacy"],
              "-o", color=COLORS["AlphaEdit"], linewidth=2.2, markersize=5,
              label="Efficacy (target retention)")
    ax_c.plot(ae_mean["total_edits"], ae_mean["spec_prob"],
              "-s", color="#4caf50", linewidth=2.2, markersize=5,
              label="Prob-specificity (locality)")

    # Show per-seed for both
    for seed in ae_data["seed"].unique():
        sd = ae_data[ae_data["seed"] == seed].sort_values("total_edits")
        ax_c.plot(sd["total_edits"], sd["efficacy"],
                  "-", color=COLORS["AlphaEdit"], alpha=0.2, linewidth=1)
        ax_c.plot(sd["total_edits"], sd["spec_prob"],
                  "-", color="#4caf50", alpha=0.2, linewidth=1)

    # Mark crossover
    # Find approximate crossover point by interpolation
    eff_vals = ae_mean["efficacy"].values
    spec_vals = ae_mean["spec_prob"].values
    edits_vals = ae_mean["total_edits"].values
    for i in range(len(eff_vals) - 1):
        if eff_vals[i] >= spec_vals[i] and eff_vals[i + 1] < spec_vals[i + 1]:
            # Linear interpolation for crossover
            t = (eff_vals[i] - spec_vals[i]) / (
                (eff_vals[i] - spec_vals[i]) - (eff_vals[i + 1] - spec_vals[i + 1])
            )
            cross_x = edits_vals[i] + t * (edits_vals[i + 1] - edits_vals[i])
            cross_y = eff_vals[i] + t * (eff_vals[i + 1] - eff_vals[i])
            ax_c.axvline(x=cross_x, color="gray", linestyle="--", alpha=0.5, linewidth=0.8)
            ax_c.plot(cross_x, cross_y, "x", color="black", markersize=10, markeredgewidth=2)
            ax_c.annotate(
                f"crossover\n≈{int(cross_x)} edits",
                xy=(cross_x, cross_y), xytext=(cross_x + 400, cross_y + 0.08),
                fontsize=7, ha="left", color="black", style="italic",
                arrowprops=dict(arrowstyle="->", color="black", lw=0.8),
            )
            break

    ax_c.axhline(y=0.5, color="gray", linestyle=":", alpha=0.4, linewidth=0.8)
    ax_c.text(10200, 0.505, "chance", fontsize=7, color="gray", va="bottom")

    ax_c.set_xlabel("Total Sequential Edits")
    ax_c.set_ylabel("Score")
    ax_c.set_title("C. Locality vs Retention (AlphaEdit)")
    ax_c.legend(loc="upper right", fontsize=8)
    ax_c.set_ylim(-0.02, 1.02)
    ax_c.set_xticks(sorted(ae_data["total_edits"].unique()))
    ax_c.tick_params(axis="x", rotation=45)

    # ─── Panel D: AlphaEdit–MEMIT Gap (convergence) ─────────────────────────
    ax_d = fig.add_subplot(gs[1, 1])

    # Compute gap at each checkpoint where both algorithms exist
    ae_means = (
        metrics_df[metrics_df["algorithm"] == "AlphaEdit"]
        .groupby("total_edits")[["efficacy", "spec_prob"]].mean()
    )
    memit_means = (
        metrics_df[metrics_df["algorithm"] == "MEMIT"]
        .groupby("total_edits")[["efficacy", "spec_prob"]].mean()
    )
    common = sorted(set(ae_means.index) & set(memit_means.index))

    if common:
        eff_gap = [ae_means.loc[e, "efficacy"] - memit_means.loc[e, "efficacy"] for e in common]
        spec_gap = [ae_means.loc[e, "spec_prob"] - memit_means.loc[e, "spec_prob"] for e in common]

        ax_d.plot(common, eff_gap, "-o", color=COLORS["AlphaEdit"], linewidth=2.2,
                  markersize=5, label="Δ Efficacy")
        ax_d.plot(common, spec_gap, "-s", color="#4caf50", linewidth=2.2,
                  markersize=5, label="Δ Prob-specificity")

        ax_d.axhline(y=0, color="black", linestyle="-", alpha=0.5, linewidth=0.8)

        # Shade: above zero = AlphaEdit better
        ax_d.fill_between(common, 0, eff_gap, alpha=0.08, color=COLORS["AlphaEdit"])
        ax_d.fill_between(common, 0, spec_gap, alpha=0.08, color="#4caf50")

        # Annotate convergence
        if len(spec_gap) > 1 and spec_gap[-1] < 0.03:
            ax_d.annotate(
                "converged",
                xy=(common[-1], spec_gap[-1]),
                xytext=(common[-1] - 800, spec_gap[-1] + 0.08),
                fontsize=7, color="#4caf50", style="italic",
                arrowprops=dict(arrowstyle="->", color="#4caf50", lw=0.8),
            )

    ax_d.set_xlabel("Total Sequential Edits")
    ax_d.set_ylabel("AlphaEdit − MEMIT (Δ)")
    ax_d.set_title("D. Advantage Gap (→ 0 = no advantage)")
    ax_d.legend(loc="upper right", fontsize=8)
    ax_d.set_xticks(common)
    ax_d.tick_params(axis="x", rotation=45)

    # ─── Suptitle ───────────────────────────────────────────────────────────
    fig.suptitle(
        "AlphaEdit Failure Characterization: 10,000 Sequential Edits\n"
        "(Llama-3-8B-Instruct, MultiCounterFact, 100-edit batches, seeds 42 & 2024)",
        fontsize=12, y=0.98,
    )

    # ─── Save ───────────────────────────────────────────────────────────────
    output_dir.mkdir(parents=True, exist_ok=True)

    for fmt in ["pdf", "png"]:
        output_path = output_dir / f"failure_curve_4panel.{fmt}"
        fig.savefig(output_path)
        print(f"Saved: {output_path}")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="4-panel failure curve figure")
    parser.add_argument(
        "--results_dir", type=Path,
        default=Path("results/failure_curve_checkpointed"),
    )
    parser.add_argument(
        "--output_dir", type=Path,
        default=Path("results/figures/failure_curve"),
    )
    args = parser.parse_args()

    print("=== 4-Panel Failure Curve Figure ===\n")

    # 1. Discover data
    print("Scanning results...")
    entries = find_run_dirs(args.results_dir)
    print(f"Found {len(entries)} data points\n")

    # 2. Compute metrics (efficacy, generalization, specificity×2) for each entry
    print("Computing metrics (sampling up to 2000 cases per checkpoint)...")
    rows = []
    for entry in entries:
        print(f"  {entry['seed']:>4} {entry['algorithm']:<10} {entry['total_edits']:>5} "
              f"({entry['n_cases']} cases)")
        rows.append(compute_metrics_for_entry(entry))

    metrics_df = pd.DataFrame(rows)
    print(f"\nMetrics table: {len(metrics_df)} rows")
    print(metrics_df[["seed", "algorithm", "total_edits", "efficacy", "spec_prob"]]
          .sort_values(["algorithm", "total_edits", "seed"]).to_string(index=False))
    print()

    # 3. Compute cohort retention heatmap
    print("Computing cohort retention heatmap...")
    heatmap, band_labels, checkpoints = compute_cohort_heatmap(entries, algorithm="AlphaEdit")
    print(f"  Heatmap shape: {heatmap.shape} (bands × checkpoints)")
    print()

    # 4. Generate figure
    print("Generating 4-panel figure...")
    plot_4panel(metrics_df, heatmap, band_labels, checkpoints, args.output_dir)

    # 5. Save metrics CSV
    csv_path = args.output_dir / "failure_curve_4panel_data.csv"
    metrics_df.to_csv(csv_path, index=False)
    print(f"Saved: {csv_path}")

    print("\n=== Done ===")


if __name__ == "__main__":
    main()

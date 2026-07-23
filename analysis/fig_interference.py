#!/usr/bin/env python3
"""
Update-Level Interference: Statistical Analysis + 3-Panel Figure

Panel A: Fine-grained first-1K cumulative path interference by ordering
         (clustered vs dispersed AlphaEdit). Median + IQR band.
Panel B: Retained vs forgotten U_path distributions within each ordering.
         Initially-successful only. Violin + points.
Panel C: Same-case matched 1K-exposure difference: dU_i = dispersed - clustered.
         Distribution with CI.

Statistics:
  - Cliff's delta with 95% CI (block bootstrap, 100-edit blocks)
  - Within-ordering logistic model
  - Same-case paired sign test + bootstrap CI

Usage:
    python analysis/fig_interference.py
    python analysis/fig_interference.py --phase1_only  # Coarse data only
    python analysis/fig_interference.py --seed 42
"""

import argparse
import json
import sys
from itertools import product
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src" / "util"))
from paths import get_result_root


# ─── Statistical Utilities ──────────────────────────────────────────────────

def cliffs_delta(x: np.ndarray, y: np.ndarray) -> float:
    """Cliff's delta: effect size for ordinal data. Range [-1, 1]."""
    n_greater = sum(1 for xi, yi in product(x, y) if xi > yi)
    n_less = sum(1 for xi, yi in product(x, y) if xi < yi)
    n = len(x) * len(y)
    return (n_greater - n_less) / n if n > 0 else 0.0


def cliffs_delta_fast(x: np.ndarray, y: np.ndarray) -> float:
    """Vectorized Cliff's delta (faster for large arrays)."""
    # Use broadcasting: x[:, None] vs y[None, :]
    if len(x) * len(y) > 1e8:
        # Too large for full matrix, use sampling
        rng = np.random.default_rng(42)
        n_samples = 100000
        xi = rng.choice(x, n_samples)
        yi = rng.choice(y, n_samples)
        return float(np.mean(np.sign(xi - yi)))
    diff = x[:, None] - y[None, :]
    return float(np.mean(np.sign(diff)))


def block_bootstrap_cliffs_delta(
    x: np.ndarray,
    y: np.ndarray,
    block_size: int = 100,
    n_bootstrap: int = 10000,
    seed: int = 42,
) -> Tuple[float, float, float]:
    """
    Block bootstrap CI for Cliff's delta.
    Returns (point_estimate, ci_lower, ci_upper).
    """
    rng = np.random.default_rng(seed)
    point = cliffs_delta_fast(x, y)

    # Block resample
    n_blocks_x = max(1, len(x) // block_size)
    n_blocks_y = max(1, len(y) // block_size)

    # Pad to full blocks
    x_padded = x[:n_blocks_x * block_size]
    y_padded = y[:n_blocks_y * block_size]
    x_blocks = x_padded.reshape(n_blocks_x, block_size)
    y_blocks = y_padded.reshape(n_blocks_y, block_size)

    boot_deltas = []
    for _ in range(n_bootstrap):
        x_sample = x_blocks[rng.integers(0, n_blocks_x, n_blocks_x)].ravel()
        y_sample = y_blocks[rng.integers(0, n_blocks_y, n_blocks_y)].ravel()
        boot_deltas.append(cliffs_delta_fast(x_sample, y_sample))

    boot_deltas = np.array(boot_deltas)
    ci_lower = float(np.percentile(boot_deltas, 2.5))
    ci_upper = float(np.percentile(boot_deltas, 97.5))

    return point, ci_lower, ci_upper


def paired_bootstrap_ci(
    differences: np.ndarray,
    n_bootstrap: int = 10000,
    seed: int = 42,
) -> Tuple[float, float, float]:
    """Bootstrap CI for mean of paired differences. Returns (mean, ci_lo, ci_hi)."""
    rng = np.random.default_rng(seed)
    point = float(np.mean(differences))
    boot_means = []
    for _ in range(n_bootstrap):
        sample = rng.choice(differences, len(differences), replace=True)
        boot_means.append(np.mean(sample))
    boot_means = np.array(boot_means)
    return point, float(np.percentile(boot_means, 2.5)), float(np.percentile(boot_means, 97.5))


# ─── Data Loading ───────────────────────────────────────────────────────────

def load_phase1(seed: int, alg: str, ordering: str) -> Optional[Dict]:
    path = get_result_root() / "interference" / alg / ordering / f"seed{seed}" / "phase1_coarse.json"
    if not path.exists():
        print(f"Phase 1 data not found: {path}")
        return None
    with open(path) as f:
        return json.load(f)


def load_phase2(seed: int, alg: str, ordering: str) -> Optional[Dict]:
    path = get_result_root() / "interference" / alg / ordering / f"seed{seed}" / "fine_grained.json"
    if not path.exists():
        print(f"Phase 2 data not found: {path}")
        return None
    with open(path) as f:
        return json.load(f)


def load_percase(seed: int, alg: str, ordering: str) -> Optional[Dict]:
    path = get_result_root() / "interference" / alg / ordering / f"seed{seed}" / "percase_eval.json"
    if not path.exists():
        print(f"Per-case eval not found: {path}")
        return None
    with open(path) as f:
        return json.load(f)


# ─── Analysis ──────────────────────────────────────────────────────────────

def within_ordering_analysis(
    U_path: np.ndarray,
    case_ids: List[int],
    percase_eval: List[Dict],
    ordering_name: str,
    block_sizes: List[int] = [50, 100, 200],
) -> Dict:
    """
    Within-ordering retained vs forgotten analysis.
    Primary: first-1K cohort classified at 5K.
    """
    print(f"\n  === Within-Ordering: {ordering_name} ===")

    # Match per-case eval
    eval_by_cid = {r["case_id"]: r for r in percase_eval}

    retained_idx = []
    forgotten_idx = []
    for i, cid in enumerate(case_ids):
        if cid in eval_by_cid:
            eff = eval_by_cid[cid]["efficacy"]
            if eff >= 0.5:
                retained_idx.append(i)
            else:
                forgotten_idx.append(i)

    n_retained = len(retained_idx)
    n_forgotten = len(forgotten_idx)
    print(f"    Retained: {n_retained}, Forgotten: {n_forgotten}")

    if n_forgotten < 5 or n_retained < 5:
        print(f"    SKIP: too few in one group")
        return {"skip": True, "n_retained": n_retained, "n_forgotten": n_forgotten}

    U_retained = U_path[retained_idx]
    U_forgotten = U_path[forgotten_idx]

    # Descriptive stats
    stats = {
        "n_retained": n_retained,
        "n_forgotten": n_forgotten,
        "retained_median": float(np.median(U_retained)),
        "retained_mean": float(np.mean(U_retained)),
        "retained_iqr": [float(np.percentile(U_retained, 25)),
                         float(np.percentile(U_retained, 75))],
        "forgotten_median": float(np.median(U_forgotten)),
        "forgotten_mean": float(np.mean(U_forgotten)),
        "forgotten_iqr": [float(np.percentile(U_forgotten, 25)),
                          float(np.percentile(U_forgotten, 75))],
        "median_diff": float(np.median(U_forgotten) - np.median(U_retained)),
        "mean_diff": float(np.mean(U_forgotten) - np.mean(U_retained)),
    }

    print(f"    Retained: median={stats['retained_median']:.6f}, mean={stats['retained_mean']:.6f}")
    print(f"    Forgotten: median={stats['forgotten_median']:.6f}, mean={stats['forgotten_mean']:.6f}")
    print(f"    Median diff (forg-ret): {stats['median_diff']:.6f}")

    # Cliff's delta with bootstrap
    print(f"    Computing Cliff's delta + bootstrap CI...")
    for bs in block_sizes:
        if n_forgotten >= bs and n_retained >= bs:
            d, ci_lo, ci_hi = block_bootstrap_cliffs_delta(
                U_forgotten, U_retained, block_size=bs
            )
            stats[f"cliffs_delta_block{bs}"] = {
                "point": round(d, 4),
                "ci_95": [round(ci_lo, 4), round(ci_hi, 4)],
            }
            print(f"      block={bs}: d={d:.4f} [{ci_lo:.4f}, {ci_hi:.4f}]")
        else:
            # Fall back to element-wise bootstrap
            d = cliffs_delta_fast(U_forgotten, U_retained)
            stats[f"cliffs_delta_block{bs}"] = {"point": round(d, 4), "ci_95": None}
            print(f"      block={bs}: d={d:.4f} (groups too small for block bootstrap)")

    # Logistic model (if statsmodels available)
    try:
        import statsmodels.api as sm
        # P(retained) ~ position + U_path
        positions = np.array(list(range(len(case_ids))), dtype=np.float64)
        X = np.column_stack([
            positions / positions.max(),  # normalized position
            U_path / (U_path.max() + 1e-10),  # normalized U_path
        ])
        y_binary = np.array([1 if i in set(retained_idx) else 0
                            for i in range(len(case_ids))
                            if i in set(retained_idx) | set(forgotten_idx)])
        X_filtered = X[sorted(retained_idx + forgotten_idx)]
        X_const = sm.add_constant(X_filtered)

        model = sm.Logit(y_binary, X_const).fit(disp=0)
        stats["logistic"] = {
            "coef_position": round(float(model.params[1]), 4),
            "coef_U_path": round(float(model.params[2]), 4),
            "pvalue_U_path": round(float(model.pvalues[2]), 6),
            "pseudo_r2": round(float(model.prsquared), 4),
        }
        print(f"    Logistic: coef_U_path={model.params[2]:.4f}, "
              f"p={model.pvalues[2]:.4e}, pseudo-R2={model.prsquared:.4f}")
    except ImportError:
        print(f"    Logistic: statsmodels not available, skipping")
    except Exception as e:
        print(f"    Logistic: failed ({e})")

    return stats


def matched_exposure_analysis(
    phase2_clustered: Dict,
    phase2_dispersed: Dict,
    seed: int,
    horizon_batches: int = 10,
) -> Dict:
    """
    Secondary: Same-case paired comparison at matched exposure horizons.
    """
    print(f"\n  === Matched-Exposure Paired Analysis (horizon={horizon_batches} batches = {horizon_batches*100} edits) ===")

    # Get installation batches for each ordering
    clust_case_ids = phase2_clustered["all_5K"]["case_ids"]
    disp_case_ids = phase2_dispersed["all_5K"]["case_ids"]
    clust_install = np.array(phase2_clustered["all_5K"]["installation_batch"])
    disp_install = np.array(phase2_dispersed["all_5K"]["installation_batch"])
    clust_U_path = np.array(phase2_clustered["all_5K"]["U_path"])
    disp_U_path = np.array(phase2_dispersed["all_5K"]["U_path"])

    # Build case_id -> index maps
    clust_cid_to_idx = {cid: i for i, cid in enumerate(clust_case_ids)}
    disp_cid_to_idx = {cid: i for i, cid in enumerate(disp_case_ids)}

    # Eligibility: installed by batch (49 - horizon_batches) in BOTH orderings
    max_install_batch = 49 - horizon_batches
    all_cids = set(clust_case_ids) & set(disp_case_ids)

    eligible = []
    for cid in all_cids:
        ci = clust_cid_to_idx[cid]
        di = disp_cid_to_idx[cid]
        if clust_install[ci] <= max_install_batch and disp_install[di] <= max_install_batch:
            eligible.append(cid)

    n_eligible = len(eligible)
    print(f"    Eligible pairs: {n_eligible}")

    if n_eligible < 30:
        print(f"    SKIP: too few eligible pairs")
        return {"n_eligible": n_eligible, "skip": True}

    # Compute paired differences: dU_i = U_dispersed - U_clustered
    differences = []
    for cid in eligible:
        ci = clust_cid_to_idx[cid]
        di = disp_cid_to_idx[cid]
        diff = disp_U_path[di] - clust_U_path[ci]
        differences.append(diff)

    differences = np.array(differences)

    # Sign test
    n_positive = int((differences > 0).sum())
    n_negative = int((differences < 0).sum())
    n_zero = int((differences == 0).sum())
    sign_fraction = n_positive / (n_positive + n_negative) if (n_positive + n_negative) > 0 else 0.5

    # Bootstrap CI on mean difference
    mean_diff, ci_lo, ci_hi = paired_bootstrap_ci(differences)

    stats = {
        "n_eligible": n_eligible,
        "horizon_batches": horizon_batches,
        "horizon_edits": horizon_batches * 100,
        "mean_diff": round(mean_diff, 8),
        "median_diff": round(float(np.median(differences)), 8),
        "ci_95_mean": [round(ci_lo, 8), round(ci_hi, 8)],
        "sign_test": {
            "n_positive": n_positive,
            "n_negative": n_negative,
            "n_zero": n_zero,
            "fraction_dispersed_greater": round(sign_fraction, 4),
        },
        "std_diff": round(float(np.std(differences)), 8),
    }

    print(f"    Mean dU (dispersed - clustered): {mean_diff:.6f} [{ci_lo:.6f}, {ci_hi:.6f}]")
    print(f"    Median dU: {np.median(differences):.6f}")
    print(f"    Sign: {n_positive}/{n_positive+n_negative} dispersed > clustered ({sign_fraction:.1%})")

    return stats


# ─── Figure ─────────────────────────────────────────────────────────────────

def make_figure(
    phase2_clustered: Optional[Dict],
    phase2_dispersed: Optional[Dict],
    percase_eval: Optional[Dict],
    seed: int,
    output_path: Path,
):
    """Generate the 3-panel main-paper figure."""
    try:
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches
    except ImportError:
        print("matplotlib not available, skipping figure generation")
        return

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))

    # ─── Panel A: Cumulative path interference by batch ─────────────
    ax = axes[0]
    if phase2_clustered and phase2_dispersed:
        # Reconstruct cumulative path per batch for first-1K
        for data, label, color in [
            (phase2_clustered, "Clustered", "#2196F3"),
            (phase2_dispersed, "Dispersed", "#F44336"),
        ]:
            batch_results = data["batch_results"]
            batches = [r["batch_idx"] for r in batch_results]
            # Use running mean interference as proxy for cumulative
            cum_mean = np.cumsum([r["mean_interference"] for r in batch_results])
            ax.plot(batches, cum_mean, color=color, label=label, linewidth=1.5)

        ax.set_xlabel("Batch index")
        ax.set_ylabel("Cumulative mean interference")
        ax.set_title("A. Path interference (first-1K keys)")
        ax.legend(frameon=False)
    else:
        ax.text(0.5, 0.5, "Phase 2 data\nnot available", ha="center", va="center",
                transform=ax.transAxes, fontsize=12, color="gray")
        ax.set_title("A. Path interference")

    # ─── Panel B: Retained vs Forgotten distributions ───────────────
    ax = axes[1]
    plotted = False
    if phase2_clustered and percase_eval:
        for data, cond_key, label, color, pos_offset in [
            (phase2_clustered, "AlphaEdit/key_clustered", "Clust.", "#2196F3", -0.2),
            (phase2_dispersed, "AlphaEdit/key_dispersed", "Disp.", "#F44336", 0.2),
        ]:
            if data is None or cond_key not in percase_eval:
                continue

            case_ids = data["first_1K"]["case_ids"]
            U_path = np.array(data["first_1K"]["U_path"])
            eval_by_cid = {r["case_id"]: r for r in percase_eval[cond_key]}

            retained = [U_path[i] for i, cid in enumerate(case_ids)
                       if cid in eval_by_cid and eval_by_cid[cid]["efficacy"] >= 0.5]
            forgotten = [U_path[i] for i, cid in enumerate(case_ids)
                        if cid in eval_by_cid and eval_by_cid[cid]["efficacy"] < 0.5]

            if retained and forgotten:
                parts = ax.violinplot(
                    [retained, forgotten],
                    positions=[1 + pos_offset, 2 + pos_offset],
                    widths=0.35, showmedians=True, showextrema=False,
                )
                for pc in parts["bodies"]:
                    pc.set_facecolor(color)
                    pc.set_alpha(0.4)
                parts["cmedians"].set_color(color)
                plotted = True

    if plotted:
        ax.set_xticks([1, 2])
        ax.set_xticklabels(["Retained", "Forgotten"])
        ax.set_ylabel("U_path")
        ax.set_title("B. Retained vs Forgotten")
        # Manual legend
        patches = [mpatches.Patch(color="#2196F3", alpha=0.4, label="Clustered"),
                   mpatches.Patch(color="#F44336", alpha=0.4, label="Dispersed")]
        ax.legend(handles=patches, frameon=False, fontsize=9)
    else:
        ax.text(0.5, 0.5, "Behavioral data\nnot available", ha="center", va="center",
                transform=ax.transAxes, fontsize=12, color="gray")
        ax.set_title("B. Retained vs Forgotten")

    # ─── Panel C: Matched-case difference distribution ──────────────
    ax = axes[2]
    if phase2_clustered and phase2_dispersed:
        clust_case_ids = phase2_clustered["all_5K"]["case_ids"]
        disp_case_ids = phase2_dispersed["all_5K"]["case_ids"]
        clust_U = np.array(phase2_clustered["all_5K"]["U_path"])
        disp_U = np.array(phase2_dispersed["all_5K"]["U_path"])
        clust_install = np.array(phase2_clustered["all_5K"]["installation_batch"])
        disp_install = np.array(phase2_dispersed["all_5K"]["installation_batch"])

        clust_map = {cid: i for i, cid in enumerate(clust_case_ids)}
        disp_map = {cid: i for i, cid in enumerate(disp_case_ids)}

        # 1K-exposure eligible
        max_batch = 39
        diffs = []
        for cid in set(clust_case_ids) & set(disp_case_ids):
            ci, di = clust_map[cid], disp_map[cid]
            if clust_install[ci] <= max_batch and disp_install[di] <= max_batch:
                diffs.append(disp_U[di] - clust_U[ci])

        if diffs:
            diffs = np.array(diffs)
            ax.hist(diffs, bins=50, color="#9C27B0", alpha=0.6, edgecolor="white", linewidth=0.5)
            ax.axvline(0, color="black", linestyle="--", linewidth=0.8)
            ax.axvline(np.mean(diffs), color="#E91E63", linestyle="-", linewidth=1.5,
                      label=f"Mean={np.mean(diffs):.4f}")
            ax.set_xlabel("$\\Delta U_i$ (dispersed - clustered)")
            ax.set_ylabel("Count")
            ax.set_title(f"C. Same-case paired (n={len(diffs)})")
            ax.legend(frameon=False, fontsize=9)
        else:
            ax.text(0.5, 0.5, "No eligible\npairs", ha="center", va="center",
                    transform=ax.transAxes, fontsize=12, color="gray")
            ax.set_title("C. Same-case paired")
    else:
        ax.text(0.5, 0.5, "Phase 2 data\nnot available", ha="center", va="center",
                transform=ax.transAxes, fontsize=12, color="gray")
        ax.set_title("C. Same-case paired")

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"\n  Figure saved: {output_path}")
    plt.close()


# ─── Main ──────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Interference figure + statistics")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--phase1_only", action="store_true",
                        help="Only analyze Phase 1 coarse data")
    parser.add_argument("--output_dir", type=str, default=None)
    args = parser.parse_args()

    output_dir = Path(args.output_dir) if args.output_dir else get_result_root() / "interference" / "figures" / f"seed{args.seed}"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load data per condition
    conditions = [
        ("AlphaEdit", "key_clustered"),
        ("AlphaEdit", "key_dispersed"),
    ]

    phase2_clust = None
    phase2_disp = None
    percase_clust = None
    percase_disp = None

    if not args.phase1_only:
        phase2_clust = load_phase2(args.seed, "AlphaEdit", "key_clustered")
        phase2_disp = load_phase2(args.seed, "AlphaEdit", "key_dispersed")

    phase1_clust = load_phase1(args.seed, "AlphaEdit", "key_clustered")
    phase1_disp = load_phase1(args.seed, "AlphaEdit", "key_dispersed")
    percase_clust = load_percase(args.seed, "AlphaEdit", "key_clustered")
    percase_disp = load_percase(args.seed, "AlphaEdit", "key_dispersed")

    use_fine = phase2_clust is not None and phase2_disp is not None

    print("=" * 70)
    print("Update-Level Interference: Statistical Analysis")
    print(f"  Seed: {args.seed}")
    print(f"  Data: {'Phase 2 (fine-grained)' if use_fine else 'Phase 1 (coarse)'}")
    print(f"  Behavioral eval: {'available' if percase_clust else 'NOT available'}")
    print("=" * 70)

    all_stats = {}

    # Within-ordering analysis (primary)
    for source, percase, ordering_name in [
        (phase2_clust if use_fine else phase1_clust, percase_clust, "AlphaEdit/key_clustered"),
        (phase2_disp if use_fine else phase1_disp, percase_disp, "AlphaEdit/key_dispersed"),
    ]:
        if source is None or percase is None:
            print(f"\n  SKIP {ordering_name}: data missing")
            continue

        first_1k = source["first_1K"] if "first_1K" in source else source
        case_ids = first_1k["case_ids"]
        U_path = np.array(first_1k["U_path"])

        stats = within_ordering_analysis(
            U_path=U_path,
            case_ids=case_ids,
            percase_eval=percase,
            ordering_name=ordering_name,
        )
        all_stats[f"within_{ordering_name}"] = stats

    # Matched-exposure analysis (secondary)
    if use_fine:
        for horizon in [10, 20]:
            stats = matched_exposure_analysis(
                phase2_clust, phase2_disp, args.seed, horizon_batches=horizon
            )
            all_stats[f"matched_exposure_{horizon*100}"] = stats

    # Save statistics
    stats_path = output_dir / "interference_stats.json"
    with open(stats_path, "w") as f:
        json.dump(all_stats, f, indent=2)
    print(f"\n  Statistics saved: {stats_path}")

    # Generate figure
    fig_path = output_dir / "fig_interference.pdf"
    percase_combined = {}
    if percase_clust:
        percase_combined["AlphaEdit/key_clustered"] = percase_clust
    if percase_disp:
        percase_combined["AlphaEdit/key_dispersed"] = percase_disp
    make_figure(
        phase2_clustered=phase2_clust,
        phase2_dispersed=phase2_disp,
        percase_eval=percase_combined if percase_combined else None,
        seed=args.seed,
        output_path=fig_path,
    )


if __name__ == "__main__":
    main()

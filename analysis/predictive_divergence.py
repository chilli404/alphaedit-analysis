#!/usr/bin/env python3
"""
Predictive Divergence Analysis: Identifies leading indicators of seed-level collapse.

Joins mechanism metrics (cache geometry from JSONL) with behavioral metrics
(retention from failure curve checkpoints) at each 1K-edit boundary, then
identifies which mechanism metric diverges between seeds BEFORE behavioral
retention collapses.

Key question: Does cache_condition or top_sv_share diverge between seeds
at 3K-4K edits, predicting the retention collapse that manifests at 5K-7K?

Data sources (all local):
  - results/mechanism_analysis/seed{S}/mechanism_seed{S}_*.jsonl
  - results/failure_curve_checkpointed/seed{S}/{N}edits/AlphaEdit/run_000/*.json

Output:
  - results/figures/predictive_divergence/divergence_timeline.png
  - results/figures/predictive_divergence/lead_lag_heatmap.png
  - results/figures/predictive_divergence/divergence_summary.csv

Usage:
    uv run python -m analysis.predictive_divergence
    uv run python -m analysis.predictive_divergence --seeds 42 2024
    uv run python -m analysis.predictive_divergence --mechanism_dir results/mechanism_analysis
"""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

# Paper-quality matplotlib settings
plt.rcParams.update({
    "font.size": 10,
    "axes.labelsize": 11,
    "axes.titlesize": 12,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "font.family": "serif",
})

SEED_COLORS = {42: "#2196F3", 2024: "#E91E63", 137: "#4CAF50", 7: "#FF9800", 99: "#9C27B0"}
DEFAULT_SEEDS = [42, 2024]
BATCH_SIZE = 100


# ─── Data Loading ────────────────────────────────────────────────────────────


def load_mechanism_metrics(results_dir: Path, seeds: list[int]) -> pd.DataFrame:
    """
    Load mechanism analysis JSONL files for specified seeds.

    Returns DataFrame with columns:
        seed, total_edits, layer_idx, cache_numerical_rank, cache_effective_rank,
        cache_condition, cache_stable_rank, cache_top_sv_share
    """
    rows = []
    for seed in seeds:
        seed_dir = results_dir / f"seed{seed}"
        if not seed_dir.is_dir():
            print(f"  WARNING: No mechanism data for seed {seed} at {seed_dir}")
            continue

        for jsonl_path in sorted(seed_dir.glob("*.jsonl")):
            with open(jsonl_path) as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    record = json.loads(line)
                    cache = record.get("cache", {})
                    rows.append({
                        "seed": record.get("seed", seed),
                        "total_edits": record["total_edits"],
                        "layer_idx": record["layer_idx"],
                        "cache_numerical_rank": cache.get("cache_numerical_rank"),
                        "cache_effective_rank": cache.get("cache_effective_rank"),
                        "cache_condition": cache.get("cache_condition"),
                        "cache_stable_rank": cache.get("cache_stable_rank"),
                        "cache_top_sv_share": cache.get("cache_top_sv_share"),
                    })

    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows)


def compute_retention_per_checkpoint(fc_dir: Path, seeds: list[int]) -> pd.DataFrame:
    """
    Compute behavioral retention metrics from failure curve checkpoint results.

    For each (seed, total_edits), loads all per-case JSON files and computes:
      - overall_efficacy: fraction of ALL edits still recalled
      - early_cohort_retention: fraction of first-1000 edits still recalled
      - generalization_rate: average paraphrase success across all edits

    Returns DataFrame with columns:
        seed, total_edits, overall_efficacy, early_cohort_retention, generalization_rate
    """
    rows = []
    for seed in seeds:
        seed_dir = fc_dir / f"seed{seed}"
        if not seed_dir.is_dir():
            print(f"  WARNING: No failure curve data for seed {seed} at {seed_dir}")
            continue

        for edits_dir in sorted(seed_dir.iterdir()):
            if not edits_dir.is_dir() or not edits_dir.name.endswith("edits"):
                continue
            total_edits = int(edits_dir.name.replace("edits", ""))

            # Find AlphaEdit results (support nested and flat layouts)
            case_files = []
            for candidate in [
                edits_dir / "AlphaEdit" / "run_000",
                edits_dir / "results" / "AlphaEdit" / "run_000",
            ]:
                if candidate.is_dir():
                    case_files = list(candidate.glob("*edits-case_*.json"))
                    break

            if not case_files:
                # Try deeper nesting patterns
                for pattern in edits_dir.rglob("*edits-case_*.json"):
                    case_files.append(pattern)

            if not case_files:
                continue

            # Parse metrics from case files
            efficacies = []
            paraphrases = []
            early_efficacies = []  # cases with case_id < 1000

            for cf in case_files:
                try:
                    with open(cf) as f:
                        case_data = json.load(f)
                except (json.JSONDecodeError, IOError):
                    continue

                case_id = case_data.get("case_id", -1)

                # Extract efficacy (post-rewrite)
                post = case_data.get("post", {})
                rewrite_prompts = post.get("rewrite_prompts_correct", [])
                if rewrite_prompts:
                    eff = np.mean(rewrite_prompts)
                    efficacies.append(eff)
                    if case_id < 10:  # First batch = early cohort
                        early_efficacies.append(eff)

                # Paraphrase success
                para_prompts = post.get("paraphrase_prompts_correct", [])
                if para_prompts:
                    paraphrases.append(np.mean(para_prompts))

            if efficacies:
                rows.append({
                    "seed": seed,
                    "total_edits": total_edits,
                    "overall_efficacy": float(np.mean(efficacies)),
                    "early_cohort_retention": float(np.mean(early_efficacies)) if early_efficacies else np.nan,
                    "generalization_rate": float(np.mean(paraphrases)) if paraphrases else np.nan,
                })

    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows)


def join_mechanism_and_behavior(mech_df: pd.DataFrame, behav_df: pd.DataFrame) -> pd.DataFrame:
    """
    Join mechanism and behavioral DataFrames on (seed, total_edits).

    Mechanism metrics are averaged across layers before joining.
    """
    if mech_df.empty or behav_df.empty:
        return pd.DataFrame()

    # Aggregate mechanism metrics across layers
    mech_agg = mech_df.groupby(["seed", "total_edits"]).agg({
        "cache_numerical_rank": "mean",
        "cache_effective_rank": "mean",
        "cache_condition": "mean",
        "cache_stable_rank": "mean",
        "cache_top_sv_share": "mean",
    }).reset_index()

    # Join
    joined = pd.merge(mech_agg, behav_df, on=["seed", "total_edits"], how="outer")
    joined = joined.sort_values(["seed", "total_edits"]).reset_index(drop=True)
    return joined


# ─── Divergence Analysis ─────────────────────────────────────────────────────


def compute_divergence_index(
    joined_df: pd.DataFrame,
    metric_col: str,
    seeds: list[int],
    threshold_pct: float = 20.0,
) -> int | None:
    """
    Find first edit count where metric diverges between seeds by > threshold_pct
    relative to the initial gap.

    Returns total_edits at first divergence, or None if never diverges.
    """
    if len(seeds) < 2:
        return None

    s1, s2 = seeds[0], seeds[1]
    df1 = joined_df[joined_df["seed"] == s1].set_index("total_edits")[metric_col].dropna()
    df2 = joined_df[joined_df["seed"] == s2].set_index("total_edits")[metric_col].dropna()

    common_edits = sorted(set(df1.index) & set(df2.index))
    if len(common_edits) < 2:
        return None

    # Compute absolute relative difference at each point
    # Normalize by mean of the two values at that point
    for edits in common_edits:
        v1, v2 = df1[edits], df2[edits]
        mean_val = (abs(v1) + abs(v2)) / 2
        if mean_val < 1e-10:
            continue
        pct_diff = abs(v1 - v2) / mean_val * 100
        if pct_diff > threshold_pct:
            return edits

    return None


def compute_lead_lag_correlation(
    joined_df: pd.DataFrame,
    metric_col: str,
    target_col: str = "overall_efficacy",
    max_lag: int = 3,
) -> dict:
    """
    Compute Pearson correlation between metric[t-lag] and target[t].

    Returns dict: {lag: {"r": float, "p": float, "n": int}}

    Operates across all seeds pooled.
    """
    results = {}
    df = joined_df.dropna(subset=[metric_col, target_col]).copy()
    df = df.sort_values(["seed", "total_edits"])

    for lag in range(max_lag + 1):
        pairs_x = []
        pairs_y = []

        for seed in df["seed"].unique():
            seed_df = df[df["seed"] == seed].sort_values("total_edits").reset_index(drop=True)
            if len(seed_df) <= lag:
                continue
            # metric at time t-lag, target at time t
            for t in range(lag, len(seed_df)):
                x_val = seed_df.iloc[t - lag][metric_col]
                y_val = seed_df.iloc[t][target_col]
                if pd.notna(x_val) and pd.notna(y_val):
                    pairs_x.append(x_val)
                    pairs_y.append(y_val)

        if len(pairs_x) >= 3:
            r, p = stats.pearsonr(pairs_x, pairs_y)
            results[lag] = {"r": round(r, 4), "p": round(p, 4), "n": len(pairs_x)}
        else:
            results[lag] = {"r": np.nan, "p": np.nan, "n": len(pairs_x)}

    return results


# ─── Plotting ────────────────────────────────────────────────────────────────


def plot_divergence_timeline(joined_df: pd.DataFrame, seeds: list[int], output_dir: Path):
    """
    6-panel figure showing per-seed trajectories with divergence markers.

    Panels: retention, effective_rank, condition, top_sv_share,
            stable_rank, numerical_rank
    """
    metrics = [
        ("overall_efficacy", "Retention (Overall Efficacy)", False),
        ("cache_effective_rank", "Effective Rank", False),
        ("cache_condition", "Condition Number", True),
        ("cache_top_sv_share", "Top SV Share", False),
        ("cache_stable_rank", "Stable Rank", False),
        ("cache_numerical_rank", "Numerical Rank", False),
    ]

    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    fig.suptitle(
        "Predictive Divergence: Mechanism Metrics vs Behavioral Collapse\n"
        f"Seeds: {', '.join(str(s) for s in seeds)}",
        fontsize=13, fontweight="bold",
    )

    for idx, (col, label, use_log) in enumerate(metrics):
        ax = axes[idx // 3, idx % 3]

        for seed in seeds:
            seed_df = joined_df[joined_df["seed"] == seed].sort_values("total_edits")
            x = seed_df["total_edits"].values
            y = seed_df[col].values

            color = SEED_COLORS.get(seed, "#666666")
            ax.plot(x, y, "o-", color=color, label=f"Seed {seed}",
                    markersize=4, linewidth=1.5)

        # Mark divergence point
        div_point = compute_divergence_index(joined_df, col, seeds)
        if div_point is not None:
            ax.axvline(div_point, color="red", linestyle="--", alpha=0.5, linewidth=1)
            ax.annotate(f"Δ>{20}%\n@ {div_point}",
                        xy=(div_point, ax.get_ylim()[1] * 0.9),
                        fontsize=7, color="red", ha="center")

        if use_log:
            ax.set_yscale("log")
        ax.set_xlabel("Total Edits")
        ax.set_ylabel(label)
        ax.set_title(f"{'ABCDEF'[idx]}. {label}")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    out_path = output_dir / "divergence_timeline.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"  Saved: {out_path}")
    plt.close()


def plot_lead_lag_heatmap(correlations: dict, output_dir: Path):
    """
    Heatmap: rows=metrics, cols=lags(0-3), color=correlation with retention.
    """
    metrics = list(correlations.keys())
    max_lag = max(len(v) for v in correlations.values()) if correlations else 4
    lags = list(range(max_lag))

    # Build matrix
    matrix = np.full((len(metrics), len(lags)), np.nan)
    for i, metric in enumerate(metrics):
        for lag_val, lag_data in correlations[metric].items():
            if lag_val < len(lags):
                matrix[i, lag_val] = lag_data["r"]

    fig, ax = plt.subplots(figsize=(8, 5))
    im = ax.imshow(matrix, aspect="auto", cmap="RdBu_r", vmin=-1, vmax=1)

    ax.set_xticks(range(len(lags)))
    ax.set_xticklabels([f"Lag {l}" for l in lags])
    ax.set_yticks(range(len(metrics)))
    ax.set_yticklabels([m.replace("cache_", "") for m in metrics])

    # Annotate cells
    for i in range(len(metrics)):
        for j in range(len(lags)):
            val = matrix[i, j]
            if not np.isnan(val):
                ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                        fontsize=9, color="white" if abs(val) > 0.5 else "black")

    plt.colorbar(im, ax=ax, label="Pearson r")
    ax.set_title("Lead-Lag Correlation: Mechanism Metrics → Retention\n"
                 "(negative r = metric increase predicts retention decrease)")
    ax.set_xlabel("Temporal Lag (1 unit = 1K edits)")

    plt.tight_layout()
    out_path = output_dir / "lead_lag_heatmap.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"  Saved: {out_path}")
    plt.close()


# ─── Main ────────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(
        description="Predictive divergence analysis: identify leading indicators of collapse"
    )
    parser.add_argument("--mechanism_dir", type=str, default="results/mechanism_analysis",
                        help="Directory with mechanism analysis JSONL files")
    parser.add_argument("--fc_dir", type=str, default="results/failure_curve_checkpointed",
                        help="Directory with failure curve checkpoint results")
    parser.add_argument("--output_dir", type=str, default="results/figures/predictive_divergence",
                        help="Output directory for plots and CSV")
    parser.add_argument("--seeds", type=int, nargs="+", default=DEFAULT_SEEDS,
                        help="Seeds to compare")
    parser.add_argument("--divergence_threshold", type=float, default=20.0,
                        help="Percent difference threshold for divergence detection")
    args = parser.parse_args()

    mechanism_dir = Path(args.mechanism_dir)
    fc_dir = Path(args.fc_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("Predictive Divergence Analysis")
    print(f"  Mechanism data:  {mechanism_dir}")
    print(f"  Failure curve:   {fc_dir}")
    print(f"  Seeds:           {args.seeds}")
    print(f"  Output:          {output_dir}")
    print("=" * 70)

    # ─── Load data ────────────────────────────────────────────────────────
    print("\n1. Loading mechanism metrics...")
    mech_df = load_mechanism_metrics(mechanism_dir, args.seeds)
    if mech_df.empty:
        print("  ERROR: No mechanism data found. Run mechanism_analyzer first.")
        return
    print(f"  Loaded {len(mech_df)} records across "
          f"{mech_df['seed'].nunique()} seeds, "
          f"{mech_df['total_edits'].nunique()} checkpoints")

    print("\n2. Computing retention from failure curve checkpoints...")
    behav_df = compute_retention_per_checkpoint(fc_dir, args.seeds)
    if behav_df.empty:
        print("  ERROR: No failure curve data found.")
        return
    print(f"  Computed retention for {len(behav_df)} (seed, checkpoint) pairs")
    print(f"  Edit counts: {sorted(behav_df['total_edits'].unique())}")

    print("\n3. Joining mechanism and behavioral data...")
    joined_df = join_mechanism_and_behavior(mech_df, behav_df)
    if joined_df.empty:
        print("  ERROR: No overlapping data points between mechanism and behavior.")
        return
    print(f"  Joined DataFrame: {len(joined_df)} rows")

    # ─── Divergence analysis ──────────────────────────────────────────────
    print("\n4. Computing divergence indices...")
    mechanism_metrics = [
        "cache_numerical_rank", "cache_effective_rank",
        "cache_condition", "cache_stable_rank", "cache_top_sv_share",
    ]

    divergence_results = {}
    for metric in mechanism_metrics + ["overall_efficacy"]:
        div_edits = compute_divergence_index(
            joined_df, metric, args.seeds, threshold_pct=args.divergence_threshold
        )
        divergence_results[metric] = div_edits
        status = f"{div_edits} edits" if div_edits else "never"
        print(f"  {metric:30s} diverges at: {status}")

    # ─── Lead-lag correlation ─────────────────────────────────────────────
    print("\n5. Computing lead-lag correlations...")
    all_correlations = {}
    for metric in mechanism_metrics:
        corr = compute_lead_lag_correlation(joined_df, metric, "overall_efficacy", max_lag=3)
        all_correlations[metric] = corr
        # Print strongest correlation
        best_lag = min(corr.keys(), key=lambda l: abs(corr[l].get("r", 0)) * -1)
        best_r = corr[best_lag]["r"]
        print(f"  {metric:30s} best lag={best_lag}, r={best_r:.3f}")

    # ─── Plotting ─────────────────────────────────────────────────────────
    print("\n6. Generating plots...")
    plot_divergence_timeline(joined_df, args.seeds, output_dir)
    plot_lead_lag_heatmap(all_correlations, output_dir)

    # ─── Summary CSV ──────────────────────────────────────────────────────
    print("\n7. Writing summary...")
    summary_rows = []
    for metric in mechanism_metrics + ["overall_efficacy"]:
        row = {
            "metric": metric,
            "divergence_at_edits": divergence_results.get(metric),
            "divergence_threshold_pct": args.divergence_threshold,
        }
        if metric in all_correlations:
            for lag, lag_data in all_correlations[metric].items():
                row[f"corr_lag{lag}_r"] = lag_data["r"]
                row[f"corr_lag{lag}_p"] = lag_data["p"]
        summary_rows.append(row)

    summary_df = pd.DataFrame(summary_rows)
    csv_path = output_dir / "divergence_summary.csv"
    summary_df.to_csv(csv_path, index=False)
    print(f"  Saved: {csv_path}")

    # ─── Key findings ─────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("KEY FINDINGS")
    print("=" * 70)

    ret_div = divergence_results.get("overall_efficacy")
    print(f"\nRetention diverges at: {ret_div} edits" if ret_div else
          "\nRetention: no significant divergence detected")

    leading_indicators = []
    for metric in mechanism_metrics:
        mech_div = divergence_results.get(metric)
        if mech_div and ret_div and mech_div < ret_div:
            lead = ret_div - mech_div
            leading_indicators.append((metric, mech_div, lead))

    if leading_indicators:
        print("\nLeading indicators (diverge BEFORE retention):")
        for metric, div_at, lead in sorted(leading_indicators, key=lambda x: x[1]):
            print(f"  {metric:30s} diverges at {div_at} edits "
                  f"({lead} edits early warning)")
    else:
        print("\nNo mechanism metrics diverge before retention.")
        print("This may indicate insufficient temporal resolution or threshold is too high.")


if __name__ == "__main__":
    main()

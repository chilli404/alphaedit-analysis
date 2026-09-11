"""Continuous NLL margin survival model.

Replaces the binary first-failure model (which broke under prob-pref due to
83% survival rate leaving too few failures for logistic regression).

Instead of binary survived/failed, predicts the continuous margin:
    margin_i(t) = NLL(target_true) - NLL(target_new)
    Positive = edit still holds (target_new more probable).
    margin <= 0 = edit has failed by prob-pref.

The NLL values are already stored in per-case JSON files at
    post.rewrite_prompts_probs[0].target_new / target_true

Models:
    M0: margin ~ age_norm + checkpoint_FE
    M1: M0 + installation_margin
    M2: M1 + max_cosine_exposure
    M3: M2 + relation_FE

Also fits a margin-decline model:
    delta_margin = margin(t) - margin(t_install)

Evaluated with leave-one-trajectory-out R², MSE, cosine coefficient + CI.

Usage:
    uv run python -m analysis.margin_survival_model
    uv run python -m analysis.margin_survival_model --output-dir results/figures/paper
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

try:
    import pandas as pd
    import statsmodels.api as sm
    from sklearn.linear_model import Ridge
    from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
except ImportError as e:
    raise ImportError(f"Missing dependency: {e}. Run: uv pip install pandas statsmodels scikit-learn")

from analysis.style import PROJECT, PAPER_OUTPUT, SEED_COLORS

RESULTS = PROJECT / "results"
FC_DIR = RESULTS / "failure_curve_checkpointed"
KEYS_DIR = RESULTS / "key_vectors"

TRAJECTORIES = [42, 2024, 137]
CHECKPOINTS = [3000, 5000, 7000, 10000]
ALG = "AlphaEdit"
BATCH_SIZE = 100


def load_edit_ordering(seed: int) -> Optional[List[int]]:
    for edits in [10000, 9000, 7000, 5000, 3000, 2000]:
        path = FC_DIR / f"seed{seed}" / f"{edits}edits" / ALG / "run_000" / "edit_ordering.json"
        if path.exists():
            with open(path) as f:
                data = json.load(f)
            return data["case_ids_ordered"]
    return None


def load_per_case_margins(seed: int, edits: int) -> Dict[int, dict]:
    """Load per-case NLL margins from per-case JSON files."""
    run_dir = FC_DIR / f"seed{seed}" / f"{edits}edits" / ALG / "run_000"
    if not run_dir.exists():
        return {}

    results = {}
    for f_path in run_dir.glob("*_edits-case_*.json"):
        with open(f_path) as f:
            data = json.load(f)

        case_id = data["case_id"]
        post = data.get("post", {})
        rewrite = data.get("requested_rewrite", {})
        probs = post.get("rewrite_prompts_probs", [])

        if not probs or not isinstance(probs[0], dict):
            continue

        nll_new = probs[0].get("target_new")
        nll_true = probs[0].get("target_true")
        if nll_new is None or nll_true is None:
            continue

        margin = nll_true - nll_new
        survived = int(margin > 0)

        results[case_id] = {
            "margin": margin,
            "nll_new": nll_new,
            "nll_true": nll_true,
            "survived": survived,
            "subject": rewrite.get("subject", ""),
            "relation_id": rewrite.get("relation_id", ""),
        }

    return results


def load_key_vectors(seed: int, keys_dir: Path = KEYS_DIR) -> Optional[Dict[int, np.ndarray]]:
    path = keys_dir / f"seed{seed}" / f"keys_seed{seed}.npz"
    if not path.exists():
        return None
    data = np.load(path)
    case_ids = data["case_ids"]
    keys = data["keys"]
    return {int(cid): keys[i] for i, cid in enumerate(case_ids)}


def compute_cosine_exposure(
    case_id: int, insertion_pos: int, max_pos: int,
    ordering: List[int], keys_matrix: np.ndarray, has_key: np.ndarray,
) -> Optional[float]:
    """Compute max cosine similarity to subsequent edits."""
    if insertion_pos >= max_pos or not has_key[insertion_pos]:
        return None
    subsequent_mask = has_key[insertion_pos + 1:max_pos]
    if not subsequent_mask.any():
        return None
    cosines = keys_matrix[insertion_pos] @ keys_matrix[insertion_pos + 1:max_pos].T
    cosines_valid = cosines[subsequent_mask]
    return float(np.max(cosines_valid)) if len(cosines_valid) > 0 else None


_keys_dir = KEYS_DIR

def build_margin_panel(seed: int) -> pd.DataFrame:
    """Build the edit x checkpoint panel with continuous margin as outcome."""
    ordering = load_edit_ordering(seed)
    if not ordering:
        print(f"  [WARN] No ordering for seed {seed}")
        return pd.DataFrame()

    key_vectors = load_key_vectors(seed, _keys_dir)

    n_total = len(ordering)
    pos_of = {cid: pos for pos, cid in enumerate(ordering)}

    # Load metadata from largest checkpoint
    metadata = {}
    for edits in sorted(CHECKPOINTS, reverse=True):
        outcomes = load_per_case_margins(seed, edits)
        for cid, data in outcomes.items():
            if cid not in metadata:
                metadata[cid] = {
                    "subject": data["subject"],
                    "relation_id": data["relation_id"],
                }

    # Pre-normalize key matrix for cosine computation
    keys_matrix = None
    has_key = None
    if key_vectors is not None:
        dim = next(iter(key_vectors.values())).shape[0]
        keys_matrix = np.zeros((n_total, dim), dtype=np.float32)
        has_key = np.zeros(n_total, dtype=bool)
        for pos, cid in enumerate(ordering):
            k = key_vectors.get(cid)
            if k is not None:
                norm = np.linalg.norm(k)
                if norm > 1e-10:
                    keys_matrix[pos] = k / norm
                    has_key[pos] = True

    # Load installation margins (first checkpoint after each edit)
    installation_margins = {}
    for edits in sorted(CHECKPOINTS):
        outcomes = load_per_case_margins(seed, edits)
        for cid, data in outcomes.items():
            if cid not in installation_margins and cid in pos_of:
                pos = pos_of[cid]
                if pos < edits:
                    installation_margins[cid] = data["margin"]

    # Build subject/relation indices for overlap
    ordered_relations = []
    for cid in ordering:
        meta = metadata.get(cid, {})
        ordered_relations.append(meta.get("relation_id", ""))

    relation_positions = defaultdict(list)
    for pos, rel in enumerate(ordered_relations):
        if rel:
            relation_positions[rel].append(pos)

    rows = []
    for checkpoint in CHECKPOINTS:
        outcomes = load_per_case_margins(seed, checkpoint)
        if not outcomes:
            continue

        max_pos = min(checkpoint, n_total)

        for cid, data in outcomes.items():
            if cid not in pos_of:
                continue
            pos = pos_of[cid]
            if pos >= max_pos:
                continue

            age = max_pos - pos - 1
            age_norm = age / max(max_pos - 1, 1)

            row = {
                "case_id": cid,
                "seed": seed,
                "checkpoint": checkpoint,
                "insertion_pos": pos,
                "age": age,
                "age_norm": age_norm,
                "margin": data["margin"],
                "survived": data["survived"],
                "nll_new": data["nll_new"],
                "nll_true": data["nll_true"],
                "relation_id": metadata.get(cid, {}).get("relation_id", ""),
            }

            # Installation margin
            row["install_margin"] = installation_margins.get(cid, np.nan)

            # Margin decline since installation
            if cid in installation_margins:
                row["margin_decline"] = installation_margins[cid] - data["margin"]
            else:
                row["margin_decline"] = np.nan

            # Relation overlap rate
            rel = ordered_relations[pos] if pos < len(ordered_relations) else ""
            if rel:
                rel_overlap = sum(1 for p in relation_positions[rel] if pos < p < max_pos)
            else:
                rel_overlap = 0
            n_subsequent = max(age, 1)
            row["relation_overlap_rate"] = rel_overlap / n_subsequent * 1000

            # Cosine exposure
            if keys_matrix is not None and has_key is not None:
                cos_val = compute_cosine_exposure(
                    cid, pos, max_pos, ordering, keys_matrix, has_key,
                )
                row["max_cosine"] = cos_val if cos_val is not None else np.nan
            else:
                row["max_cosine"] = np.nan

            rows.append(row)

    df = pd.DataFrame(rows)
    print(f"  seed {seed}: {len(df)} rows, "
          f"{df['survived'].mean():.1%} survived, "
          f"margin mean={df['margin'].mean():.3f}, "
          f"cosine coverage={df['max_cosine'].notna().mean():.1%}")
    return df


def fit_ols_nested(df: pd.DataFrame, seed_label: str) -> dict:
    """Fit nested OLS models on one trajectory's panel."""
    df_clean = df.dropna(subset=["margin", "age_norm", "install_margin", "max_cosine"])

    if len(df_clean) < 100:
        return {"error": f"Too few rows: {len(df_clean)}"}

    # Checkpoint dummies
    ckpt_dummies = pd.get_dummies(df_clean["checkpoint"], prefix="ckpt", drop_first=True, dtype=float)
    df_model = pd.concat([df_clean.reset_index(drop=True), ckpt_dummies.reset_index(drop=True)], axis=1)
    ckpt_cols = list(ckpt_dummies.columns)

    y = df_model["margin"].values

    # M0: age + checkpoint FE
    X0_cols = ["age_norm"] + ckpt_cols
    X0 = sm.add_constant(df_model[X0_cols].values.astype(float))
    m0 = sm.OLS(y, X0).fit()

    # M1: + installation margin
    X1_cols = ["age_norm", "install_margin"] + ckpt_cols
    X1 = sm.add_constant(df_model[X1_cols].values.astype(float))
    m1 = sm.OLS(y, X1).fit()

    # M2: + max_cosine
    X2_cols = ["age_norm", "install_margin", "max_cosine"] + ckpt_cols
    X2 = sm.add_constant(df_model[X2_cols].values.astype(float))
    m2 = sm.OLS(y, X2).fit()

    # Extract cosine coefficient from M2
    cos_idx = X2_cols.index("max_cosine") + 1  # +1 for constant
    cos_coef = m2.params[cos_idx]
    cos_se = m2.bse[cos_idx]
    cos_pval = m2.pvalues[cos_idx]
    cos_ci = m2.conf_int()[cos_idx]

    result = {
        "seed": seed_label,
        "n_obs": len(df_clean),
        "M0_R2": float(m0.rsquared),
        "M0_R2_adj": float(m0.rsquared_adj),
        "M0_AIC": float(m0.aic),
        "M1_R2": float(m1.rsquared),
        "M1_R2_adj": float(m1.rsquared_adj),
        "M1_AIC": float(m1.aic),
        "M2_R2": float(m2.rsquared),
        "M2_R2_adj": float(m2.rsquared_adj),
        "M2_AIC": float(m2.aic),
        "cosine_coef": float(cos_coef),
        "cosine_se": float(cos_se),
        "cosine_pval": float(cos_pval),
        "cosine_ci_lo": float(cos_ci[0]),
        "cosine_ci_hi": float(cos_ci[1]),
        "cosine_effect_per_01": float(cos_coef * 0.1),
        "R2_improvement_M2_over_M0": float(m2.rsquared - m0.rsquared),
        "R2_improvement_M2_over_M1": float(m2.rsquared - m1.rsquared),
        "AIC_improvement_M2_over_M1": float(m1.aic - m2.aic),
        "age_coef_M2": float(m2.params[1]),
        "install_margin_coef_M2": float(m2.params[2]),
    }

    return result


def fit_margin_decline(df: pd.DataFrame, seed_label: str) -> dict:
    """Fit model on margin decline (how much each edit's margin degraded)."""
    df_clean = df.dropna(subset=["margin_decline", "age_norm", "max_cosine"])
    df_clean = df_clean[df_clean["margin_decline"].notna()]

    if len(df_clean) < 100:
        return {"error": f"Too few rows: {len(df_clean)}"}

    ckpt_dummies = pd.get_dummies(df_clean["checkpoint"], prefix="ckpt", drop_first=True, dtype=float)
    df_model = pd.concat([df_clean.reset_index(drop=True), ckpt_dummies.reset_index(drop=True)], axis=1)
    ckpt_cols = list(ckpt_dummies.columns)

    y = df_model["margin_decline"].values

    X_cols = ["age_norm", "max_cosine"] + ckpt_cols
    X = sm.add_constant(df_model[X_cols].values.astype(float))
    model = sm.OLS(y, X).fit()

    cos_idx = X_cols.index("max_cosine") + 1
    return {
        "seed": seed_label,
        "n_obs": len(df_clean),
        "R2": float(model.rsquared),
        "cosine_coef": float(model.params[cos_idx]),
        "cosine_pval": float(model.pvalues[cos_idx]),
        "cosine_ci_lo": float(model.conf_int()[cos_idx][0]),
        "cosine_ci_hi": float(model.conf_int()[cos_idx][1]),
        "age_coef": float(model.params[1]),
    }


def loo_trajectory_evaluation(panels: Dict[int, pd.DataFrame]) -> dict:
    """Leave-one-trajectory-out evaluation using Ridge regression."""
    seeds = sorted(panels.keys())
    loo_results = {}

    feature_cols = ["age_norm", "install_margin", "max_cosine"]

    for held_out_seed in seeds:
        # Train on other seeds
        train_dfs = []
        for s in seeds:
            if s == held_out_seed:
                continue
            df = panels[s].dropna(subset=["margin"] + feature_cols)
            train_dfs.append(df)

        if not train_dfs:
            continue

        train = pd.concat(train_dfs, ignore_index=True)
        test = panels[held_out_seed].dropna(subset=["margin"] + feature_cols)

        if len(test) < 50:
            continue

        # Add checkpoint dummies
        all_checkpoints = sorted(set(train["checkpoint"].unique()) | set(test["checkpoint"].unique()))
        for ckpt in all_checkpoints[1:]:
            col = f"ckpt_{ckpt}"
            train[col] = (train["checkpoint"] == ckpt).astype(float)
            test[col] = (test["checkpoint"] == ckpt).astype(float)

        ckpt_cols = [f"ckpt_{c}" for c in all_checkpoints[1:]]
        all_features = feature_cols + ckpt_cols

        X_train = train[all_features].values.astype(float)
        y_train = train["margin"].values
        X_test = test[all_features].values.astype(float)
        y_test = test["margin"].values

        # M0: without cosine (baseline)
        no_cos_features = [c for c in all_features if c != "max_cosine"]
        no_cos_idx = [all_features.index(c) for c in no_cos_features]

        ridge_base = Ridge(alpha=1.0)
        ridge_base.fit(X_train[:, no_cos_idx], y_train)
        y_pred_base = ridge_base.predict(X_test[:, no_cos_idx])

        # M2: with cosine
        ridge_full = Ridge(alpha=1.0)
        ridge_full.fit(X_train, y_train)
        y_pred_full = ridge_full.predict(X_test)

        r2_base = r2_score(y_test, y_pred_base)
        r2_full = r2_score(y_test, y_pred_full)
        mse_base = mean_squared_error(y_test, y_pred_base)
        mse_full = mean_squared_error(y_test, y_pred_full)
        mae_base = mean_absolute_error(y_test, y_pred_base)
        mae_full = mean_absolute_error(y_test, y_pred_full)

        # Cosine coefficient from full model
        cos_feature_idx = all_features.index("max_cosine")
        cos_coef = ridge_full.coef_[cos_feature_idx]

        loo_results[held_out_seed] = {
            "n_test": len(test),
            "R2_base": float(r2_base),
            "R2_full": float(r2_full),
            "R2_improvement": float(r2_full - r2_base),
            "MSE_base": float(mse_base),
            "MSE_full": float(mse_full),
            "MAE_base": float(mae_base),
            "MAE_full": float(mae_full),
            "cosine_coef": float(cos_coef),
            "margin_mean": float(y_test.mean()),
            "margin_std": float(y_test.std()),
        }

    return loo_results


def generate_figure(panels: Dict[int, pd.DataFrame], per_traj: dict, loo: dict, output_dir: Path):
    """Generate 4-panel figure."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 11))

    # Panel A: Margin by age cohort at 10K
    ax = axes[0, 0]
    for seed in TRAJECTORIES:
        df = panels.get(seed)
        if df is None or df.empty:
            continue
        df_10k = df[df["checkpoint"] == 10000].copy()
        if df_10k.empty:
            continue
        n_cohorts = 10
        df_10k["cohort"] = pd.cut(df_10k["insertion_pos"], bins=n_cohorts, labels=False)
        cohort_means = df_10k.groupby("cohort")["margin"].mean()
        xs = np.arange(len(cohort_means))
        ax.plot(xs, cohort_means.values, "o-", color=SEED_COLORS.get(seed, "#666"),
                label=f"Seed {seed}", markersize=5, linewidth=1.5)
    ax.axhline(0, color="gray", linestyle="--", alpha=0.5, linewidth=1)
    ax.set_xlabel("Installation cohort (0 = oldest)")
    ax.set_ylabel("NLL margin (positive = edit holds)")
    ax.set_title("(a) Margin by Age Cohort at 10K")
    ax.legend(fontsize=9)

    # Panel B: Margin vs cosine exposure
    ax = axes[0, 1]
    for seed in TRAJECTORIES:
        df = panels.get(seed)
        if df is None or df.empty:
            continue
        df_10k = df[(df["checkpoint"] == 10000) & df["max_cosine"].notna()].copy()
        if df_10k.empty:
            continue
        sample = df_10k.sample(min(2000, len(df_10k)), random_state=42)
        ax.scatter(sample["max_cosine"], sample["margin"],
                   alpha=0.15, s=8, color=SEED_COLORS.get(seed, "#666"),
                   label=f"Seed {seed}")
    ax.axhline(0, color="gray", linestyle="--", alpha=0.5)
    ax.set_xlabel("Max cosine exposure to subsequent edits")
    ax.set_ylabel("NLL margin at 10K")
    ax.set_title("(b) Margin vs Key-Cosine Exposure")
    ax.legend(fontsize=9)

    # Panel C: LOO predictions
    ax = axes[1, 0]
    for seed in TRAJECTORIES:
        if seed not in loo:
            continue
        df = panels.get(seed)
        if df is None or df.empty:
            continue
        r2 = loo[seed]["R2_full"]
        r2_base = loo[seed]["R2_base"]
        c = SEED_COLORS.get(seed, "#666666")
        ax.bar(f"s{seed}\nbase", r2_base, color=c, alpha=0.4,
               edgecolor="black", linewidth=0.5)
        ax.bar(f"s{seed}\n+cosine", r2, color=c, alpha=0.9,
               edgecolor="black", linewidth=0.5)
    ax.set_ylabel("R² (held-out trajectory)")
    ax.set_title("(c) Leave-One-Trajectory-Out R²")
    ax.set_ylim(0, max(0.5, max(r.get("R2_full", 0) for r in loo.values()) * 1.2) if loo else 0.5)

    # Panel D: Cosine coefficient forest plot
    ax = axes[1, 1]
    labels = []
    coefs = []
    ci_los = []
    ci_his = []
    colors = []
    for seed in TRAJECTORIES:
        info = per_traj.get(str(seed), per_traj.get(seed, {}))
        if "cosine_coef" not in info:
            continue
        labels.append(f"Seed {seed}")
        coefs.append(info["cosine_coef"])
        ci_los.append(info["cosine_ci_lo"])
        ci_his.append(info["cosine_ci_hi"])
        colors.append(SEED_COLORS.get(seed, "#666"))

    if labels:
        y_pos = np.arange(len(labels))
        for i in range(len(labels)):
            ax.plot([ci_los[i], ci_his[i]], [y_pos[i], y_pos[i]],
                    color=colors[i], linewidth=2.5, solid_capstyle="round")
            ax.plot(coefs[i], y_pos[i], "o", color=colors[i], markersize=10,
                    markeredgecolor="black", markeredgewidth=0.5)
        ax.axvline(0, color="gray", linestyle="--", alpha=0.5, linewidth=1)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(labels)
        ax.set_xlabel("Cosine coefficient on NLL margin (negative = damaging)")
        ax.set_title("(d) Cosine Effect on Margin (95% CI)")

    plt.tight_layout()
    out_path = output_dir / "fig_margin_survival.png"
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    fig.savefig(out_path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    print(f"  Figure saved: {out_path}")


def main():
    parser = argparse.ArgumentParser(description="Continuous margin survival model")
    parser.add_argument("--output-dir", default=str(PAPER_OUTPUT))
    parser.add_argument("--keys-dir", default=str(KEYS_DIR))
    args = parser.parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    global _keys_dir
    _keys_dir = Path(args.keys_dir)

    print("Building continuous margin survival model...")
    print(f"  Keys: {KEYS_DIR}")
    print(f"  Output: {output_dir}")
    print()

    # Build panels
    panels = {}
    for seed in TRAJECTORIES:
        print(f"Building panel for seed {seed}...")
        df = build_margin_panel(seed)
        if not df.empty:
            panels[seed] = df

    if not panels:
        print("ERROR: No panels built. Check data availability.")
        return

    # Per-trajectory OLS models
    print("\nFitting per-trajectory OLS models...")
    per_traj_results = {}
    for seed, df in panels.items():
        print(f"  Seed {seed}:")
        result = fit_ols_nested(df, str(seed))
        per_traj_results[seed] = result
        if "error" in result:
            print(f"    ERROR: {result['error']}")
        else:
            print(f"    M0 R²={result['M0_R2']:.4f}, M1 R²={result['M1_R2']:.4f}, M2 R²={result['M2_R2']:.4f}")
            print(f"    Cosine coef={result['cosine_coef']:.4f} [{result['cosine_ci_lo']:.4f}, {result['cosine_ci_hi']:.4f}], p={result['cosine_pval']:.2e}")
            print(f"    R² improvement (cosine): M2-M1 = {result['R2_improvement_M2_over_M1']:+.4f}")
            print(f"    Effect per +0.1 cosine: {result['cosine_effect_per_01']:.4f} NLL margin points")

    # Per-trajectory margin decline models
    print("\nFitting margin decline models...")
    decline_results = {}
    for seed, df in panels.items():
        result = fit_margin_decline(df, str(seed))
        decline_results[seed] = result
        if "error" not in result:
            print(f"  Seed {seed}: R²={result['R2']:.4f}, cosine_coef={result['cosine_coef']:.4f}, p={result['cosine_pval']:.2e}")

    # Leave-one-trajectory-out
    print("\nLeave-one-trajectory-out evaluation...")
    loo_results = loo_trajectory_evaluation(panels)
    for seed, result in loo_results.items():
        print(f"  Held-out seed {seed}: R²_base={result['R2_base']:.4f}, R²_full={result['R2_full']:.4f}, "
              f"improvement={result['R2_improvement']:+.4f}, cosine_coef={result['cosine_coef']:.4f}")

    # Summary statistics
    all_df = pd.concat(panels.values(), ignore_index=True)
    summary = {
        "total_rows": len(all_df),
        "survival_rate": float(all_df["survived"].mean()),
        "margin_mean": float(all_df["margin"].mean()),
        "margin_std": float(all_df["margin"].std()),
        "margin_failed_mean": float(all_df[all_df["survived"] == 0]["margin"].mean()) if (all_df["survived"] == 0).any() else None,
        "margin_survived_mean": float(all_df[all_df["survived"] == 1]["margin"].mean()),
        "cosine_coverage": float(all_df["max_cosine"].notna().mean()),
    }

    if all_df["max_cosine"].notna().any():
        cos_data = all_df.dropna(subset=["max_cosine"])
        survived_cos = cos_data[cos_data["survived"] == 1]["max_cosine"].mean()
        failed_cos = cos_data[cos_data["survived"] == 0]["max_cosine"].mean() if (cos_data["survived"] == 0).any() else None
        summary["cosine_survived_mean"] = float(survived_cos)
        summary["cosine_failed_mean"] = float(failed_cos) if failed_cos is not None else None
        summary["cosine_diff"] = float(failed_cos - survived_cos) if failed_cos is not None else None

    # Save results
    output = {
        "model_type": "continuous_margin_survival",
        "outcome": "NLL_margin = NLL(target_true) - NLL(target_new)",
        "interpretation": "Positive margin = edit holds. Higher cosine → lower margin → edit more likely to fail.",
        "trajectories": TRAJECTORIES,
        "checkpoints": CHECKPOINTS,
        "summary": summary,
        "per_trajectory": {str(k): v for k, v in per_traj_results.items()},
        "margin_decline": {str(k): v for k, v in decline_results.items()},
        "leave_one_out": {str(k): v for k, v in loo_results.items()},
    }

    out_json = output_dir / "margin_survival_model.json" if output_dir == PAPER_OUTPUT else output_dir / "margin_survival_model.json"
    with open(out_json, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\n  Results saved: {out_json}")

    # Also save to results/ top level
    results_json = RESULTS / "margin_survival_model.json"
    with open(results_json, "w") as f:
        json.dump(output, f, indent=2)
    print(f"  Results saved: {results_json}")

    # Generate figure
    print("\nGenerating figure...")
    generate_figure(panels, per_traj_results, loo_results, output_dir)

    # Print summary
    print("\n" + "=" * 80)
    print("  CONTINUOUS MARGIN SURVIVAL MODEL SUMMARY")
    print("=" * 80)
    print(f"  Panel: {summary['total_rows']} rows, {summary['survival_rate']:.1%} survived (prob-pref)")
    print(f"  Margin: mean={summary['margin_mean']:.3f}, std={summary['margin_std']:.3f}")
    if summary.get("cosine_diff") is not None:
        print(f"  Cosine diff (failed - survived): {summary['cosine_diff']:+.4f}")
    print()
    for seed in TRAJECTORIES:
        r = per_traj_results.get(seed, {})
        if "error" in r:
            print(f"  Seed {seed}: {r['error']}")
            continue
        print(f"  Seed {seed}: cosine coef={r['cosine_coef']:.4f} "
              f"[{r['cosine_ci_lo']:.4f}, {r['cosine_ci_hi']:.4f}], "
              f"p={r['cosine_pval']:.2e}, "
              f"R² M0→M2: {r['M0_R2']:.4f}→{r['M2_R2']:.4f}")
    print()
    for seed in TRAJECTORIES:
        r = loo_results.get(seed, {})
        if r:
            print(f"  LOO seed {seed}: R²={r['R2_full']:.4f} (base {r['R2_base']:.4f}, +{r['R2_improvement']:+.4f})")


if __name__ == "__main__":
    main()

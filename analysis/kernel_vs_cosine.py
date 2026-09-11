#!/usr/bin/env python3
"""Compare interference kernel η vs raw cosine as first-failure predictors.

Loads kernel_scores.json from both orderings, merges into the survival panel,
and fits competing models to compare predictive performance.
"""

import json
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RESULTS = PROJECT_ROOT / "results"


def load_kernel_scores(ordering: str, seed: int) -> dict:
    path = RESULTS / "interference_kernel" / ordering / f"seed{seed}" / "kernel_scores.json"
    if not path.exists():
        return {}
    with open(path) as f:
        d = json.load(f)
    return {int(cid): eta for cid, eta in zip(d["case_ids"], d["eta_cumulative"])}


def build_comparison_panel(seed: int = 42):
    """Build a panel with both η and cosine for first-failure analysis."""
    # Load kernel scores for both orderings
    eta_clust = load_kernel_scores("key_clustered", seed)
    eta_disp = load_kernel_scores("key_dispersed", seed)
    eta_all = {**eta_clust, **eta_disp}

    if not eta_all:
        print("No kernel scores found")
        return None

    # Load the existing survival panel
    sys.path.insert(0, str(PROJECT_ROOT / "src" / "util"))
    sys.path.insert(0, str(PROJECT_ROOT / "analysis"))
    from interference_panel import build_panel

    panel = build_panel(seed, keys_dir=RESULTS / "matched_ordering" / "key_geometry")

    # Add η to each row
    n_added = 0
    for row in panel:
        cid = row["case_id"]
        if cid in eta_all:
            row["eta_kernel"] = eta_all[cid]
            n_added += 1

    print(f"Added η to {n_added}/{len(panel)} rows")
    return panel


def compare_predictors(panel):
    """Fit competing first-failure models: cosine-only vs kernel-only vs both."""
    import pandas as pd
    import statsmodels.formula.api as smf
    from sklearn.metrics import roc_auc_score

    df = pd.DataFrame(panel)

    # First-failure censoring
    df = df.sort_values(["case_id", "checkpoint"])
    rows = []
    for (cid, seed), group in df.groupby(["case_id", "seed"]):
        for _, row in group.iterrows():
            rows.append(row)
            if row["survived"] == 0:
                break
    df_ff = pd.DataFrame(rows)

    # Need both cosine and kernel
    df_ff = df_ff.dropna(subset=["max_cosine_subsequent"])
    df_ff = df_ff[df_ff["eta_kernel"].notna()]

    if len(df_ff) < 50:
        print(f"Insufficient data after filtering: {len(df_ff)} rows")
        return

    # Categorical
    df_ff["checkpoint_f"] = df_ff["checkpoint"].astype(str)
    max_edits = df_ff["checkpoint"].max()
    df_ff["age_bin_f"] = pd.cut(
        df_ff["insertion_pos"] / max_edits * 10000,
        bins=[0, 2500, 5000, 7500, 10001],
        labels=["Q1_young", "Q2", "Q3", "Q4_old"],
    ).astype(str)

    print(f"\nFirst-failure panel: {len(df_ff)} rows, {df_ff['survived'].sum()} survived, "
          f"{(1-df_ff['survived']).sum()} failed")

    # Standardize predictors for comparable coefficients
    for col in ["max_cosine_subsequent", "eta_kernel", "cumulative_interference"]:
        if col in df_ff.columns:
            mu = df_ff[col].mean()
            sd = df_ff[col].std()
            if sd > 0:
                df_ff[f"{col}_z"] = (df_ff[col] - mu) / sd

    results = {}

    # Drop any remaining NaN rows to ensure consistent sizes
    model_cols = ["survived", "checkpoint_f", "age_bin_f", "max_cosine_subsequent",
                  "cumulative_interference", "eta_kernel"]
    df_ff = df_ff.dropna(subset=model_cols).reset_index(drop=True)

    print(f"  After dropna: {len(df_ff)} rows")

    def fit_and_auc(formula, data):
        m = smf.logit(formula, data=data).fit(disp=0)
        pred = m.fittedvalues
        y_sub = data.loc[pred.index, "survived"]
        return m, float(roc_auc_score(y_sub, pred))

    # M_baseline: age + checkpoint only
    m_base, auc_base = fit_and_auc(
        "survived ~ C(checkpoint_f) + C(age_bin_f)", df_ff)
    results["M_baseline"] = {"AIC": m_base.aic, "AUC": auc_base}

    # M_cosine: + raw cosine
    m_cos, auc_cos = fit_and_auc(
        "survived ~ C(checkpoint_f) + C(age_bin_f) + max_cosine_subsequent + cumulative_interference",
        df_ff)
    results["M_cosine"] = {
        "AIC": m_cos.aic, "AUC": auc_cos,
        "cosine_coef": float(m_cos.params.get("max_cosine_subsequent", np.nan)),
        "cosine_pval": float(m_cos.pvalues.get("max_cosine_subsequent", np.nan)),
    }

    # M_kernel: + η only (no raw cosine)
    m_eta, auc_eta = fit_and_auc(
        "survived ~ C(checkpoint_f) + C(age_bin_f) + eta_kernel", df_ff)
    results["M_kernel"] = {
        "AIC": m_eta.aic, "AUC": auc_eta,
        "eta_coef": float(m_eta.params.get("eta_kernel", np.nan)),
        "eta_pval": float(m_eta.pvalues.get("eta_kernel", np.nan)),
    }

    # M_both: cosine + kernel
    m_both, auc_both = fit_and_auc(
        "survived ~ C(checkpoint_f) + C(age_bin_f) + max_cosine_subsequent + cumulative_interference + eta_kernel",
        df_ff)
    results["M_both"] = {
        "AIC": m_both.aic, "AUC": auc_both,
        "cosine_coef": float(m_both.params.get("max_cosine_subsequent", np.nan)),
        "cosine_pval": float(m_both.pvalues.get("max_cosine_subsequent", np.nan)),
        "eta_coef": float(m_both.params.get("eta_kernel", np.nan)),
        "eta_pval": float(m_both.pvalues.get("eta_kernel", np.nan)),
    }

    # LR tests
    from scipy.stats import chi2
    lr_cos_vs_eta = -2 * (m_cos.llf - m_both.llf)
    lr_eta_vs_cos = -2 * (m_eta.llf - m_both.llf)
    results["LR_cosine_adds_to_kernel"] = {
        "stat": float(lr_eta_vs_cos),
        "pval": float(chi2.sf(lr_eta_vs_cos, df=2)),
    }
    results["LR_kernel_adds_to_cosine"] = {
        "stat": float(lr_cos_vs_eta),
        "pval": float(chi2.sf(lr_cos_vs_eta, df=1)),
    }

    # Print summary
    print("\n" + "=" * 60)
    print("KERNEL vs COSINE: FIRST-FAILURE PREDICTION")
    print("=" * 60)
    for name, r in results.items():
        if "AUC" in r:
            print(f"  {name:15s}: AIC={r['AIC']:.1f}, AUC={r['AUC']:.4f}")
        elif "stat" in r:
            print(f"  {name}: χ²={r['stat']:.2f}, p={r['pval']:.2e}")

    print(f"\n  η-kernel adds to cosine: {'YES' if results['LR_kernel_adds_to_cosine']['pval'] < 0.05 else 'NO'}")
    print(f"  Cosine adds to η-kernel: {'YES' if results['LR_cosine_adds_to_kernel']['pval'] < 0.05 else 'NO'}")

    # Save
    out_path = RESULTS / "interference_kernel" / "kernel_vs_cosine_comparison.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  Saved: {out_path}")

    return results


if __name__ == "__main__":
    panel = build_comparison_panel(seed=42)
    if panel:
        compare_predictors(panel)

"""Multilayer key predictor comparison — tests whether layer-6 is special.

Fits the retention model with keys from each layer (4, 5, 6, 7, 8) independently,
then reports per-layer AUC. Addresses "layer-6 selection not justified" and
"multilayer predictor not tested."

Run: uv run python -m analysis.multilayer_predictor
"""

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from pathlib import Path
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.model_selection import GroupKFold

from analysis.interference_panel import (
    build_panel, TRAJECTORIES, load_key_vectors, batch_key_similarity,
    load_edit_ordering, CHECKPOINTS,
)

KEYS_BASE = Path("results/key_vectors")
LAYERS = [4, 5, 6, 7, 8]


def build_panel_for_layer(layer: int) -> pd.DataFrame:
    """Build panel with keys from a specific layer."""
    keys_dir = KEYS_BASE / f"layer{layer}"
    all_panels = []
    for seed in TRAJECTORIES:
        panel = build_panel(seed, keys_dir=keys_dir)
        all_panels.extend(panel)
    df = pd.DataFrame(all_panels)
    df["checkpoint_f"] = df["checkpoint"].astype(str)
    df["age_bin_f"] = pd.Categorical(
        df["age_bin"], categories=["Q1_young", "Q2", "Q3", "Q4_old"], ordered=True
    )
    return df


def evaluate_layer(df: pd.DataFrame, layer: int, n_splits: int = 5) -> dict:
    """Run group-disjoint CV for a single layer's key predictors."""
    df_keys = df.dropna(subset=["max_cosine_subsequent"]).copy()
    if len(df_keys) < 100:
        return {"layer": layer, "error": f"Only {len(df_keys)} rows with keys"}

    formula = (
        "survived ~ C(checkpoint_f) + C(age_bin_f) + "
        "max_cosine_subsequent + cumulative_interference + target_margin"
    )

    gkf = GroupKFold(n_splits=n_splits)
    groups = df_keys["case_id"].values
    aucs, pr_aucs = [], []

    for train_idx, test_idx in gkf.split(df_keys, groups=groups):
        train = df_keys.iloc[train_idx]
        test = df_keys.iloc[test_idx]
        try:
            model = smf.logit(formula, data=train).fit(disp=0, maxiter=50)
            y_pred = model.predict(test)
            y_true = test["survived"].values
            if len(np.unique(y_true)) < 2:
                continue
            aucs.append(roc_auc_score(y_true, y_pred))
            pr_aucs.append(average_precision_score(y_true, y_pred))
        except Exception:
            continue

    if not aucs:
        return {"layer": layer, "error": "No folds converged"}

    # Full-sample cosine OR
    try:
        full_fit = smf.logit(formula, data=df_keys).fit(disp=0, maxiter=100)
        cosine_coef = full_fit.params.get("max_cosine_subsequent", None)
        ci = full_fit.conf_int().loc["max_cosine_subsequent"]
        pval = full_fit.pvalues.get("max_cosine_subsequent", 1.0)
        cosine_or = float(np.exp(cosine_coef)) if cosine_coef is not None else None
        or_ci = (float(np.exp(ci[0])), float(np.exp(ci[1])))
    except Exception:
        cosine_or, or_ci, pval = None, (None, None), None

    return {
        "layer": layer,
        "auc_mean": np.mean(aucs),
        "auc_std": np.std(aucs),
        "pr_auc_mean": np.mean(pr_aucs),
        "pr_auc_std": np.std(pr_aucs),
        "n_folds": len(aucs),
        "n_rows": len(df_keys),
        "cosine_or": cosine_or,
        "or_ci": or_ci,
        "pval": pval,
    }


def main():
    print("=" * 70)
    print("MULTILAYER KEY PREDICTOR COMPARISON")
    print("=" * 70)
    print(f"\nLayers: {LAYERS}")
    print(f"Trajectories: {TRAJECTORIES}")
    print()

    results = []
    for layer in LAYERS:
        keys_dir = KEYS_BASE / f"layer{layer}"
        if not (keys_dir / "seed42").exists():
            # Try the original location for layer 6
            if layer == 6 and (KEYS_BASE / "seed42").exists():
                keys_dir = KEYS_BASE
            else:
                print(f"  Layer {layer}: SKIPPED (no keys at {keys_dir})")
                results.append({"layer": layer, "error": "keys not found"})
                continue

        print(f"  Building panel for layer {layer}...")
        df = build_panel_for_layer(layer) if layer != 6 else None

        # Layer 6 uses the original key location
        if layer == 6:
            all_panels = []
            for seed in TRAJECTORIES:
                panel = build_panel(seed, keys_dir=KEYS_BASE)
                all_panels.extend(panel)
            df = pd.DataFrame(all_panels)
            df["checkpoint_f"] = df["checkpoint"].astype(str)
            df["age_bin_f"] = pd.Categorical(
                df["age_bin"], categories=["Q1_young", "Q2", "Q3", "Q4_old"], ordered=True
            )

        result = evaluate_layer(df, layer)
        results.append(result)

        if "error" in result:
            print(f"  Layer {layer}: {result['error']}")
        else:
            or_lo, or_hi = result['or_ci']
            print(f"  Layer {layer}: AUC = {result['auc_mean']:.3f} ± {result['auc_std']:.3f}, "
                  f"OR = {result['cosine_or']:.3f} [{or_lo:.3f}, {or_hi:.3f}], "
                  f"p = {result['pval']:.1e}")

    # Summary table
    print("\n" + "=" * 70)
    print("SUMMARY: Per-Layer Predictor Performance")
    print("=" * 70)
    print(f"\n{'Layer':<8} {'AUC':<14} {'PR-AUC':<14} {'Cosine OR':<20} {'p-value':<12} {'N':<8}")
    print("-" * 70)
    for r in results:
        if "error" in r:
            print(f"  {r['layer']:<6} {r['error']}")
            continue
        or_str = f"{r['cosine_or']:.3f} [{r['or_ci'][0]:.3f}, {r['or_ci'][1]:.3f}]"
        print(f"  {r['layer']:<6} {r['auc_mean']:.3f}±{r['auc_std']:.3f}  "
              f"{r['pr_auc_mean']:.3f}±{r['pr_auc_std']:.3f}  "
              f"{or_str:<20} {r['pval']:.1e}  {r['n_rows']}")

    # Find best layer
    valid = [r for r in results if "error" not in r]
    if valid:
        best = max(valid, key=lambda r: r["auc_mean"])
        print(f"\n  Best layer: {best['layer']} (AUC = {best['auc_mean']:.3f})")
        l6 = next((r for r in valid if r["layer"] == 6), None)
        if l6 and l6 != best:
            gap = best["auc_mean"] - l6["auc_mean"]
            print(f"  Layer 6 AUC: {l6['auc_mean']:.3f} (Δ = {gap:+.3f} from best)")
        print(f"\n  CONCLUSION: {'Layer 6 is optimal or near-optimal' if l6 and abs(best['auc_mean'] - l6['auc_mean']) < 0.01 else 'Layer ' + str(best['layer']) + ' outperforms layer 6'}")


if __name__ == "__main__":
    main()

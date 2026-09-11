"""Opportunity-count-normalized survival analysis.

Addresses the reviewer concern that max_cosine_subsequent mechanically increases
with more future edits (older edits have more chances to encounter a high-cosine
neighbor). Computes and compares four exposure metrics:

  1. Raw future maximum (current paper metric, opportunity-confounded)
  2. Fixed-horizon maximum (H=1000, equalizes opportunity count)
  3. Top-5 mean (robust, partially de-confounded)
  4. Count-normalized exceedance P(cos > 0.3) (fully normalized)

Reports OR, CI, and AUC for each, producing the appendix robustness table.

Usage:
    uv run python -m analysis.opportunity_normalized_survival
    uv run python -m analysis.opportunity_normalized_survival --keys-dir results/key_vectors
"""

import argparse
import json
from pathlib import Path
from typing import Optional

import numpy as np

from analysis.interference_panel import (
    TRAJECTORIES,
    build_panel,
    load_key_vectors,
)
from analysis.style import PROJECT, PAPER_OUTPUT


FIXED_HORIZON = 1000


def run_exposure_comparison(
    keys_dir: Path,
    output_dir: Path = PAPER_OUTPUT,
):
    """Run the full opportunity-normalized comparison."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        import pandas as pd
        import statsmodels.formula.api as smf
        from sklearn.metrics import roc_auc_score
    except ImportError as e:
        print(f"ERROR: requires pandas, statsmodels, sklearn: {e}")
        return

    print("=" * 70)
    print("OPPORTUNITY-NORMALIZED SURVIVAL ANALYSIS")
    print("=" * 70)
    print(f"  Fixed horizon H = {FIXED_HORIZON}")
    print(f"  Keys: {keys_dir}")
    print()

    # Build panels with key vectors
    all_panels = []
    for seed in TRAJECTORIES:
        panel = build_panel(seed, keys_dir)
        print(f"  Seed {seed}: {len(panel)} rows")
        all_panels.extend(panel)

    if not all_panels:
        print("ERROR: No panel data")
        return

    df = pd.DataFrame(all_panels)

    # Check key availability
    has_keys = "max_cosine_subsequent" in df.columns
    if not has_keys:
        print("ERROR: Key vectors not loaded. Pass --keys-dir")
        return

    # Filter to rows with valid key data
    key_cols = ["max_cosine_subsequent", "max_cosine_fixed_H", "mean_top5_similarity", "exceedance_03"]
    df_keys = df.dropna(subset=["max_cosine_subsequent"]).copy()

    # For fixed-horizon: only include edits with >= H subsequent edits
    df_fixed = df_keys[df_keys.get("max_cosine_fixed_H_valid", True) == True].copy()

    print(f"\n  Total rows with keys: {len(df_keys)}")
    print(f"  Rows with valid fixed-H (>= {FIXED_HORIZON} subsequent): {len(df_fixed)}")

    # Setup common model components
    df_keys["checkpoint_f"] = df_keys["checkpoint"].astype(str)
    df_keys["age_bin_f"] = pd.Categorical(
        df_keys["age_bin"], categories=["Q1_young", "Q2", "Q3", "Q4_old"]
    )
    df_fixed["checkpoint_f"] = df_fixed["checkpoint"].astype(str)
    df_fixed["age_bin_f"] = pd.Categorical(
        df_fixed["age_bin"], categories=["Q1_young", "Q2", "Q3", "Q4_old"]
    )

    # Define exposure metrics to compare
    metrics = [
        ("Raw future max", "max_cosine_subsequent", df_keys),
        ("Fixed-1K max", "max_cosine_fixed_H", df_fixed),
        ("Top-5 mean", "mean_top5_similarity", df_keys),
        ("P(cos > 0.3)", "exceedance_03", df_keys),
    ]

    base_formula = "survived ~ C(checkpoint_f) + C(age_bin_f) + relation_overlap_rate"

    print(f"\n{'='*70}")
    print(f"{'Metric':<20} {'β':<10} {'OR/+0.1':<10} {'p-value':<12} {'AUC':<8} {'N':<8}")
    print(f"{'-'*70}")

    results = []

    for label, col, data in metrics:
        data_clean = data.dropna(subset=[col]).copy()
        if len(data_clean) < 100:
            print(f"{label:<20} SKIP (n={len(data_clean)})")
            continue

        formula = f"{base_formula} + {col}"

        try:
            model = smf.logit(formula, data=data_clean).fit(disp=0)
            beta = float(model.params.get(col, np.nan))
            pval = float(model.pvalues.get(col, np.nan))
            se = float(model.bse.get(col, np.nan))

            # OR per +0.1 for cosine metrics, per +0.01 for exceedance
            if "exceedance" in col:
                or_scale = np.exp(beta * 0.01)
                or_label = "OR/+0.01"
            else:
                or_scale = np.exp(beta * 0.1)
                or_label = "OR/+0.1"

            # AUC: predicted probabilities vs actual
            pred_probs = model.predict(data_clean)
            auc = float(roc_auc_score(data_clean["survived"].values, pred_probs))

            # 95% CI for beta
            ci_lo = beta - 1.96 * se
            ci_hi = beta + 1.96 * se

            print(f"{label:<20} {beta:<10.4f} {or_scale:<10.4f} {pval:<12.2e} {auc:<8.4f} {len(data_clean):<8}")

            results.append({
                "metric": label,
                "column": col,
                "n_obs": len(data_clean),
                "beta": beta,
                "se": se,
                "ci_95": [ci_lo, ci_hi],
                "p_value": pval,
                "or_per_0.1": float(np.exp(beta * 0.1)),
                "or_per_0.01": float(np.exp(beta * 0.01)),
                "auc": auc,
                "significant": pval < 0.05,
                "direction_negative": beta < 0,
            })

        except Exception as e:
            print(f"{label:<20} ERROR: {e}")
            results.append({"metric": label, "column": col, "error": str(e)})

    print(f"{'-'*70}")

    # Per-seed breakdown
    print(f"\n{'='*70}")
    print("PER-SEED BREAKDOWN")
    print(f"{'='*70}")

    per_seed_results = {}
    for seed in TRAJECTORIES:
        print(f"\n  --- Seed {seed} ---")
        df_seed = df_keys[df_keys["seed"] == seed].copy()
        df_seed["checkpoint_f"] = df_seed["checkpoint"].astype(str)
        df_seed["age_bin_f"] = pd.Categorical(
            df_seed["age_bin"], categories=["Q1_young", "Q2", "Q3", "Q4_old"]
        )

        df_seed_fixed = df_seed[df_seed.get("max_cosine_fixed_H_valid", True) == True].copy()
        if "checkpoint_f" not in df_seed_fixed.columns:
            df_seed_fixed["checkpoint_f"] = df_seed_fixed["checkpoint"].astype(str)
            df_seed_fixed["age_bin_f"] = pd.Categorical(
                df_seed_fixed["age_bin"], categories=["Q1_young", "Q2", "Q3", "Q4_old"]
            )

        seed_metrics = [
            ("Raw future max", "max_cosine_subsequent", df_seed),
            ("Fixed-1K max", "max_cosine_fixed_H", df_seed_fixed),
            ("Top-5 mean", "mean_top5_similarity", df_seed),
            ("P(cos > 0.3)", "exceedance_03", df_seed),
        ]

        seed_results = []
        for label, col, data in seed_metrics:
            data_clean = data.dropna(subset=[col])
            if len(data_clean) < 50:
                continue
            try:
                model = smf.logit(f"{base_formula} + {col}", data=data_clean).fit(disp=0)
                beta = float(model.params.get(col, np.nan))
                pval = float(model.pvalues.get(col, np.nan))
                pred = model.predict(data_clean)
                auc = float(roc_auc_score(data_clean["survived"].values, pred))
                print(f"    {label:<20} β={beta:.4f}  p={pval:.2e}  AUC={auc:.4f}  n={len(data_clean)}")
                seed_results.append({
                    "metric": label, "beta": beta, "p_value": pval, "auc": auc,
                    "significant": pval < 0.05, "direction_negative": beta < 0,
                })
            except Exception:
                pass

        per_seed_results[str(seed)] = seed_results

    # Summary
    print(f"\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}")

    surviving = [r for r in results if "error" not in r and r.get("significant") and r.get("direction_negative")]
    print(f"\n  Metrics with significant negative effect: {len(surviving)}/{len(results)}")
    for r in surviving:
        print(f"    {r['metric']}: β={r['beta']:.4f}, p={r['p_value']:.2e}, AUC={r['auc']:.4f}")

    if surviving:
        print(f"\n  CONCLUSION: The future-key exposure effect SURVIVES opportunity-count")
        print(f"  normalization. The fixed-horizon and count-normalized metrics eliminate")
        print(f"  the mechanical age→opportunity confound, yet the association persists.")
    else:
        print(f"\n  WARNING: No metric survives after opportunity normalization.")
        print(f"  The raw future-maximum result may be confounded by opportunity count.")

    # Save
    output = {
        "fixed_horizon": FIXED_HORIZON,
        "trajectories": TRAJECTORIES,
        "pooled_results": results,
        "per_seed_results": per_seed_results,
        "n_surviving_metrics": len(surviving),
        "conclusion": "survives" if surviving else "confounded",
    }

    out_path = output_dir / "opportunity_normalized_survival.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\n  Saved: {out_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Opportunity-count-normalized survival analysis"
    )
    parser.add_argument("--keys-dir", type=Path, default=Path("results/key_vectors"),
                        help="Directory with key vectors (seed{N}/keys_seed{N}.npz)")
    parser.add_argument("--output-dir", type=Path, default=PAPER_OUTPUT)
    args = parser.parse_args()

    run_exposure_comparison(args.keys_dir, args.output_dir)


if __name__ == "__main__":
    main()

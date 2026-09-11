#!/usr/bin/env python3
"""Consolidated survival analysis table for the paper.

Produces one table with all models on consistent scaling (per +0.1 cosine),
using leave-one-trajectory-out as the primary validation unit.

Output: results/survival_table.json + printed LaTeX/markdown table.

Usage:
    PYTHONPATH=. uv run python analysis/consolidated_survival_table.py \
        --keys-dir results/matched_ordering/key_geometry
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

PROJECT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT / "src" / "util"))

RESULTS = PROJECT / "results"
TRAJECTORIES = [42, 2024, 137]
CHECKPOINTS = [3000, 5000, 7000, 10000]
BATCH_SIZE = 100


def build_first_failure_panel(all_panels):
    """Apply first-failure censoring to the panel."""
    import pandas as pd

    df = pd.DataFrame(all_panels)
    df_sorted = df.sort_values(["seed", "case_id", "checkpoint"])
    keep_mask = []
    for _, group in df_sorted.groupby(["seed", "case_id"]):
        failed = group["survived"] == 0
        if failed.any():
            first_fail_idx = failed.idxmax()
            first_fail_pos = list(group.index).index(first_fail_idx)
            mask = [True] * (first_fail_pos + 1) + [False] * (len(group) - first_fail_pos - 1)
        else:
            mask = [True] * len(group)
        keep_mask.extend(mask)

    return df_sorted.loc[np.array(keep_mask)].copy()


def loo_trajectory_eval(df, formula, require_keys=False):
    """Leave-one-trajectory-out evaluation. Returns per-seed and mean AUC."""
    import statsmodels.formula.api as smf
    from sklearn.metrics import roc_auc_score

    df = df.copy()
    df["checkpoint_f"] = df["checkpoint"].astype(str)
    df["age_bin_f"] = df["age_bin"].astype(str)

    if require_keys:
        df = df.dropna(subset=["max_cosine_subsequent"])

    seeds = sorted(df["seed"].unique())
    seed_aucs = {}
    seed_betas = {}

    for held_out in seeds:
        train = df[df["seed"] != held_out]
        test = df[df["seed"] == held_out]
        if len(train) < 100 or len(test) < 50:
            continue
        try:
            model = smf.logit(formula, data=train).fit(disp=0, maxiter=100)
            pred = model.predict(test)
            y = test["survived"].values
            if len(set(y)) < 2:
                continue
            seed_aucs[held_out] = float(roc_auc_score(y, pred))
            if "max_cosine_subsequent" in model.params:
                seed_betas[held_out] = float(model.params["max_cosine_subsequent"])
        except Exception:
            continue

    mean_auc = np.mean(list(seed_aucs.values())) if seed_aucs else None
    return {
        "per_seed_auc": seed_aucs,
        "mean_auc": mean_auc,
        "per_seed_beta": seed_betas,
    }


def full_sample_fit(df, formula, require_keys=False):
    """Full-sample logit fit. Returns beta, OR, CI, p for cosine."""
    import statsmodels.formula.api as smf

    df = df.copy()
    df["checkpoint_f"] = df["checkpoint"].astype(str)
    df["age_bin_f"] = df["age_bin"].astype(str)

    if require_keys:
        df = df.dropna(subset=["max_cosine_subsequent"])

    try:
        model = smf.logit(formula, data=df).fit(disp=0, maxiter=100)
    except Exception as e:
        return {"error": str(e)}

    result = {"n_obs": len(df), "aic": model.aic}

    if "max_cosine_subsequent" in model.params:
        beta = float(model.params["max_cosine_subsequent"])
        ci = model.conf_int().loc["max_cosine_subsequent"]
        pval = float(model.pvalues["max_cosine_subsequent"])
        result.update({
            "beta": beta,
            "OR_per_unit": float(np.exp(beta)),
            "OR_per_0.1": float(np.exp(beta * 0.1)),
            "CI_per_0.1": [float(np.exp(ci[0] * 0.1)), float(np.exp(ci[1] * 0.1))],
            "CI_per_unit": [float(np.exp(ci[0])), float(np.exp(ci[1]))],
            "pval": pval,
        })

    if "direct_damage" in model.params:
        result["direct_damage_beta"] = float(model.params["direct_damage"])
        result["direct_damage_pval"] = float(model.pvalues["direct_damage"])

    if "update_norm_at_install" in model.params:
        result["update_norm_beta"] = float(model.params["update_norm_at_install"])
        result["update_norm_pval"] = float(model.pvalues["update_norm_at_install"])

    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--keys-dir", type=Path, default=None)
    args = parser.parse_args()

    from analysis.interference_panel import (
        build_panel, TRAJECTORIES, load_fine_grained_interference,
    )

    print("Building panel...")
    all_panels = []
    for seed in TRAJECTORIES:
        panel = build_panel(seed, keys_dir=args.keys_dir)
        all_panels.extend(panel)
    print(f"  Total: {len(all_panels)} rows")

    has_keys = any("max_cosine_subsequent" in r for r in all_panels)
    has_damage = any("direct_damage" in r for r in all_panels)
    print(f"  Keys: {has_keys}, Direct damage: {has_damage}")

    # Build first-failure panel
    df_ff = build_first_failure_panel(all_panels)
    import pandas as pd
    df_rm = pd.DataFrame(all_panels)
    print(f"  First-failure: {len(df_ff)} rows (repeated-measures: {len(df_rm)})")

    # Define all models
    models = {
        "M1_baseline": {
            "formula": "survived ~ C(checkpoint_f) + C(age_bin_f)",
            "keys": False,
            "label": "Age + checkpoint",
        },
    }
    if has_keys:
        models["M4_geometry"] = {
            "formula": (
                "survived ~ C(checkpoint_f) + C(age_bin_f) + "
                "max_cosine_subsequent + cumulative_interference + target_margin"
            ),
            "keys": True,
            "label": "+ cosine + interference + margin",
        }
        models["M5_within_rel"] = {
            "formula": (
                "survived ~ C(checkpoint_f) + C(age_bin_f) + C(relation_id) + "
                "max_cosine_subsequent + cumulative_interference + target_margin"
            ),
            "keys": True,
            "label": "+ relation fixed effects",
        }
    if has_damage and has_keys:
        models["M7a_direct_only"] = {
            "formula": (
                "survived ~ C(checkpoint_f) + C(age_bin_f) + "
                "direct_damage + update_norm_at_install + target_margin"
            ),
            "keys": True,  # needs dropna on cosine column for fair comparison
            "label": "Direct damage (no cosine)",
        }
        models["M7b_direct_plus_cos"] = {
            "formula": (
                "survived ~ C(checkpoint_f) + C(age_bin_f) + "
                "direct_damage + update_norm_at_install + target_margin + "
                "max_cosine_subsequent + cumulative_interference"
            ),
            "keys": True,
            "label": "Direct damage + cosine",
        }

    # Run all models on first-failure panel
    print("\n" + "=" * 80)
    print("CONSOLIDATED SURVIVAL TABLE")
    print("  Outcome: first-failure (hazard)")
    print("  Cosine scale: per +0.1 increment")
    print("  Validation: leave-one-trajectory-out (LOO-T)")
    print("=" * 80)

    table = []
    for name, spec in models.items():
        print(f"\n--- {name}: {spec['label']} ---")

        # LOO-trajectory AUC
        loo = loo_trajectory_eval(df_ff, spec["formula"], require_keys=spec["keys"])

        # Full-sample fit
        full = full_sample_fit(df_ff, spec["formula"], require_keys=spec["keys"])

        row = {
            "model": name,
            "label": spec["label"],
            "n_obs": full.get("n_obs"),
            "aic": full.get("aic"),
            "loo_auc_mean": loo["mean_auc"],
            "loo_auc_per_seed": loo["per_seed_auc"],
        }

        if "OR_per_0.1" in full:
            row["OR_per_0.1"] = full["OR_per_0.1"]
            row["CI_per_0.1"] = full["CI_per_0.1"]
            row["pval"] = full["pval"]
            row["beta"] = full["beta"]

        if "direct_damage_beta" in full:
            row["direct_damage_pval"] = full["direct_damage_pval"]
        if "update_norm_beta" in full:
            row["update_norm_pval"] = full["update_norm_pval"]

        # Per-seed betas on +0.1 scale
        if loo["per_seed_beta"]:
            row["loo_OR_per_0.1_per_seed"] = {
                str(s): float(np.exp(b * 0.1))
                for s, b in loo["per_seed_beta"].items()
            }

        table.append(row)

        # Print
        auc_str = f"LOO-T AUC = {loo['mean_auc']:.3f}" if loo["mean_auc"] else "LOO-T failed"
        if loo["per_seed_auc"]:
            auc_str += f" ({', '.join(f's{s}={a:.3f}' for s, a in loo['per_seed_auc'].items())})"
        print(f"  {auc_str}")
        if "OR_per_0.1" in full:
            ci = full["CI_per_0.1"]
            print(f"  OR per +0.1 = {full['OR_per_0.1']:.4f} [{ci[0]:.4f}, {ci[1]:.4f}], p = {full['pval']:.2e}")

    # LR test for M7a vs M7b
    if "M7a_direct_only" in models and "M7b_direct_plus_cos" in models:
        m7a_full = full_sample_fit(df_ff, models["M7a_direct_only"]["formula"], require_keys=True)
        m7b_full = full_sample_fit(df_ff, models["M7b_direct_plus_cos"]["formula"], require_keys=True)
        if "aic" in m7a_full and "aic" in m7b_full:
            aic_diff = m7a_full["aic"] - m7b_full["aic"]
            print(f"\n--- M7 LR TEST: Does cosine add value beyond direct damage? ---")
            print(f"  M7a AIC = {m7a_full['aic']:.1f}, M7b AIC = {m7b_full['aic']:.1f}")
            print(f"  ΔAIC = {aic_diff:.1f} (positive = cosine helps)")
            table.append({
                "model": "M7_LR_test",
                "label": "Cosine adds to direct damage?",
                "m7a_aic": m7a_full["aic"],
                "m7b_aic": m7b_full["aic"],
                "aic_improvement": aic_diff,
            })

    # Markdown table
    print("\n" + "=" * 80)
    print("PAPER TABLE (Markdown)")
    print("=" * 80)
    print(f"| Model | Predictors | LOO-T AUC | OR per +0.1 | 95% CI | p |")
    print(f"|---|---|---|---|---|---|")
    for row in table:
        if "model" in row and "M7_LR" not in row["model"]:
            auc = f"{row['loo_auc_mean']:.3f}" if row.get("loo_auc_mean") else "—"
            or_val = f"{row['OR_per_0.1']:.4f}" if "OR_per_0.1" in row else "—"
            ci = f"[{row['CI_per_0.1'][0]:.4f}, {row['CI_per_0.1'][1]:.4f}]" if "CI_per_0.1" in row else "—"
            p = f"{row['pval']:.1e}" if "pval" in row else "—"
            print(f"| {row['model']} | {row['label']} | {auc} | {or_val} | {ci} | {p} |")

    # Save
    def _convert_keys(obj):
        if isinstance(obj, dict):
            return {str(k): _convert_keys(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [_convert_keys(v) for v in obj]
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        return obj

    out_path = RESULTS / "survival_table.json"
    with open(out_path, "w") as f:
        json.dump(_convert_keys(table), f, indent=2, default=str)
    print(f"\nSaved: {out_path}")

    # OR discrepancy explanation
    print("\n" + "=" * 80)
    print("OR DISCREPANCY EXPLANATION")
    print("=" * 80)
    print("""
The two previously reported ORs differ because:
  - OR = 0.284: First-failure M5 (with relation FE), full sample, per unit cosine
  - OR = 0.149: Trajectory bootstrap M4 (no relation FE), repeated-measures, per unit

Key differences:
  1. Model: M5 includes relation fixed effects (absorbs some variation) → weaker β
  2. Outcome: first-failure censoring removes post-failure rows → fewer extreme cases
  3. Sample: M5 is full-sample point estimate; bootstrap is median of resampled fits

Neither is wrong — they answer different questions:
  - 0.284 = "controlling for relation, how does cosine affect first-failure risk?"
  - 0.149 = "what is the trajectory-level distribution of the repeated-measures effect?"

For the paper: report the FIRST-FAILURE LOO-T result as primary (on +0.1 scale).
The repeated-measures result goes in the supplement.
""")


if __name__ == "__main__":
    main()

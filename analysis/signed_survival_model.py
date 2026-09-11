#!/usr/bin/env python3
"""Signed-displacement survival models for first-failure prediction.

Integrates signed operator displacement data into the existing survival panel
to test whether directional update-key interactions improve prediction beyond
unsigned cosine exposure.

New models:
  M8:  M4 + cumulative_signed_damage
  M9:  M4 + cancellation_ratio
  M10: cosine + signed_damage + η (all three)

Key hypothesis: signed damage explains why seed 137 doesn't collapse under
high exposure while seeds 42/2024 do.

Usage:
    PYTHONPATH=. uv run python analysis/signed_survival_model.py
    PYTHONPATH=. uv run python analysis/signed_survival_model.py --output results/signed_survival_results.json
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

PROJECT = Path(__file__).resolve().parent.parent
RESULTS = PROJECT / "results"


# ─── Load signed displacement data ──────────────────────────────────────────


def load_signed_data(path: Path = RESULTS / "signed_displacement_analysis.json") -> Optional[Dict]:
    """Load per-edit signed displacement scores.

    Expected format: {
        "per_edit": {
            "<case_id>": {
                "cumulative_signed_damage": float,
                "cumulative_unsigned_damage": float,
                "cancellation_ratio": float,
                "max_destructive_interval": float,
                "seed": int,
                "ordering": str,
            }
        }
    }

    Falls back to computing from fine-grained interference data if the
    pre-computed file doesn't exist.
    """
    if path.exists():
        with open(path) as f:
            return json.load(f)

    print(f"  Signed data not found at {path}")
    print(f"  Attempting to compute from fine-grained interference data...")
    return compute_signed_from_fine_grained()


def compute_signed_from_fine_grained() -> Optional[Dict]:
    """Compute signed displacement from fine_grained.json files.

    The fine-grained interference data has per-batch ||ΔW @ k_i|| (unsigned).
    For signed displacement, we need the actual ΔW @ k_i vector and its
    projection onto the edit's installation direction.

    Currently, fine_grained.json only stores unsigned norms (U_path).
    True signed displacement requires the directional_alignment data
    which has cos(d_i, e_i) — the alignment between later displacement
    and installation direction.

    Strategy: combine U_path (magnitude) with directional alignment (sign)
    to get an approximate signed damage score.
    """
    signed_data = {"per_edit": {}, "metadata": {"source": "fine_grained + directional_alignment"}}

    for seed in [42, 2024]:
        for ordering in ["key_clustered", "key_dispersed"]:
            # Load fine-grained (unsigned magnitude)
            fg_path = RESULTS / "interference" / "AlphaEdit" / ordering / f"seed{seed}" / "fine_grained.json"
            if not fg_path.exists():
                continue

            with open(fg_path) as f:
                fg = json.load(f)

            # Load directional alignment (sign information)
            da_path = RESULTS / "interference" / "AlphaEdit" / ordering / f"seed{seed}" / "directional_alignment.json"
            if not da_path.exists():
                print(f"    No directional_alignment for {ordering}/seed{seed}")
                continue

            with open(da_path) as f:
                da = json.load(f)

            # The directional alignment has per-edit:
            #   alignment = cos(displacement, installation_direction)
            #   damage_score = -<displacement, installation> / ||installation||^2
            # Negative alignment = displacement opposes installation = destructive
            all_5k = fg.get("all_5K", {})
            case_ids = all_5k.get("case_ids", [])
            u_path = all_5k.get("U_path", [])
            i_fro_path = all_5k.get("I_fro_path", [])

            # Directional alignment data (only for first-1K typically)
            da_cases = da.get("case_ids", da.get("first_1k_case_ids", []))
            da_alignments = da.get("alignments", da.get("cosine_alignments", []))
            da_damage = da.get("damage_scores", [])

            # Build case_id → alignment map
            cid_to_align = {}
            for i, cid in enumerate(da_cases):
                if i < len(da_alignments):
                    cid_to_align[int(cid)] = {
                        "alignment": da_alignments[i],
                        "damage_score": da_damage[i] if i < len(da_damage) else 0.0,
                    }

            for i, cid in enumerate(case_ids):
                cid = int(cid)
                unsigned = u_path[i] if i < len(u_path) else 0.0
                fro_norm = i_fro_path[i] if i < len(i_fro_path) else 0.0

                align_info = cid_to_align.get(cid, {})
                alignment = align_info.get("alignment", 0.0)
                damage_score = align_info.get("damage_score", 0.0)

                # Signed damage: unsigned magnitude * sign from alignment
                # alignment < 0 means displacement opposes installation → destructive
                # We define signed_damage as positive = destructive
                signed = unsigned * max(0, -alignment) if alignment != 0 else 0.0

                cancellation = abs(signed) / max(unsigned, 1e-10) if unsigned > 0 else 0.0

                key = f"{cid}_{seed}_{ordering}"
                signed_data["per_edit"][key] = {
                    "case_id": cid,
                    "seed": seed,
                    "ordering": ordering,
                    "cumulative_unsigned_damage": unsigned,
                    "cumulative_signed_damage": signed,
                    "cancellation_ratio": cancellation,
                    "directional_alignment": alignment,
                    "damage_score": damage_score,
                }

    n = len(signed_data["per_edit"])
    print(f"  Computed signed data for {n} edit×seed×ordering combinations")
    return signed_data if n > 0 else None


# ─── Integrate into panel ────────────────────────────────────────────────────


def augment_panel_with_signed(
    panel: List[dict], signed_data: Dict
) -> List[dict]:
    """Add signed displacement columns to existing panel rows."""
    per_edit = signed_data.get("per_edit", {})

    augmented = 0
    for row in panel:
        cid = row["case_id"]
        seed = row["seed"]

        # Try both orderings (panel doesn't know which ordering produced the data)
        for ordering in ["key_clustered", "key_dispersed"]:
            key = f"{cid}_{seed}_{ordering}"
            if key in per_edit:
                entry = per_edit[key]
                row["cumulative_signed_damage"] = entry["cumulative_signed_damage"]
                row["cumulative_unsigned_damage"] = entry["cumulative_unsigned_damage"]
                row["cancellation_ratio"] = entry["cancellation_ratio"]
                row["directional_alignment"] = entry.get("directional_alignment", 0.0)
                augmented += 1
                break

    print(f"  Augmented {augmented}/{len(panel)} panel rows with signed data")
    return panel


# ─── Model fitting ───────────────────────────────────────────────────────────


def fit_signed_models(panel: List[dict], output_path: Path = None):
    """Fit M8-M10 models and compare to M4 baseline."""
    import pandas as pd
    import statsmodels.formula.api as smf
    from sklearn.metrics import roc_auc_score, log_loss

    df = pd.DataFrame(panel)
    df["checkpoint_f"] = df["checkpoint"].astype(str)
    df["age_bin_f"] = pd.Categorical(
        df["age_bin"], categories=["Q1_young", "Q2", "Q3", "Q4_old"]
    )

    # First-failure censoring
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

    df_ff = df_sorted.loc[np.array(keep_mask)].copy()
    print(f"\n  First-failure panel: {len(df_ff)} rows, "
          f"{(df_ff['survived']==0).sum()} events")

    # Check what columns are available
    has_cosine = "max_cosine_subsequent" in df_ff.columns
    has_signed = "cumulative_signed_damage" in df_ff.columns
    has_direct = "direct_damage" in df_ff.columns

    n_signed = df_ff["cumulative_signed_damage"].notna().sum() if has_signed else 0
    print(f"  Available: cosine={has_cosine}, signed={has_signed} (n={n_signed}), direct={has_direct}")

    if n_signed < 100:
        print("  WARNING: Too few rows with signed data for meaningful analysis")

    # Define model formulas
    base = "C(checkpoint_f) + C(age_bin_f)"
    models = {}

    # M4 baseline (cosine)
    if has_cosine:
        models["M4_cosine"] = f"survived ~ {base} + max_cosine_subsequent + cumulative_interference + target_margin"

    # M8: M4 + signed damage
    if has_cosine and has_signed:
        models["M8_cosine_plus_signed"] = (
            f"survived ~ {base} + max_cosine_subsequent + cumulative_interference + "
            f"target_margin + cumulative_signed_damage"
        )

    # M9: M4 + cancellation ratio
    if has_cosine and has_signed:
        models["M9_cosine_plus_cancel"] = (
            f"survived ~ {base} + max_cosine_subsequent + cumulative_interference + "
            f"target_margin + cancellation_ratio"
        )

    # M10: all three (cosine + signed + direct if available)
    if has_cosine and has_signed and has_direct:
        models["M10_all_predictors"] = (
            f"survived ~ {base} + max_cosine_subsequent + cumulative_interference + "
            f"target_margin + cumulative_signed_damage + direct_damage"
        )
    elif has_cosine and has_signed:
        models["M10_cosine_signed"] = (
            f"survived ~ {base} + max_cosine_subsequent + "
            f"target_margin + cumulative_signed_damage"
        )

    # M_signed_only: signed damage without cosine
    if has_signed:
        models["M_signed_only"] = (
            f"survived ~ {base} + cumulative_signed_damage + target_margin"
        )

    # ── Leave-one-trajectory-out evaluation ──
    seeds = sorted(df_ff["seed"].unique())
    results = {"models": {}, "seeds": list(seeds), "n_ff_rows": len(df_ff)}

    print(f"\n  === Leave-One-Trajectory-Out Evaluation ===")
    print(f"  Seeds: {seeds}")

    for name, formula in models.items():
        # Determine which columns are needed
        needed_cols = []
        if "cosine" in formula:
            needed_cols.extend(["max_cosine_subsequent", "cumulative_interference"])
        if "signed" in formula:
            needed_cols.append("cumulative_signed_damage")
        if "cancel" in formula:
            needed_cols.append("cancellation_ratio")
        if "direct" in formula:
            needed_cols.append("direct_damage")
        if "target_margin" in formula:
            needed_cols.append("target_margin")

        df_model = df_ff.dropna(subset=[c for c in needed_cols if c in df_ff.columns]).copy()

        if len(df_model) < 100:
            results["models"][name] = {"error": f"Too few rows ({len(df_model)})"}
            continue

        # LOO-trajectory
        loo_aucs = {}
        loo_logloss = {}
        loo_coefs = {}

        for held_out_seed in seeds:
            train = df_model[df_model["seed"] != held_out_seed]
            test = df_model[df_model["seed"] == held_out_seed]

            if len(train) < 50 or len(test) < 50:
                continue
            if test["survived"].nunique() < 2:
                continue

            try:
                fit = smf.logit(formula, data=train).fit(disp=0, maxiter=100)
                preds = fit.predict(test)
                auc = roc_auc_score(test["survived"], preds)
                ll = log_loss(test["survived"], preds)
                loo_aucs[str(held_out_seed)] = auc
                loo_logloss[str(held_out_seed)] = ll

                # Extract key coefficients
                seed_coefs = {}
                for param in ["max_cosine_subsequent", "cumulative_signed_damage",
                              "cancellation_ratio", "direct_damage"]:
                    if param in fit.params:
                        seed_coefs[param] = {
                            "beta": float(fit.params[param]),
                            "pval": float(fit.pvalues[param]),
                            "OR_per_0.1": float(np.exp(fit.params[param] * 0.1))
                            if "cosine" in param else None,
                        }
                loo_coefs[str(held_out_seed)] = seed_coefs
            except Exception as e:
                loo_aucs[str(held_out_seed)] = None
                loo_logloss[str(held_out_seed)] = None

        valid_aucs = [v for v in loo_aucs.values() if v is not None]
        valid_ll = [v for v in loo_logloss.values() if v is not None]

        model_result = {
            "formula": formula,
            "n_obs": len(df_model),
            "loo_auc_mean": float(np.mean(valid_aucs)) if valid_aucs else None,
            "loo_auc_per_seed": loo_aucs,
            "loo_logloss_mean": float(np.mean(valid_ll)) if valid_ll else None,
            "loo_logloss_per_seed": loo_logloss,
            "loo_coefs": loo_coefs,
        }

        # Full-sample fit for AIC and coefficients
        try:
            full_fit = smf.logit(formula, data=df_model).fit(disp=0, maxiter=100)
            model_result["aic"] = full_fit.aic
            model_result["full_coefs"] = {}
            for param in full_fit.params.index:
                if param.startswith("C("):
                    continue
                model_result["full_coefs"][param] = {
                    "beta": float(full_fit.params[param]),
                    "pval": float(full_fit.pvalues[param]),
                    "se": float(full_fit.bse[param]),
                }
        except Exception as e:
            model_result["full_fit_error"] = str(e)

        results["models"][name] = model_result

        auc_str = f"{model_result['loo_auc_mean']:.3f}" if model_result['loo_auc_mean'] else "N/A"
        ll_str = f"{model_result['loo_logloss_mean']:.4f}" if model_result['loo_logloss_mean'] else "N/A"
        print(f"  {name:30s}: LOO-AUC={auc_str}, LOO-LL={ll_str}, n={len(df_model)}")

    # ── LR tests: does signed add to cosine? ──
    if "M4_cosine" in results["models"] and "M8_cosine_plus_signed" in results["models"]:
        m4_aic = results["models"]["M4_cosine"].get("aic")
        m8_aic = results["models"]["M8_cosine_plus_signed"].get("aic")
        if m4_aic and m8_aic:
            results["lr_test_signed_over_cosine"] = {
                "m4_aic": m4_aic,
                "m8_aic": m8_aic,
                "aic_improvement": m4_aic - m8_aic,
                "interpretation": "Positive = signed improves over cosine alone",
            }
            print(f"\n  LR test (signed over cosine): ΔAIC = {m4_aic - m8_aic:.1f}")

    # ── Seed-137 analysis ──
    if has_signed and 137 in seeds:
        print(f"\n  === Seed-137 Heterogeneity Analysis ===")
        for seed in seeds:
            seed_data = df_ff[df_ff["seed"] == seed]
            n_with_signed = seed_data["cumulative_signed_damage"].notna().sum()
            if n_with_signed > 0:
                signed_vals = seed_data["cumulative_signed_damage"].dropna()
                print(f"  Seed {seed}: n={n_with_signed}, "
                      f"mean_signed={signed_vals.mean():.4f}, "
                      f"std={signed_vals.std():.4f}")

    # Save results
    if output_path is None:
        output_path = RESULTS / "signed_survival_results.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\n  Results saved to {output_path}")

    return results


# ─── Figure outline ──────────────────────────────────────────────────────────

FIGURE_SPEC = """
SIGNED HAZARD FIGURE (the 8-level figure)
==========================================

Panel A: Cumulative signed vs unsigned damage trajectories
  - X-axis: edit count (0 → 10K)
  - Y-axis: cumulative damage per first-1K edit
  - Lines: seeds 42 and 137 under fb_high_exposure
  - Solid lines: unsigned (should be similar — same exposure magnitude)
  - Dashed lines: signed (should diverge — seed 42 accumulates destructive,
    seed 137 has more cancellation)
  - Key message: same unsigned exposure, different signed consequences

Panel B: Signed damage predicts seed-level collapse
  - Scatter: x = mean cumulative signed damage, y = overall efficacy at 10K
  - Points: 6 conditions (3 seeds × high/low) or all 18 (3 seeds × 6 orderings)
  - Correlation should be strong and negative
  - Seed 137 fb_high should have LOWER signed damage than seeds 42/2024 fb_high
    despite similar unsigned exposure
  - Key message: signed, not unsigned, predicts collapse

Panel C: Signed hazard improves first-failure AUC
  - Nested bar chart:
    M1 (baseline): ~0.84
    M4 (+ cosine): ~0.92
    M8 (+ cosine + signed): hopefully ~0.93+
    M_signed_only: show that signed alone approaches cosine performance
  - Per-seed error bars from LOO-trajectory
  - Key message: directional information adds predictive value
"""


# ─── Main ────────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(description="Signed-displacement survival models")
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--keys-dir", type=str, default=None)
    parser.add_argument("--signed-data", type=str, default=None)
    args = parser.parse_args()

    sys.path.insert(0, str(PROJECT / "src" / "util"))
    from analysis.interference_panel import (
        build_panel, TRAJECTORIES, load_key_vectors
    )

    # Build the panel
    print("Building survival panel...")
    keys_dir = Path(args.keys_dir) if args.keys_dir else PROJECT / "results" / "matched_ordering" / "key_geometry"
    all_panels = []
    for seed in TRAJECTORIES:
        panel = build_panel(seed, keys_dir=keys_dir)
        all_panels.extend(panel)
    print(f"  Panel: {len(all_panels)} rows across {len(TRAJECTORIES)} seeds")

    # Load signed data
    print("\nLoading signed displacement data...")
    signed_path = Path(args.signed_data) if args.signed_data else None
    signed_data = load_signed_data(signed_path) if signed_path else load_signed_data()

    if signed_data is None:
        print("  No signed data available. Running with unsigned predictors only.")
        print("  To add signed data, run the directional alignment analysis first.")
    else:
        all_panels = augment_panel_with_signed(all_panels, signed_data)

    # Fit models
    output_path = Path(args.output) if args.output else None
    results = fit_signed_models(all_panels, output_path)

    # Print figure spec
    print(f"\n{FIGURE_SPEC}")


if __name__ == "__main__":
    main()

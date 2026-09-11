#!/usr/bin/env python3
"""Mediation analysis: does historical-key displacement mediate
the effect of cosine exposure on behavioral damage?

Uses the existing A/B logit-damage trial data (no new GPU runs).

Mediation chain:
  cosine exposure → historical displacement → logprob damage → first failure

Tests:
  Step 1: exposure → displacement
  Step 2: displacement → damage
  Step 3: displacement mediates exposure→damage (attenuation)
  Step 4: seed-blocked robustness
  Step 5: margin-susceptibility risk score
  CLaRE 2×2 contingency (McNemar)
"""

import json
import os
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def load_different_batch_trials():
    """Load 30 different-batch A/B trials (3 seeds × 10 trials)."""
    trials = []
    for seed in [42, 2024, 137]:
        path = PROJECT_ROOT / f"results/logit_damage/seed{seed}/intervention_results.json"
        if not path.exists():
            continue
        d = json.load(open(path))
        baseline_lp = d.get("baseline_mean_logprob", None)
        for t in d["trials"]:
            trials.append({
                "seed": seed,
                "trial": t["trial"],
                "type": "different_batch",
                "high_cosine": t["high_cosine"],
                "low_cosine": t["low_cosine"],
                "delta_cosine": t["high_cosine"] - t["low_cosine"],
                "D_hist_high": float(np.mean(t["high_damage"])),
                "D_hist_low": float(np.mean(t["low_damage"])),
                "delta_D": float(np.mean(t["high_damage"]) - np.mean(t["low_damage"])),
                "L_high": float(np.mean(t["high_delta_logprob"])),
                "L_low": float(np.mean(t["low_delta_logprob"])),
                "delta_L": float(np.mean(t["high_delta_logprob"]) - np.mean(t["low_delta_logprob"])),
                "baseline_lp": baseline_lp,
                "high_damage_per_edit": t["high_damage"],
                "low_damage_per_edit": t["low_damage"],
                "high_lp_per_edit": t["high_delta_logprob"],
                "low_lp_per_edit": t["low_delta_logprob"],
            })
    return trials


def load_same_fact_trials():
    """Load same-fact trials (no displacement data, only logprob)."""
    trials = []
    for seed in [42, 2024, 137]:
        path = PROJECT_ROOT / f"results/same_fact_damage/seed{seed}/intervention_results.json"
        if not path.exists():
            continue
        d = json.load(open(path))
        for t in d["trials"]:
            trials.append({
                "seed": seed,
                "trial": t["trial"],
                "type": "same_fact",
                "high_cosine": t.get("high_mean_cosine", 0),
                "low_cosine": t.get("low_mean_cosine", 0),
                "delta_cosine": t.get("cosine_gap", 0),
                "delta_L": float(np.mean(t["high_delta_logprob"]) - np.mean(t["low_delta_logprob"])),
                "L_high": float(np.mean(t["high_delta_logprob"])),
                "L_low": float(np.mean(t["low_delta_logprob"])),
            })
    return trials


def spearman(x, y):
    from scipy.stats import spearmanr
    r, p = spearmanr(x, y)
    return float(r), float(p)


def pearson(x, y):
    from scipy.stats import pearsonr
    r, p = pearsonr(x, y)
    return float(r), float(p)


def ols_summary(y, X_dict, names=None):
    """Simple OLS with named coefficients."""
    import statsmodels.api as sm
    X = np.column_stack(list(X_dict.values()))
    X = sm.add_constant(X)
    model = sm.OLS(y, X).fit()
    result = {"r_squared": float(model.rsquared), "f_pvalue": float(model.f_pvalue)}
    col_names = ["const"] + list(X_dict.keys())
    for i, name in enumerate(col_names):
        result[f"{name}_coef"] = float(model.params[i])
        result[f"{name}_pval"] = float(model.pvalues[i])
    return result


def main():
    diff_trials = load_different_batch_trials()
    sf_trials = load_same_fact_trials()

    print(f"Loaded {len(diff_trials)} different-batch + {len(sf_trials)} same-fact trials")

    results = {}

    # ── Step 1: Does exposure predict displacement? ──────────────────
    print("\n=== STEP 1: Exposure → Displacement ===")
    delta_cos = np.array([t["delta_cosine"] for t in diff_trials])
    delta_D = np.array([t["delta_D"] for t in diff_trials])
    delta_L = np.array([t["delta_L"] for t in diff_trials])
    seeds = np.array([t["seed"] for t in diff_trials])

    r_cos_D, p_cos_D = spearman(delta_cos, delta_D)
    print(f"  Δcosine → ΔD: Spearman r={r_cos_D:.3f}, p={p_cos_D:.4f}")
    print(f"  Sign: {sum(delta_D > 0)}/30 trials have ΔD > 0 (HIGH more displacement)")

    results["step1_exposure_to_displacement"] = {
        "spearman_r": r_cos_D, "p": p_cos_D,
        "n_positive_delta_D": int(sum(delta_D > 0)),
        "mean_delta_D": float(np.mean(delta_D)),
    }

    # ── Step 2: Does displacement predict damage? ─────────────────────
    print("\n=== STEP 2: Displacement → Damage ===")
    r_D_L, p_D_L = spearman(delta_D, delta_L)
    r_D_L_pear, p_D_L_pear = pearson(delta_D, delta_L)
    print(f"  ΔD → ΔL: Spearman r={r_D_L:.3f}, p={p_D_L:.4f}")
    print(f"  ΔD → ΔL: Pearson  r={r_D_L_pear:.3f}, p={p_D_L_pear:.4f}")

    results["step2_displacement_to_damage"] = {
        "spearman_r": r_D_L, "spearman_p": p_D_L,
        "pearson_r": r_D_L_pear, "pearson_p": p_D_L_pear,
    }

    # ── Step 3: Mediation — does displacement attenuate cosine? ──────
    print("\n=== STEP 3: Mediation (cosine attenuates after controlling for D) ===")

    seed_dummies = np.column_stack([
        (seeds == 2024).astype(float),
        (seeds == 137).astype(float),
    ])

    # Model A: total effect of cosine
    model_a = ols_summary(delta_L, {
        "delta_cosine": delta_cos,
        "seed_2024": seed_dummies[:, 0],
        "seed_137": seed_dummies[:, 1],
    })

    # Model B: cosine + displacement
    model_b = ols_summary(delta_L, {
        "delta_cosine": delta_cos,
        "delta_D": delta_D,
        "seed_2024": seed_dummies[:, 0],
        "seed_137": seed_dummies[:, 1],
    })

    cos_total = model_a["delta_cosine_coef"]
    cos_after_D = model_b["delta_cosine_coef"]
    attenuation = 1 - abs(cos_after_D) / max(abs(cos_total), 1e-10) if cos_total != 0 else 0

    print(f"  Model A (cosine only): cosine coef={cos_total:.4f}, p={model_a['delta_cosine_pval']:.4f}, R²={model_a['r_squared']:.3f}")
    print(f"  Model B (cosine + D):  cosine coef={cos_after_D:.4f}, p={model_b['delta_cosine_pval']:.4f}, R²={model_b['r_squared']:.3f}")
    print(f"  Displacement coef: {model_b['delta_D_coef']:.4f}, p={model_b['delta_D_pval']:.4f}")
    print(f"  Attenuation of cosine: {attenuation:.1%}")

    results["step3_mediation"] = {
        "model_a_cosine_only": model_a,
        "model_b_cosine_plus_D": model_b,
        "cosine_attenuation_fraction": float(attenuation),
        "interpretation": (
            "Positive attenuation means displacement mediates part of cosine's effect. "
            "If cosine becomes non-significant in Model B, displacement fully mediates."
        ),
    }

    # ── Step 4: Seed-blocked analysis ─────────────────────────────────
    print("\n=== STEP 4: Seed-specific effects ===")
    seed_results = {}
    for seed in [42, 2024, 137]:
        mask = seeds == seed
        if mask.sum() < 3:
            continue
        dD = delta_D[mask]
        dL = delta_L[mask]
        dC = delta_cos[mask]

        r_DL, p_DL = spearman(dD, dL)
        r_CL, p_CL = spearman(dC, dL)
        r_CD, p_CD = spearman(dC, dD)

        seed_results[str(seed)] = {
            "n": int(mask.sum()),
            "mean_delta_D": float(np.mean(dD)),
            "mean_delta_L": float(np.mean(dL)),
            "D_to_L_spearman": {"r": r_DL, "p": p_DL},
            "cos_to_L_spearman": {"r": r_CL, "p": p_CL},
            "cos_to_D_spearman": {"r": r_CD, "p": p_CD},
            "sign_D_positive": int(sum(dD > 0)),
            "sign_L_negative": int(sum(dL < 0)),
        }
        print(f"  Seed {seed}: ΔD→ΔL r={r_DL:.3f} (p={p_DL:.3f}), "
              f"cos→ΔL r={r_CL:.3f}, cos→ΔD r={r_CD:.3f}, "
              f"ΔD>0: {sum(dD>0)}/{mask.sum()}, ΔL<0: {sum(dL<0)}/{mask.sum()}")

    # Leave-one-seed-out
    print("\n  Leave-one-seed-out (D→L correlation on held-out seed):")
    loo_results = {}
    for held_out in [42, 2024, 137]:
        train_mask = seeds != held_out
        test_mask = seeds == held_out
        if test_mask.sum() < 3:
            continue
        # Simple: just report held-out correlation
        r_test, p_test = spearman(delta_D[test_mask], delta_L[test_mask])
        loo_results[str(held_out)] = {"r": r_test, "p": p_test}
        print(f"    Hold out seed {held_out}: r={r_test:.3f}, p={p_test:.3f}")

    results["step4_seed_analysis"] = {
        "per_seed": seed_results,
        "leave_one_out": loo_results,
    }

    # ── Step 5: Margin susceptibility ─────────────────────────────────
    print("\n=== STEP 5: Displacement × margin susceptibility ===")

    # Per-edit analysis within the different-batch trials
    # For each focal edit: does displacement/margin predict failure better than displacement alone?
    all_D_high = []
    all_D_low = []
    all_L_high = []
    all_L_low = []

    for t in diff_trials:
        all_D_high.extend(t["high_damage_per_edit"])
        all_D_low.extend(t["low_damage_per_edit"])
        all_L_high.extend(t["high_lp_per_edit"])
        all_L_low.extend(t["low_lp_per_edit"])

    all_D_high = np.array(all_D_high)
    all_D_low = np.array(all_D_low)
    all_L_high = np.array(all_L_high)
    all_L_low = np.array(all_L_low)

    # Per-edit paired difference
    edit_delta_D = all_D_high - all_D_low
    edit_delta_L = all_L_high - all_L_low

    r_edit, p_edit = spearman(edit_delta_D[:10000], edit_delta_L[:10000])
    r_edit_p, p_edit_p = pearson(edit_delta_D[:10000], edit_delta_L[:10000])

    print(f"  Per-edit (n={len(edit_delta_D)}): ΔD→ΔL Spearman r={r_edit:.3f}, p={p_edit:.6f}")
    print(f"  Per-edit: ΔD→ΔL Pearson  r={r_edit_p:.3f}, p={p_edit_p:.6f}")

    # Margin-based risk: use baseline logprob as margin proxy
    # For each trial, we have baseline_lp (mean across focal edits)
    # Higher absolute baseline_lp = stronger installed edit = more margin
    # Risk = displacement / |margin|
    margins = []
    for t in diff_trials:
        if t["baseline_lp"] is not None:
            margins.append(abs(t["baseline_lp"]))

    if margins:
        margins = np.array(margins[:len(delta_D)])
        risk_high = np.array([t["D_hist_high"] for t in diff_trials]) / np.maximum(margins, 1e-10)
        risk_low = np.array([t["D_hist_low"] for t in diff_trials]) / np.maximum(margins, 1e-10)
        delta_risk = risk_high - risk_low
        r_risk, p_risk = spearman(delta_risk, delta_L)
        print(f"  Risk (D/margin) → ΔL: Spearman r={r_risk:.3f}, p={p_risk:.4f}")
        results["step5_margin_susceptibility"] = {
            "edit_level_D_to_L_spearman": {"r": r_edit, "p": p_edit},
            "edit_level_D_to_L_pearson": {"r": r_edit_p, "p": p_edit_p},
            "n_edits": len(edit_delta_D),
            "risk_to_L_spearman": {"r": r_risk, "p": p_risk},
        }
    else:
        results["step5_margin_susceptibility"] = {
            "edit_level_D_to_L_spearman": {"r": r_edit, "p": p_edit},
            "edit_level_D_to_L_pearson": {"r": r_edit_p, "p": p_edit_p},
            "n_edits": len(edit_delta_D),
        }

    # ── CLaRE 2×2 contingency ────────────────────────────────────────
    print("\n=== CLaRE 2×2 Contingency ===")
    clare_path = PROJECT_ROOT / "results/clare_comparison.json"
    if clare_path.exists():
        clare = json.load(open(clare_path))
        # Extract per-trial correctness for max_cosine and CLaRE
        branch_ranking = clare.get("branch_ranking", {})
        max_cos_correct = branch_ranking.get("max_cosine_per_trial_correct", [])
        clare_correct = branch_ranking.get("clare_mean_per_trial_correct", [])

        if max_cos_correct and clare_correct:
            n = min(len(max_cos_correct), len(clare_correct))
            mc = np.array(max_cos_correct[:n])
            cl = np.array(clare_correct[:n])

            # 2×2: both right, max only, clare only, both wrong
            both_right = int(((mc == 1) & (cl == 1)).sum())
            max_only = int(((mc == 1) & (cl == 0)).sum())
            clare_only = int(((mc == 0) & (cl == 1)).sum())
            both_wrong = int(((mc == 0) & (cl == 0)).sum())

            print(f"  Both right: {both_right}")
            print(f"  Max-cosine only: {max_only}")
            print(f"  CLaRE only: {clare_only}")
            print(f"  Both wrong: {both_wrong}")

            # McNemar's test on discordant pairs
            from scipy.stats import binomtest
            discordant = max_only + clare_only
            if discordant > 0:
                mcnemar = binomtest(max_only, discordant, 0.5)
                print(f"  McNemar p={mcnemar.pvalue:.4f} (discordant: {max_only} vs {clare_only})")
                results["clare_contingency"] = {
                    "both_right": both_right, "max_only": max_only,
                    "clare_only": clare_only, "both_wrong": both_wrong,
                    "mcnemar_p": float(mcnemar.pvalue),
                }
            else:
                print("  No discordant pairs — identical predictions")
                results["clare_contingency"] = {
                    "both_right": both_right, "max_only": max_only,
                    "clare_only": clare_only, "both_wrong": both_wrong,
                    "note": "identical predictions",
                }
        else:
            print("  Per-trial correctness not available in CLaRE results")
            results["clare_contingency"] = {"note": "per-trial data not available"}
    else:
        print("  CLaRE comparison not found")
        results["clare_contingency"] = {"note": "file not found"}

    # ── Summary ──────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("MEDIATION SUMMARY")
    print("=" * 60)

    chain_holds = (
        results["step1_exposure_to_displacement"]["n_positive_delta_D"] > 15
        and r_D_L < -0.2
        and attenuation > 0.1
    )

    print(f"""
  Chain: exposure → displacement → damage

  Step 1 (exposure → displacement):
    {sum(delta_D > 0)}/30 trials: HIGH has more displacement
    Spearman r = {r_cos_D:.3f}, p = {p_cos_D:.4f}

  Step 2 (displacement → damage):
    Spearman r = {r_D_L:.3f}, p = {p_D_L:.4f}
    Pearson  r = {r_D_L_pear:.3f}, p = {p_D_L_pear:.4f}

  Step 3 (mediation):
    Cosine effect (total):     coef = {cos_total:.4f}, p = {model_a['delta_cosine_pval']:.4f}
    Cosine effect (after D):   coef = {cos_after_D:.4f}, p = {model_b['delta_cosine_pval']:.4f}
    Displacement effect:       coef = {model_b['delta_D_coef']:.4f}, p = {model_b['delta_D_pval']:.4f}
    Attenuation: {attenuation:.1%}

  Step 5 (per-edit):
    Edit-level ΔD → ΔL: r = {r_edit:.3f} (n = {len(edit_delta_D)})

  Chain supported: {'YES' if chain_holds else 'PARTIALLY — check details'}
""")

    results["summary"] = {
        "chain_supported": chain_holds,
        "exposure_to_displacement_sign": int(sum(delta_D > 0)),
        "displacement_to_damage_r": r_D_L,
        "cosine_attenuation": float(attenuation),
    }

    # Save
    out_path = PROJECT_ROOT / "results/mediation_results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()

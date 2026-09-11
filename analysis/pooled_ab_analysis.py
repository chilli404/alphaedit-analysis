#!/usr/bin/env python3
"""Pooled trial-level analysis combining different-batch and same-fact A/B interventions.

Pools at the TRIAL level (one observation per trial pair, not per edit).
Tests whether logit damage scales with achieved cosine contrast and whether
the same-fact indicator explains away the effect.

Usage:
    uv run python analysis/pooled_ab_analysis.py
"""

import json
import sys
from pathlib import Path

import numpy as np
from scipy import stats as sp_stats

PROJECT = Path(__file__).resolve().parent.parent
RESULTS = PROJECT / "results"


def load_trials():
    """Load all trials from both experiment types into a flat list."""
    records = []

    # Different-batch trials
    for seed in [42, 2024, 137]:
        path = RESULTS / "logit_damage" / f"seed{seed}" / "intervention_results.json"
        if not path.exists():
            continue
        data = json.load(open(path))
        for t in data["trials"]:
            hi_mean = np.mean(t["high_delta_logprob"])
            lo_mean = np.mean(t["low_delta_logprob"])
            records.append({
                "seed": seed,
                "trial": t["trial"],
                "experiment": "different_batch",
                "delta_logprob": hi_mean - lo_mean,
                "cosine_gap": t["high_cosine"] - t["low_cosine"],
                "cos_high": t["high_cosine"],
                "cos_low": t["low_cosine"],
                "frac_hi_worse": np.mean([p < 0 for p in t["paired_logprob_diff"]]),
            })

    # Same-fact trials
    for seed in [42, 2024, 137]:
        path = RESULTS / "same_fact_damage" / f"seed{seed}" / "intervention_results.json"
        if not path.exists():
            continue
        data = json.load(open(path))
        for t in data["trials"]:
            hi_mean = np.mean(t["high_delta_logprob"])
            lo_mean = np.mean(t["low_delta_logprob"])
            records.append({
                "seed": seed,
                "trial": t["trial"],
                "experiment": "same_fact",
                "delta_logprob": hi_mean - lo_mean,
                "cosine_gap": t["cosine_gap"],
                "cos_high": t["high_mean_cosine"],
                "cos_low": t["low_mean_cosine"],
                "frac_hi_worse": np.mean([p < 0 for p in t["paired_logprob_diff"]]),
            })

    return records


def trial_level_summary(records):
    """Basic summary statistics at the trial level."""
    diff_batch = [r for r in records if r["experiment"] == "different_batch"]
    same_fact = [r for r in records if r["experiment"] == "same_fact"]

    summary = {}

    for label, subset in [("different_batch", diff_batch), ("same_fact", same_fact), ("pooled", records)]:
        deltas = [r["delta_logprob"] for r in subset]
        n_neg = sum(1 for d in deltas if d < 0)
        n_total = len(deltas)

        sign_test_p = sp_stats.binomtest(n_neg, n_total, 0.5).pvalue if n_total > 0 else 1.0

        summary[label] = {
            "n_trials": n_total,
            "n_seeds": len(set(r["seed"] for r in subset)),
            "mean_delta": float(np.mean(deltas)) if deltas else 0,
            "std_delta": float(np.std(deltas, ddof=1)) if len(deltas) > 1 else 0,
            "median_delta": float(np.median(deltas)) if deltas else 0,
            "sign_count": f"{n_neg}/{n_total}",
            "sign_test_p": float(sign_test_p),
            "mean_cosine_gap": float(np.mean([r["cosine_gap"] for r in subset])) if subset else 0,
        }

    return summary


def seed_specific_effects(records):
    """Per-seed mean effects with CIs."""
    results = {}
    for seed in sorted(set(r["seed"] for r in records)):
        for exp in ["different_batch", "same_fact"]:
            subset = [r for r in records if r["seed"] == seed and r["experiment"] == exp]
            if not subset:
                continue
            deltas = [r["delta_logprob"] for r in subset]
            n_neg = sum(1 for d in deltas if d < 0)
            mean = np.mean(deltas)
            se = np.std(deltas, ddof=1) / np.sqrt(len(deltas)) if len(deltas) > 1 else 0
            results[f"{exp}_seed{seed}"] = {
                "n": len(deltas),
                "mean": float(mean),
                "se": float(se),
                "ci_lo": float(mean - 1.96 * se),
                "ci_hi": float(mean + 1.96 * se),
                "sign": f"{n_neg}/{len(deltas)}",
            }
    return results


def cosine_dose_response(records):
    """Test whether damage scales with achieved cosine contrast."""
    gaps = np.array([r["cosine_gap"] for r in records])
    deltas = np.array([r["delta_logprob"] for r in records])

    spearman_r, spearman_p = sp_stats.spearmanr(gaps, deltas)
    pearson_r, pearson_p = sp_stats.pearsonr(gaps, deltas)

    # OLS: delta = a + b * gap
    slope, intercept, r_val, p_val, se = sp_stats.linregress(gaps, deltas)

    return {
        "spearman_r": float(spearman_r),
        "spearman_p": float(spearman_p),
        "pearson_r": float(pearson_r),
        "pearson_p": float(pearson_p),
        "ols_slope": float(slope),
        "ols_intercept": float(intercept),
        "ols_r2": float(r_val**2),
        "ols_p": float(p_val),
        "ols_se": float(se),
    }


def regression_with_experiment_type(records):
    """Test: δlogprob ~ Δcosine + 𝟙[same_fact] + seed FE.

    Does the same-fact indicator explain away the cosine effect?
    """
    try:
        import statsmodels.formula.api as smf
        import pandas as pd
    except ImportError:
        return {"error": "statsmodels not available"}

    df = pd.DataFrame(records)
    df["is_same_fact"] = (df["experiment"] == "same_fact").astype(int)
    df["seed_f"] = df["seed"].astype(str)

    # Model 1: cosine gap only
    m1 = smf.ols("delta_logprob ~ cosine_gap", data=df).fit()

    # Model 2: cosine gap + experiment type
    m2 = smf.ols("delta_logprob ~ cosine_gap + is_same_fact", data=df).fit()

    # Model 3: cosine gap + experiment type + seed FE
    m3 = smf.ols("delta_logprob ~ cosine_gap + is_same_fact + C(seed_f)", data=df).fit()

    # Model 4: interaction — does cosine effect differ by experiment type?
    m4 = smf.ols("delta_logprob ~ cosine_gap * is_same_fact + C(seed_f)", data=df).fit()

    results = {}
    for name, m in [("m1_cosine_only", m1), ("m2_plus_type", m2),
                     ("m3_plus_seed_fe", m3), ("m4_interaction", m4)]:
        results[name] = {
            "r2": float(m.rsquared),
            "aic": float(m.aic),
            "formula": str(m.model.formula),
        }
        for param in m.params.index:
            if param != "Intercept" and not param.startswith("C("):
                results[name][f"coef_{param}"] = float(m.params[param])
                results[name][f"pval_{param}"] = float(m.pvalues[param])
                results[name][f"se_{param}"] = float(m.bse[param])

    # Key question: is cosine_gap significant after controlling for experiment type?
    results["cosine_significant_after_type_control"] = float(m3.pvalues.get("cosine_gap", 1.0)) < 0.05
    results["same_fact_indicator_significant"] = float(m3.pvalues.get("is_same_fact", 1.0)) < 0.05

    return results


def hierarchical_bootstrap(records, n_boot=5000, rng_seed=42):
    """Resample seeds, then trials within seed. Reports pooled mean effect."""
    rng = np.random.default_rng(rng_seed)
    seeds = sorted(set(r["seed"] for r in records))
    by_seed = {s: [r["delta_logprob"] for r in records if r["seed"] == s] for s in seeds}

    boot_means = []
    for _ in range(n_boot):
        sampled_seeds = rng.choice(seeds, size=len(seeds), replace=True)
        boot_deltas = []
        for s in sampled_seeds:
            trials = by_seed[s]
            boot_deltas.extend(rng.choice(trials, size=len(trials), replace=True))
        boot_means.append(np.mean(boot_deltas))

    boot_means = np.array(boot_means)
    ci_lo, ci_hi = np.percentile(boot_means, [2.5, 97.5])
    p_neg = np.mean(boot_means < 0)

    return {
        "n_boot": n_boot,
        "mean": float(np.mean(boot_means)),
        "ci_lo": float(ci_lo),
        "ci_hi": float(ci_hi),
        "p_negative": float(p_neg),
        "se": float(np.std(boot_means)),
    }


def make_plot_data(records):
    """Data for dose-response scatter: cosine gap vs damage, colored by type."""
    return [
        {
            "cosine_gap": r["cosine_gap"],
            "delta_logprob": r["delta_logprob"],
            "experiment": r["experiment"],
            "seed": r["seed"],
        }
        for r in records
    ]


def main():
    records = load_trials()

    n_diff = sum(1 for r in records if r["experiment"] == "different_batch")
    n_same = sum(1 for r in records if r["experiment"] == "same_fact")
    print(f"Loaded {len(records)} trials: {n_diff} different-batch, {n_same} same-fact")

    print("\n=== TRIAL-LEVEL SUMMARY ===")
    summary = trial_level_summary(records)
    for label, s in summary.items():
        print(f"  {label}: n={s['n_trials']}, mean={s['mean_delta']:.4f}, "
              f"sign={s['sign_count']}, p_sign={s['sign_test_p']:.4f}, "
              f"mean_cos_gap={s['mean_cosine_gap']:.4f}")

    print("\n=== SEED-SPECIFIC EFFECTS ===")
    seed_effects = seed_specific_effects(records)
    for k, v in seed_effects.items():
        print(f"  {k}: mean={v['mean']:.4f} [{v['ci_lo']:.4f}, {v['ci_hi']:.4f}], sign={v['sign']}")

    print("\n=== COSINE DOSE-RESPONSE ===")
    dose = cosine_dose_response(records)
    print(f"  Spearman r={dose['spearman_r']:.3f}, p={dose['spearman_p']:.4f}")
    print(f"  OLS: delta = {dose['ols_intercept']:.4f} + {dose['ols_slope']:.4f} * gap, "
          f"R²={dose['ols_r2']:.3f}, p={dose['ols_p']:.4f}")

    print("\n=== REGRESSION WITH EXPERIMENT TYPE ===")
    reg = regression_with_experiment_type(records)
    for name in ["m1_cosine_only", "m2_plus_type", "m3_plus_seed_fe", "m4_interaction"]:
        if name in reg:
            m = reg[name]
            print(f"  {name} (R²={m['r2']:.3f}, AIC={m['aic']:.1f}):")
            for k, v in m.items():
                if k.startswith("coef_"):
                    param = k.replace("coef_", "")
                    pval = m.get(f"pval_{param}", "?")
                    print(f"    {param}: β={v:.4f}, p={pval:.4f}")

    if "cosine_significant_after_type_control" in reg:
        print(f"\n  Cosine significant after type control: {reg['cosine_significant_after_type_control']}")
        print(f"  Same-fact indicator significant: {reg['same_fact_indicator_significant']}")

    print("\n=== HIERARCHICAL BOOTSTRAP (pooled) ===")
    boot = hierarchical_bootstrap(records)
    print(f"  Mean={boot['mean']:.4f}, 95% CI=[{boot['ci_lo']:.4f}, {boot['ci_hi']:.4f}]")
    print(f"  P(negative)={boot['p_negative']:.4f}")

    # Save
    output = {
        "n_trials": len(records),
        "n_different_batch": n_diff,
        "n_same_fact": n_same,
        "summary": summary,
        "seed_effects": seed_effects,
        "dose_response": dose,
        "regression": reg,
        "hierarchical_bootstrap": boot,
        "plot_data": make_plot_data(records),
    }

    out_path = RESULTS / "pooled_ab_results.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()

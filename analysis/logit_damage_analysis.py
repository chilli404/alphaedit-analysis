#!/usr/bin/env python3
"""Corrected statistical analysis of the logit-damage A/B intervention.

Addresses reviewer concerns:
  1. Treatment unit is the batch pair, not the 1000 focal edits.
  2. 30 trials are nested in 3 seed/base states.
  3. Batch-property balance between HIGH and LOW must be assessed.

Primary inference: trial-level paired effects (n=30) with seed as a
blocking variable. Hierarchical bootstrap resamples seeds then trials
within seed to properly account for the nested structure.

Usage:
    uv run python analysis/logit_damage_analysis.py
    PYTHONPATH=. uv run python analysis/logit_damage_analysis.py
"""

import json
import os
import sys
from pathlib import Path

import numpy as np

SEEDS = [42, 2024, 137]
RESULTS_BASE = Path(__file__).resolve().parent.parent / "results" / "logit_damage"


def load_all_seeds():
    """Load intervention results for all seeds."""
    all_data = {}
    for seed in SEEDS:
        path = RESULTS_BASE / f"seed{seed}" / "intervention_results.json"
        if not path.exists():
            print(f"  WARNING: {path} not found, skipping seed {seed}")
            continue
        with open(path) as f:
            all_data[seed] = json.load(f)
    return all_data


def compute_trial_effects(all_data):
    """Extract trial-level paired effects.

    Each trial yields one observation: the mean logprob difference
    across 1000 focal edits (HIGH - LOW). Negative = HIGH more damaging.
    """
    records = []
    for seed, data in all_data.items():
        for t in data["trials"]:
            hi_mean = np.mean(t["high_delta_logprob"])
            lo_mean = np.mean(t["low_delta_logprob"])
            hi_dmg = np.mean(t["high_damage"])
            lo_dmg = np.mean(t["low_damage"])
            frac_hi_worse = np.mean([p < 0 for p in t["paired_logprob_diff"]])

            records.append({
                "seed": seed,
                "trial": t["trial"],
                "cos_high": t["high_cosine"],
                "cos_low": t["low_cosine"],
                "cos_gap": t["high_cosine"] - t["low_cosine"],
                "delta_lp_high": hi_mean,
                "delta_lp_low": lo_mean,
                "paired_delta_lp": hi_mean - lo_mean,
                "damage_high": hi_dmg,
                "damage_low": lo_dmg,
                "paired_damage": hi_dmg - lo_dmg,
                "frac_hi_worse": frac_hi_worse,
                "n_hi_relations": len(t["high_relations"]),
                "n_lo_relations": len(t["low_relations"]),
            })
    return records


def batch_balance_diagnostics(all_data):
    """Assess confound balance between HIGH and LOW batches."""
    diagnostics = []
    for seed, data in all_data.items():
        for t in data["trials"]:
            hi_rels = t["high_relations"]
            lo_rels = t["low_relations"]
            all_rels = set(hi_rels.keys()) | set(lo_rels.keys())
            overlap_rels = set(hi_rels.keys()) & set(lo_rels.keys())

            jaccard = len(overlap_rels) / len(all_rels) if all_rels else 0

            hi_counts = np.array([hi_rels.get(r, 0) for r in all_rels])
            lo_counts = np.array([lo_rels.get(r, 0) for r in all_rels])
            hi_dist = hi_counts / hi_counts.sum() if hi_counts.sum() > 0 else hi_counts
            lo_dist = lo_counts / lo_counts.sum() if lo_counts.sum() > 0 else lo_counts
            tvd = 0.5 * np.abs(hi_dist - lo_dist).sum()

            diagnostics.append({
                "seed": seed,
                "trial": t["trial"],
                "hi_n_relations": len(hi_rels),
                "lo_n_relations": len(lo_rels),
                "relation_jaccard": jaccard,
                "relation_tvd": tvd,
            })
    return diagnostics


def seed_specific_effects(records):
    """Compute per-seed effect estimates."""
    results = {}
    for seed in SEEDS:
        seed_trials = [r for r in records if r["seed"] == seed]
        if not seed_trials:
            continue
        deltas = [r["paired_delta_lp"] for r in seed_trials]
        n = len(deltas)
        mean = np.mean(deltas)
        se = np.std(deltas, ddof=1) / np.sqrt(n)

        from scipy.stats import ttest_1samp
        t_stat, p_val = ttest_1samp(deltas, 0)

        sign_neg = sum(1 for d in deltas if d < 0)
        results[seed] = {
            "n_trials": n,
            "mean_paired_delta_lp": float(mean),
            "se": float(se),
            "t_stat": float(t_stat),
            "p_value": float(p_val),
            "ci_95_lower": float(mean - 1.96 * se),
            "ci_95_upper": float(mean + 1.96 * se),
            "sign_negative": sign_neg,
            "sign_fraction": sign_neg / n,
        }
    return results


def pooled_with_seed_fe(records):
    """Pooled analysis with seed fixed effects (demeaned)."""
    all_deltas = []
    demeaned = []
    seed_means = {}
    for seed in SEEDS:
        deltas = [r["paired_delta_lp"] for r in records if r["seed"] == seed]
        if deltas:
            seed_means[seed] = np.mean(deltas)
            all_deltas.extend(deltas)
            demeaned.extend([d - np.mean(deltas) for d in deltas])

    grand_mean = np.mean(all_deltas)
    n = len(all_deltas)
    k = len(seed_means)

    from scipy.stats import ttest_1samp
    t_raw, p_raw = ttest_1samp(all_deltas, 0)

    se_pooled = np.std(all_deltas, ddof=1) / np.sqrt(n)
    se_clustered = np.std(list(seed_means.values()), ddof=1) / np.sqrt(k)

    t_clustered, p_clustered = ttest_1samp(list(seed_means.values()), 0)

    cohens_d = grand_mean / np.std(all_deltas, ddof=1)

    return {
        "n_trials": n,
        "n_seeds": k,
        "grand_mean": float(grand_mean),
        "se_pooled_naive": float(se_pooled),
        "se_seed_clustered": float(se_clustered),
        "t_naive": float(t_raw),
        "p_naive": float(p_raw),
        "t_seed_clustered": float(t_clustered),
        "p_seed_clustered": float(p_clustered),
        "cohens_d": float(cohens_d),
        "sign_negative": sum(1 for d in all_deltas if d < 0),
        "sign_fraction": sum(1 for d in all_deltas if d < 0) / n,
        "seed_means": {int(k): float(v) for k, v in seed_means.items()},
    }


def hierarchical_bootstrap(records, n_boot=10000, rng_seed=42):
    """Hierarchical bootstrap: resample seeds, then trials within seed.

    This correctly propagates trajectory-level uncertainty.
    """
    rng = np.random.default_rng(rng_seed)
    by_seed = {}
    for seed in SEEDS:
        by_seed[seed] = [r["paired_delta_lp"] for r in records if r["seed"] == seed]

    available_seeds = [s for s in SEEDS if s in by_seed and len(by_seed[s]) > 0]
    k = len(available_seeds)

    boot_means = []
    for _ in range(n_boot):
        sampled_seeds = rng.choice(available_seeds, size=k, replace=True)
        boot_trials = []
        for s in sampled_seeds:
            trials = by_seed[s]
            resampled = rng.choice(trials, size=len(trials), replace=True)
            boot_trials.extend(resampled)
        boot_means.append(np.mean(boot_trials))

    boot_means = np.array(boot_means)
    return {
        "n_bootstrap": n_boot,
        "mean": float(np.mean(boot_means)),
        "se": float(np.std(boot_means)),
        "ci_95_lower": float(np.percentile(boot_means, 2.5)),
        "ci_95_upper": float(np.percentile(boot_means, 97.5)),
        "frac_negative": float((boot_means < 0).mean()),
        "method": "Hierarchical bootstrap: resample seeds (with replacement), "
                  "then trials within each resampled seed (with replacement).",
    }


def sign_flip_test(records, n_perm=10000, rng_seed=42):
    """Seed-stratified sign-flip permutation test.

    Under H0, the sign of each seed-level mean is equally likely to be
    positive or negative. With k=3 seeds, there are 2^3=8 sign patterns.
    This is a conservative exact-like test at the trajectory level.
    """
    rng = np.random.default_rng(rng_seed)
    seed_means = {}
    for seed in SEEDS:
        deltas = [r["paired_delta_lp"] for r in records if r["seed"] == seed]
        if deltas:
            seed_means[seed] = np.mean(deltas)

    observed = np.mean(list(seed_means.values()))
    abs_means = {s: abs(m) for s, m in seed_means.items()}
    seeds_list = list(abs_means.keys())
    k = len(seeds_list)

    n_extreme = 0
    for _ in range(n_perm):
        signs = rng.choice([-1, 1], size=k)
        perm_mean = np.mean([signs[i] * abs_means[seeds_list[i]] for i in range(k)])
        if perm_mean <= observed:
            n_extreme += 1

    return {
        "observed_mean": float(observed),
        "p_value": float(n_extreme / n_perm),
        "n_permutations": n_perm,
        "n_seeds": k,
        "note": f"With {k} seeds, exact test has {2**k} sign patterns. "
                f"All {k} negative → p = 1/{2**k} = {1/2**k:.3f}.",
    }


def dose_response(records):
    """Trial-level dose-response: cosine gap vs logit-damage gap."""
    cos_gaps = [r["cos_gap"] for r in records]
    lp_gaps = [r["paired_delta_lp"] for r in records]
    dmg_gaps = [r["paired_damage"] for r in records]

    from scipy.stats import pearsonr, spearmanr
    r_lp, p_lp = pearsonr(cos_gaps, lp_gaps)
    rho_lp, prho_lp = spearmanr(cos_gaps, lp_gaps)
    r_dmg, p_dmg = pearsonr(cos_gaps, dmg_gaps)

    return {
        "n_trials": len(records),
        "cosine_gap_vs_logprob_gap": {
            "pearson_r": float(r_lp),
            "pearson_p": float(p_lp),
            "spearman_rho": float(rho_lp),
            "spearman_p": float(prho_lp),
        },
        "cosine_gap_vs_damage_gap": {
            "pearson_r": float(r_dmg),
            "pearson_p": float(p_dmg),
        },
        "note": "Trial-level correlation (n=30). Modest expected because "
                "the design compares discrete HIGH vs LOW, not a continuous gradient.",
    }


def main():
    print("=" * 70)
    print("Logit-Damage A/B Intervention — Corrected Analysis")
    print("=" * 70)

    all_data = load_all_seeds()
    if not all_data:
        print("ERROR: No data found")
        sys.exit(1)

    records = compute_trial_effects(all_data)
    balance = batch_balance_diagnostics(all_data)
    print(f"\n  Loaded {len(records)} trial-level observations across {len(all_data)} seeds")

    # 1. Batch balance
    print("\n--- Batch-Property Balance ---")
    jaccard_vals = [b["relation_jaccard"] for b in balance]
    tvd_vals = [b["relation_tvd"] for b in balance]
    print(f"  Relation Jaccard overlap:  mean={np.mean(jaccard_vals):.3f} "
          f"[{np.min(jaccard_vals):.3f}, {np.max(jaccard_vals):.3f}]")
    print(f"  Relation TVD (0=identical): mean={np.mean(tvd_vals):.3f} "
          f"[{np.min(tvd_vals):.3f}, {np.max(tvd_vals):.3f}]")
    print(f"  Note: batches are NOT relation-matched. HIGH and LOW differ in "
          f"semantic composition. This is a limitation; the effect should be "
          f"interpreted as 'applying this specific HIGH batch vs this specific "
          f"LOW batch', not purely geometric.")

    # 2. Seed-specific effects
    print("\n--- Seed-Specific Effects (trial-level means) ---")
    seed_fx = seed_specific_effects(records)
    for seed, fx in seed_fx.items():
        print(f"  Seed {seed}: mean={fx['mean_paired_delta_lp']:+.5f} "
              f"± {fx['se']:.5f}, t={fx['t_stat']:.2f}, p={fx['p_value']:.4f}, "
              f"sign={fx['sign_negative']}/{fx['n_trials']} negative")

    # 3. Pooled with seed clustering
    print("\n--- Pooled Analysis ---")
    pooled = pooled_with_seed_fe(records)
    print(f"  Grand mean: {pooled['grand_mean']:+.5f}")
    print(f"  Naive SE (treating trials as iid): {pooled['se_pooled_naive']:.5f}")
    print(f"  Seed-clustered SE: {pooled['se_seed_clustered']:.5f}")
    print(f"  Naive t={pooled['t_naive']:.2f}, p={pooled['p_naive']:.6f}")
    print(f"  Seed-clustered t={pooled['t_seed_clustered']:.2f}, p={pooled['p_seed_clustered']:.4f}")
    print(f"  Cohen's d: {pooled['cohens_d']:.2f}")
    print(f"  Sign: {pooled['sign_negative']}/{pooled['n_trials']} negative ({pooled['sign_fraction']:.1%})")

    # 4. Hierarchical bootstrap
    print("\n--- Hierarchical Bootstrap (n=10000) ---")
    boot = hierarchical_bootstrap(records)
    print(f"  Mean: {boot['mean']:+.5f} ± {boot['se']:.5f}")
    print(f"  95% CI: [{boot['ci_95_lower']:+.5f}, {boot['ci_95_upper']:+.5f}]")
    print(f"  P(mean < 0): {boot['frac_negative']:.4f}")

    # 5. Sign-flip test
    print("\n--- Seed-Stratified Sign-Flip Test ---")
    signflip = sign_flip_test(records)
    print(f"  Observed seed-level mean: {signflip['observed_mean']:+.5f}")
    print(f"  p = {signflip['p_value']:.4f} ({signflip['note']})")

    # 6. Dose-response
    print("\n--- Dose-Response (trial-level) ---")
    dr = dose_response(records)
    cr = dr["cosine_gap_vs_logprob_gap"]
    print(f"  Cosine gap → logprob gap: r={cr['pearson_r']:.3f} (p={cr['pearson_p']:.3f}), "
          f"ρ={cr['spearman_rho']:.3f} (p={cr['spearman_p']:.3f})")
    cd = dr["cosine_gap_vs_damage_gap"]
    print(f"  Cosine gap → damage gap:  r={cd['pearson_r']:.3f} (p={cd['pearson_p']:.3f})")

    # 7. Paper-ready summary table
    print("\n" + "=" * 70)
    print("PAPER TABLE: Direct Logit-Damage A/B Intervention")
    print("=" * 70)
    print(f"{'Seed':<8} {'Trials':<8} {'Mean Δlp':<12} {'SE':<10} {'t':<8} {'p':<10} {'Sign(−)':<8}")
    print("-" * 64)
    for seed in SEEDS:
        if seed in seed_fx:
            fx = seed_fx[seed]
            print(f"{seed:<8} {fx['n_trials']:<8} {fx['mean_paired_delta_lp']:+.5f}    "
                  f"{fx['se']:.5f}   {fx['t_stat']:<+7.2f} {fx['p_value']:<10.4f} "
                  f"{fx['sign_negative']}/{fx['n_trials']}")
    print("-" * 64)
    print(f"{'Pooled':<8} {pooled['n_trials']:<8} {pooled['grand_mean']:+.5f}    "
          f"{pooled['se_seed_clustered']:.5f}   {pooled['t_seed_clustered']:<+7.2f} "
          f"{pooled['p_seed_clustered']:<10.4f} {pooled['sign_negative']}/{pooled['n_trials']}")
    print(f"{'Boot':<8} {'':<8} {boot['mean']:+.5f}    {boot['se']:.5f}   "
          f"{'':8} {'':10} P(−)={boot['frac_negative']:.3f}")
    print(f"\n  Hierarchical bootstrap 95% CI: [{boot['ci_95_lower']:+.5f}, {boot['ci_95_upper']:+.5f}]")
    print(f"  Sign-flip test: p = {signflip['p_value']:.3f}")
    print(f"  Cohen's d = {pooled['cohens_d']:.2f}")

    # Save
    output = {
        "description": (
            "Corrected A/B intervention analysis. Treatment unit is the batch pair (n=30 trials, "
            "3 seeds × 10 trials). Seed-clustered SEs and hierarchical bootstrap account for "
            "nested structure. Sign-flip test uses seed-level means (3 units)."
        ),
        "batch_balance": {
            "mean_relation_jaccard": float(np.mean(jaccard_vals)),
            "mean_relation_tvd": float(np.mean(tvd_vals)),
            "note": "Batches are not relation-matched. HIGH and LOW differ in semantic composition.",
            "per_trial": balance,
        },
        "seed_specific": {int(k): v for k, v in seed_fx.items()},
        "pooled": pooled,
        "hierarchical_bootstrap": boot,
        "sign_flip_test": signflip,
        "dose_response": dr,
        "trial_records": records,
    }

    out_path = RESULTS_BASE / "corrected_analysis.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\n  Saved: {out_path}")


if __name__ == "__main__":
    main()

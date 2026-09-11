#!/usr/bin/env python3
"""Build a frozen result registry for the ICLR paper.

Reads all corrected (prob-pref) result files and produces:
  1. paper/result_registry.json — every headline number with its source
  2. paper/data/ updates — copies corrected files
  3. paper/RESULTS_MANIFEST_v2.md — updated claim-to-number manifest

Usage:
    uv run python scripts/build_result_registry.py
"""

import json
import os
import shutil
import sys
from pathlib import Path

PROJECT = Path(__file__).resolve().parent.parent
RESULTS = PROJECT / "results"
PAPER = PROJECT / "paper"
PAPER_DATA = PAPER / "data"

registry = {}


def reg(key, value, source, field, metric="prob_pref", checkpoint=None, evaluator=None):
    if value is None:
        return
    registry[key] = {
        "value": round(value, 4) if isinstance(value, float) else value,
        "metric": metric,
        "source": str(source),
        "field": field,
        "checkpoint": checkpoint,
        "evaluator": evaluator,
    }


def read_probpref(path):
    with open(path) as f:
        d = json.load(f)
    if d.get("n_cases", 0) == 0:
        return None
    return d


def read_v2(path):
    with open(path) as f:
        d = json.load(f)
    checkpoints = sorted(d.keys(), key=lambda k: int(k.replace("_edits", "")))
    last_key = checkpoints[-1]
    return d[last_key], last_key


def mean(vals):
    vals = [v for v in vals if v is not None]
    return sum(vals) / len(vals) if vals else None


# =====================================================================
# SECTION 3: Selective Forgetting
# =====================================================================
print("Section 3: Selective Forgetting...")

# MVE 2K reproduction
for alg, exp in [("AlphaEdit", "mve1_alphaedit_mcf"), ("MEMIT", "mve2_memit_mcf")]:
    alg_dir = "AlphaEdit" if alg == "AlphaEdit" else "MEMIT"
    effs, neighs, paras = [], [], []
    for seed in [42, 2024, 137, 7, 99]:
        path = RESULTS / exp / f"seed{seed}" / "2000edits" / alg_dir / "run_000" / "probpref_summary.json"
        if not path.exists():
            continue
        d = read_probpref(path)
        if not d:
            continue
        pp = d["all_facts"]["prob_preference"]
        eff = pp["rewrite_success"]
        neigh = pp["neighborhood_success"]
        para = pp["paraphrase_success"]
        effs.append(eff)
        neighs.append(neigh)
        paras.append(para)
        prefix = f"s3_mve_{alg.lower()}_s{seed}"
        reg(f"{prefix}_eff", eff, path, "all_facts.prob_preference.rewrite_success")
        reg(f"{prefix}_neigh", neigh, path, "all_facts.prob_preference.neighborhood_success")
        reg(f"{prefix}_para", para, path, "all_facts.prob_preference.paraphrase_success")
    reg(f"s3_mve_{alg.lower()}_mean_eff", mean(effs), "computed", "5-seed mean")
    reg(f"s3_mve_{alg.lower()}_mean_neigh", mean(neighs), "computed", "5-seed mean")
    reg(f"s3_mve_{alg.lower()}_mean_para", mean(paras), "computed", "5-seed mean")

# Failure curve 10K
for alg in ["AlphaEdit", "MEMIT"]:
    effs, f1ks, l1ks, neighs = [], [], [], []
    for seed in [42, 2024, 137]:
        path = RESULTS / "failure_curve_checkpointed" / f"seed{seed}" / "10000edits" / alg / "run_000" / "probpref_summary.json"
        if not path.exists():
            continue
        d = read_probpref(path)
        if not d:
            continue
        pp = d["all_facts"]["prob_preference"]
        f1k = d["first_1k"]["prob_preference"]["rewrite_success"]
        l1k = d["latest_1k"]["prob_preference"]["rewrite_success"]
        effs.append(pp["rewrite_success"])
        neighs.append(pp["neighborhood_success"])
        f1ks.append(f1k)
        l1ks.append(l1k)
        prefix = f"s3_fc10k_{alg.lower()}_s{seed}"
        reg(f"{prefix}_eff", pp["rewrite_success"], path, "all_facts.prob_preference.rewrite_success", checkpoint="10000edits")
        reg(f"{prefix}_neigh", pp["neighborhood_success"], path, "all_facts.prob_preference.neighborhood_success", checkpoint="10000edits")
        reg(f"{prefix}_f1k", f1k, path, "first_1k.prob_preference.rewrite_success", checkpoint="10000edits")
        reg(f"{prefix}_l1k", l1k, path, "latest_1k.prob_preference.rewrite_success", checkpoint="10000edits")
    if effs:
        alg_l = alg.lower()
        reg(f"s3_fc10k_{alg_l}_mean_eff", mean(effs), "computed", "3-seed mean")
        reg(f"s3_fc10k_{alg_l}_mean_neigh", mean(neighs), "computed", "3-seed mean")
        reg(f"s3_fc10k_{alg_l}_mean_f1k", mean(f1ks), "computed", "3-seed mean")
        reg(f"s3_fc10k_{alg_l}_mean_l1k", mean(l1ks), "computed", "3-seed mean")
        reg(f"s3_fc10k_{alg_l}_age_gap", mean(l1ks) - mean(f1ks), "computed", "mean(l1k) - mean(f1k)")

# Failure curve intermediate checkpoints (AlphaEdit seed 42)
for edits in [2000, 3000, 5000, 7000, 10000]:
    path = RESULTS / "failure_curve_checkpointed" / "seed42" / f"{edits}edits" / "AlphaEdit" / "run_000" / "probpref_summary.json"
    if not path.exists():
        continue
    d = read_probpref(path)
    if not d:
        continue
    pp = d["all_facts"]["prob_preference"]
    reg(f"s3_fc_ae_s42_{edits}_eff", pp["rewrite_success"], path, "all_facts.prob_preference.rewrite_success", checkpoint=f"{edits}edits")


# =====================================================================
# SECTION 4: Fixed-Batch Path Dependence
# =====================================================================
print("Section 4: Fixed-Batch Path Dependence...")

orderings = ["fb_high_exposure", "fb_low_exposure", "fb_random0", "fb_random1", "fb_random2"]
seeds = [42, 2024, 137]

for ordering in orderings:
    effs_across_seeds = []
    f1ks_across_seeds = []
    for seed in seeds:
        path = RESULTS / "matched_ordering" / "AlphaEdit" / ordering / f"seed{seed}" / f"full_eval_seed{seed}_v2.json"
        if not path.exists():
            continue
        m, ckpt = read_v2(path)
        af = m["all_facts"]
        f1k = m["first_1k"]
        l1k = m["latest_1k"]
        prefix = f"s4_{ordering}_s{seed}"
        reg(f"{prefix}_eff", af["efficacy"], path, "all_facts.efficacy", checkpoint=ckpt, evaluator="eval_matched_ordering.py v2")
        reg(f"{prefix}_neigh", af["neighborhood"], path, "all_facts.neighborhood", checkpoint=ckpt, evaluator="eval_matched_ordering.py v2")
        reg(f"{prefix}_para", af["paraphrase"], path, "all_facts.paraphrase", checkpoint=ckpt, evaluator="eval_matched_ordering.py v2")
        reg(f"{prefix}_f1k", f1k["efficacy"], path, "first_1k.efficacy", checkpoint=ckpt, evaluator="eval_matched_ordering.py v2")
        reg(f"{prefix}_l1k", l1k["efficacy"], path, "latest_1k.efficacy", checkpoint=ckpt, evaluator="eval_matched_ordering.py v2")
        effs_across_seeds.append(af["efficacy"])
        f1ks_across_seeds.append(f1k["efficacy"])
    if effs_across_seeds:
        o_short = ordering.replace("fb_", "").replace("_exposure", "")
        reg(f"s4_{ordering}_mean_eff", mean(effs_across_seeds), "computed", "3-seed mean")
        reg(f"s4_{ordering}_mean_f1k", mean(f1ks_across_seeds), "computed", "3-seed mean")

# Compute gaps
hi_mean = registry.get("s4_fb_high_exposure_mean_eff", {}).get("value")
lo_mean = registry.get("s4_fb_low_exposure_mean_eff", {}).get("value")
rnd_means = [registry.get(f"s4_fb_random{i}_mean_eff", {}).get("value") for i in range(3)]
rnd_mean = mean(rnd_means)
if hi_mean and lo_mean:
    reg("s4_hi_lo_gap", hi_mean - lo_mean, "computed", "HIGH mean - LOW mean")
if hi_mean and rnd_mean:
    reg("s4_hi_rnd_gap", hi_mean - rnd_mean, "computed", "HIGH mean - RANDOM mean")


# =====================================================================
# SECTION 5: Why Temporal Order Matters
# =====================================================================
print("Section 5: Mechanism...")

# Margin model
margin_path = RESULTS / "margin_survival_model.json"
if margin_path.exists():
    with open(margin_path) as f:
        mm = json.load(f)
    for seed_str, traj in mm.get("per_trajectory", {}).items():
        reg(f"s5_margin_s{seed_str}_cos_coef", traj.get("cosine_coef"), margin_path, f"per_trajectory.{seed_str}.cosine_coef")
        reg(f"s5_margin_s{seed_str}_cos_pval", traj.get("cosine_pval"), margin_path, f"per_trajectory.{seed_str}.cosine_pval")
        reg(f"s5_margin_s{seed_str}_m0_r2", traj.get("M0_R2"), margin_path, f"per_trajectory.{seed_str}.M0_R2")
        reg(f"s5_margin_s{seed_str}_m1_r2", traj.get("M1_R2"), margin_path, f"per_trajectory.{seed_str}.M1_R2")
        reg(f"s5_margin_s{seed_str}_m2_r2", traj.get("M2_R2"), margin_path, f"per_trajectory.{seed_str}.M2_R2")
        reg(f"s5_margin_s{seed_str}_cos_effect_per01", traj.get("cosine_effect_per_01"), margin_path, f"per_trajectory.{seed_str}.cosine_effect_per_01")
    for seed_str, loo in mm.get("leave_one_out", {}).items():
        reg(f"s5_margin_loo_s{seed_str}_r2_base", loo.get("R2_base"), margin_path, f"leave_one_out.{seed_str}.R2_base")
        reg(f"s5_margin_loo_s{seed_str}_r2_full", loo.get("R2_full"), margin_path, f"leave_one_out.{seed_str}.R2_full")

# A/B interventions (continuous logprob — unchanged)
ab_path = RESULTS / "pooled_ab_results.json"
if ab_path.exists():
    with open(ab_path) as f:
        ab = json.load(f)
    s = ab.get("summary", {})
    pooled = s.get("pooled", {})
    diff_batch = s.get("different_batch", {})
    same_fact = s.get("same_fact", {})
    reg("s5_ab_pooled_sign", pooled.get("sign_count"), ab_path, "summary.pooled.sign_count", metric="continuous_logprob")
    reg("s5_ab_pooled_p", pooled.get("sign_test_p"), ab_path, "summary.pooled.sign_test_p", metric="continuous_logprob")
    reg("s5_ab_pooled_mean", pooled.get("mean_delta"), ab_path, "summary.pooled.mean_delta", metric="continuous_logprob")
    reg("s5_ab_diffbatch_sign", diff_batch.get("sign_count"), ab_path, "summary.different_batch.sign_count", metric="continuous_logprob")
    reg("s5_ab_samefact_sign", same_fact.get("sign_count"), ab_path, "summary.same_fact.sign_count", metric="continuous_logprob")
    reg("s5_ab_samefact_p", same_fact.get("sign_test_p"), ab_path, "summary.same_fact.sign_test_p", metric="continuous_logprob")
    boot = ab.get("hierarchical_bootstrap", {})
    reg("s5_ab_boot_ci_lo", boot.get("ci_lo"), ab_path, "hierarchical_bootstrap.ci_lo", metric="continuous_logprob")
    reg("s5_ab_boot_ci_hi", boot.get("ci_hi"), ab_path, "hierarchical_bootstrap.ci_hi", metric="continuous_logprob")

# Displacement correlations (prob-pref)
disp_path = RESULTS / "norm_vs_displacement_probpref.json"
if disp_path.exists():
    with open(disp_path) as f:
        disp = json.load(f)
    corr = disp.get("correlations", {})
    pcorr = disp.get("partial_correlations", {})
    for k, v in corr.items():
        if isinstance(v, dict):
            reg(f"s5_disp_{k}_r", v.get("spearman_r"), disp_path, f"correlations.{k}.spearman_r")
            reg(f"s5_disp_{k}_p", v.get("p"), disp_path, f"correlations.{k}.p")
    for k, v in pcorr.items():
        if isinstance(v, dict):
            reg(f"s5_pdisp_{k}_r", v.get("r"), disp_path, f"partial_correlations.{k}.r")
            reg(f"s5_pdisp_{k}_p", v.get("p"), disp_path, f"partial_correlations.{k}.p")


# =====================================================================
# SECTION 6: Cross-Method
# =====================================================================
print("Section 6: Cross-Method...")

cross_algs = [
    ("AlphaEdit", "AlphaEdit"),
    ("EvoEdit", "EvoEdit"),
    ("MEMIT-Seq", "MEMIT-Seq-lp1.0-ld0.0-cache0"),
    ("PathGuard", "PathGuard-ED-M200-e0.1"),
    ("PathGuard-poly2", "PathGuard-ED-poly2-hybrid-M200-e0.1"),
]

cross_method_table = {}
for label, alg_dir in cross_algs:
    cross_method_table[label] = {}
    for ordering in ["fb_high_exposure", "fb_low_exposure", "fb_random0"]:
        path = RESULTS / "matched_ordering" / alg_dir / ordering / "seed42" / "full_eval_seed42_v2.json"
        if not path.exists():
            continue
        m, ckpt = read_v2(path)
        af = m["all_facts"]
        f1k_m = m["first_1k"]
        o_short = ordering.replace("fb_", "").replace("_exposure", "")
        prefix = f"s6_{label.lower().replace('-','_')}_{o_short}"
        reg(f"{prefix}_eff", af["efficacy"], path, "all_facts.efficacy", checkpoint=ckpt, evaluator="eval_matched_ordering.py v2")
        reg(f"{prefix}_neigh", af["neighborhood"], path, "all_facts.neighborhood", checkpoint=ckpt, evaluator="eval_matched_ordering.py v2")
        reg(f"{prefix}_f1k", f1k_m["efficacy"], path, "first_1k.efficacy", checkpoint=ckpt, evaluator="eval_matched_ordering.py v2")
        cross_method_table[label][o_short] = {
            "eff": af["efficacy"], "neigh": af["neighborhood"], "f1k": f1k_m["efficacy"],
        }
    hi = cross_method_table[label].get("high", {}).get("eff")
    lo = cross_method_table[label].get("low", {}).get("eff")
    rnd = cross_method_table[label].get("random0", {}).get("eff")
    if hi is not None and rnd is not None:
        reg(f"s6_{label.lower().replace('-','_')}_hi_rnd_gap", hi - rnd, "computed", "HIGH - RANDOM")


# =====================================================================
# APPENDIX: Scheduler
# =====================================================================
print("Appendix: Scheduler...")

for ordering in ["fb_high_exposure", "sched_exposure_only", "sched_conditioning_only", "sched_balanced", "fb_random0", "fb_low_exposure"]:
    effs = []
    for seed in seeds:
        path = RESULTS / "matched_ordering" / "AlphaEdit" / ordering / f"seed{seed}" / f"full_eval_seed{seed}_v2.json"
        if not path.exists():
            continue
        m, ckpt = read_v2(path)
        effs.append(m["all_facts"]["efficacy"])
    if effs:
        o_short = ordering.replace("fb_", "").replace("sched_", "")
        reg(f"app_sched_{o_short}_mean_eff", mean(effs), "computed", "3-seed mean", evaluator="eval_matched_ordering.py v2")


# =====================================================================
# APPENDIX: Suffix Rescue
# =====================================================================
print("Appendix: Suffix Rescue...")

for ordering in ["fb_high_exposure", "suffix_random", "suffix_spread", "suffix_random_late", "suffix_spread_late", "fb_random0"]:
    for seed in seeds:
        path = RESULTS / "matched_ordering" / "AlphaEdit" / ordering / f"seed{seed}" / f"full_eval_seed{seed}_v2.json"
        if not path.exists():
            continue
        m, ckpt = read_v2(path)
        o_short = ordering.replace("fb_", "").replace("suffix_", "sfx_")
        reg(f"app_suffix_{o_short}_s{seed}_eff", m["all_facts"]["efficacy"], path, "all_facts.efficacy", checkpoint=ckpt, evaluator="eval_matched_ordering.py v2")
        reg(f"app_suffix_{o_short}_s{seed}_f1k", m["first_1k"]["efficacy"], path, "first_1k.efficacy", checkpoint=ckpt, evaluator="eval_matched_ordering.py v2")


# =====================================================================
# APPENDIX: Polykernel curves (seed 42)
# =====================================================================
print("Appendix: Polykernel curves...")

poly_algs = [
    "MEMIT-Seq-poly2-hybrid-lp1.0-ld0.0-cache0",
    "MEMIT-Seq-poly3-hybrid-lp1.0-ld0.0-cache0",
    "MEMIT-Seq-poly4-hybrid-lp1.0-ld0.0-cache0",
    "MEMIT-Seq-poly2-lp1.0-ld0.0-cache0",
    "MEMIT-Seq-lp1.0-ld0.0-cache0",
]

for alg in poly_algs:
    for edits in [2000, 5000, 10000]:
        path = RESULTS / "failure_curve_checkpointed" / "seed42" / f"{edits}edits" / alg / "run_000" / "probpref_summary.json"
        if not path.exists():
            continue
        d = read_probpref(path)
        if not d:
            continue
        pp = d["all_facts"]["prob_preference"]
        short = alg.replace("MEMIT-Seq-", "").replace("-lp1.0-ld0.0-cache0", "")
        if not short:
            short = "memit_seq"
        reg(f"app_poly_{short}_{edits}_eff", pp["rewrite_success"], path, "all_facts.prob_preference.rewrite_success", checkpoint=f"{edits}edits")


# =====================================================================
# APPENDIX: GPT-J
# =====================================================================
print("Appendix: GPT-J...")

for alg in ["AlphaEdit", "MEMIT", "AlphaEdit-C0-15000.0", "MEMIT-Seq-lp1.0-ld0.0-cache0"]:
    for seed in [42, 2024]:
        path = RESULTS / "failure_curve_gptj" / f"seed{seed}" / "10000edits" / alg / "run_000" / "probpref_summary.json"
        if not path.exists():
            continue
        d = read_probpref(path)
        if not d:
            continue
        pp = d["all_facts"]["prob_preference"]
        alg_short = alg.replace("-C0-15000.0", "_c0").replace("-lp1.0-ld0.0-cache0", "").replace("MEMIT-Seq", "mseq").lower()
        reg(f"app_gptj_{alg_short}_s{seed}_eff", pp["rewrite_success"], path, "all_facts.prob_preference.rewrite_success", checkpoint="10000edits")
        reg(f"app_gptj_{alg_short}_s{seed}_neigh", pp["neighborhood_success"], path, "all_facts.prob_preference.neighborhood_success", checkpoint="10000edits")


# =====================================================================
# APPENDIX: Key clustered/dispersed
# =====================================================================
print("Appendix: Key clustered/dispersed...")

for ordering in ["key_clustered", "key_dispersed"]:
    for seed in [42, 2024, 137]:
        path = RESULTS / "matched_ordering" / "AlphaEdit" / ordering / f"seed{seed}" / f"full_eval_seed{seed}_v2.json"
        if not path.exists():
            continue
        m, ckpt = read_v2(path)
        af = m["all_facts"]
        reg(f"app_{ordering}_s{seed}_eff", af["efficacy"], path, "all_facts.efficacy", checkpoint=ckpt, evaluator="eval_matched_ordering.py v2")
        reg(f"app_{ordering}_s{seed}_neigh", af["neighborhood"], path, "all_facts.neighborhood", checkpoint=ckpt, evaluator="eval_matched_ordering.py v2")
        reg(f"app_{ordering}_s{seed}_f1k", m["first_1k"]["efficacy"], path, "first_1k.efficacy", checkpoint=ckpt, evaluator="eval_matched_ordering.py v2")


# =====================================================================
# SAVE REGISTRY
# =====================================================================
print(f"\nRegistry: {len(registry)} entries")

PAPER_DATA.mkdir(parents=True, exist_ok=True)
reg_path = PAPER / "result_registry.json"
with open(reg_path, "w") as f:
    json.dump(registry, f, indent=2, default=str)
print(f"Saved: {reg_path}")

# =====================================================================
# COPY/UPDATE paper/data/
# =====================================================================
print("\nUpdating paper/data/...")

copies = [
    (RESULTS / "margin_survival_model.json", PAPER_DATA / "margin_survival_model.json"),
    (RESULTS / "norm_vs_displacement_probpref.json", PAPER_DATA / "norm_vs_displacement_probpref.json"),
]
for src, dst in copies:
    if src.exists():
        shutil.copy2(src, dst)
        print(f"  Copied: {dst.name}")

# Write cross-method table
with open(PAPER_DATA / "cross_method_table.json", "w") as f:
    json.dump(cross_method_table, f, indent=2)
print("  Wrote: cross_method_table.json")

# =====================================================================
# BUILD MANIFEST
# =====================================================================
print("\nBuilding RESULTS_MANIFEST_v2.md...")


def pct(v):
    return f"{v*100:.1f}%" if v is not None else "N/A"


def pp(v):
    return f"{v*100:+.1f}pp" if v is not None else "N/A"


def rv(key):
    return registry.get(key, {}).get("value")


lines = ["# Results Manifest v2 — Probability-Preference Metric", ""]
lines.append("All numbers use the benchmark-standard probability-preference metric unless noted.")
lines.append("Generated by `scripts/build_result_registry.py`. Source: `paper/result_registry.json`.")
lines.append("")

# Section 3
lines.append("## Section 3: Age-Selective Forgetting")
lines.append("")
lines.append("### MVE Reproduction @ 2K (5-seed mean)")
lines.append(f"- AlphaEdit: eff={pct(rv('s3_mve_alphaedit_mean_eff'))}, neigh={pct(rv('s3_mve_alphaedit_mean_neigh'))}, para={pct(rv('s3_mve_alphaedit_mean_para'))}")
lines.append(f"- MEMIT: eff={pct(rv('s3_mve_memit_mean_eff'))}, neigh={pct(rv('s3_mve_memit_mean_neigh'))}, para={pct(rv('s3_mve_memit_mean_para'))}")
lines.append("")
lines.append("### Failure Curve @ 10K (3-seed mean)")
lines.append(f"- AlphaEdit: eff={pct(rv('s3_fc10k_alphaedit_mean_eff'))}, neigh={pct(rv('s3_fc10k_alphaedit_mean_neigh'))}")
lines.append(f"  - Oldest 1K: {pct(rv('s3_fc10k_alphaedit_mean_f1k'))}, Newest 1K: {pct(rv('s3_fc10k_alphaedit_mean_l1k'))}")
lines.append(f"  - **Age gap: {pp(rv('s3_fc10k_alphaedit_age_gap'))}**")
lines.append(f"- MEMIT: eff={pct(rv('s3_fc10k_memit_mean_eff'))}")
lines.append("")

# Section 4
lines.append("## Section 4: Fixed-Batch Path Dependence (10K)")
lines.append("")
lines.append("### 5-Path Dose-Response (3-seed mean)")
lines.append(f"- HIGH: {pct(rv('s4_fb_high_exposure_mean_eff'))}")
lines.append(f"- RANDOM0: {pct(rv('s4_fb_random0_mean_eff'))}")
lines.append(f"- RANDOM1: {pct(rv('s4_fb_random1_mean_eff'))}")
lines.append(f"- RANDOM2: {pct(rv('s4_fb_random2_mean_eff'))}")
lines.append(f"- LOW: {pct(rv('s4_fb_low_exposure_mean_eff'))}")
lines.append(f"- **HI-LO gap: {pp(rv('s4_hi_lo_gap'))}**")
lines.append(f"- **HI-RAND gap: {pp(rv('s4_hi_rnd_gap'))}**")
lines.append("")
lines.append("### Per-seed")
for seed in seeds:
    hi = rv(f"s4_fb_high_exposure_s{seed}_eff")
    lo = rv(f"s4_fb_low_exposure_s{seed}_eff")
    r0 = rv(f"s4_fb_random0_s{seed}_eff")
    gap = (hi - lo) if hi and lo else None
    lines.append(f"- Seed {seed}: HIGH={pct(hi)}, RAND0={pct(r0)}, LOW={pct(lo)}, gap={pp(gap)}")
lines.append("")

# Section 5
lines.append("## Section 5: Why Temporal Order Matters")
lines.append("")
lines.append("### Continuous Margin Model")
lines.append("")
lines.append("| Seed | M0 R² (age) | M1 R² (+margin) | M2 R² (+cosine) | Cosine p | Effect/+0.1 |")
lines.append("|---|---|---|---|---|---|")
for seed in [42, 2024, 137]:
    r2_m0 = rv(f"s5_margin_s{seed}_m0_r2")
    r2_m1 = rv(f"s5_margin_s{seed}_m1_r2")
    r2_m2 = rv(f"s5_margin_s{seed}_m2_r2")
    cos_p = rv(f"s5_margin_s{seed}_cos_pval")
    cos_eff = rv(f"s5_margin_s{seed}_cos_effect_per01")
    lines.append(f"| {seed} | {r2_m0:.3f} | {r2_m1:.3f} | {r2_m2:.3f} | {cos_p:.2e} | {cos_eff:.3f} |" if r2_m0 else f"| {seed} | N/A | N/A | N/A | N/A | N/A |")
lines.append("")
lines.append("### A/B Interventions (continuous logprob, unaffected by metric change)")
lines.append(f"- Pooled: {rv('s5_ab_pooled_sign')} trials, p={rv('s5_ab_pooled_p')}")
lines.append(f"- Different-batch: {rv('s5_ab_diffbatch_sign')} trials")
lines.append(f"- Same-fact: {rv('s5_ab_samefact_sign')} trials, p={rv('s5_ab_samefact_p')}")
lines.append(f"- Bootstrap CI: [{rv('s5_ab_boot_ci_lo')}, {rv('s5_ab_boot_ci_hi')}]")
lines.append("")
lines.append("### Displacement (prob-pref retention)")
lines.append("")
for k in ["D_vs_first1k", "N_vs_first1k", "D_unrel_vs_first1k"]:
    r = rv(f"s5_disp_{k}_r")
    p = rv(f"s5_disp_{k}_p")
    if r is not None:
        lines.append(f"- {k}: r={r:.3f}, p={p:.4f}")
for k in ["D_given_N_vs_first1k", "N_given_D_vs_first1k"]:
    r = rv(f"s5_pdisp_{k}_r")
    p = rv(f"s5_pdisp_{k}_p")
    if r is not None:
        lines.append(f"- Partial {k}: r={r:.3f}, p={p:.4f}")
lines.append("")

# Section 6
lines.append("## Section 6: Cross-Method (seed 42)")
lines.append("")
lines.append("| Method | HIGH | LOW | RANDOM | HI-RND Gap |")
lines.append("|---|---|---|---|---|")
for label, _ in cross_algs:
    hi = cross_method_table.get(label, {}).get("high", {}).get("eff")
    lo = cross_method_table.get(label, {}).get("low", {}).get("eff")
    rnd = cross_method_table.get(label, {}).get("random0", {}).get("eff")
    gap = (hi - rnd) if hi and rnd else None
    lines.append(f"| {label} | {pct(hi)} | {pct(lo)} | {pct(rnd)} | {pp(gap)} |")
lines.append("")

# Validation
lines.append("## Cross-Checks")
ae_mean = rv("s3_fc10k_alphaedit_mean_eff")
lines.append(f"- AlphaEdit 10K 3-seed mean: {pct(ae_mean)} (published: 66.78%)")
evo_rnd = rv("s6_evoedit_random0_eff")
lines.append(f"- EvoEdit random0 seed42: {pct(evo_rnd)} (published: 98.29%)")
lines.append("")

manifest_path = PAPER / "RESULTS_MANIFEST_v2.md"
with open(manifest_path, "w") as f:
    f.write("\n".join(lines))
print(f"Saved: {manifest_path}")

# =====================================================================
# VALIDATION
# =====================================================================
print("\n=== Validation ===")
if ae_mean:
    diff = abs(ae_mean - 0.6678)
    status = "PASS" if diff < 0.01 else "WARN"
    print(f"  [{status}] AlphaEdit 10K mean: {ae_mean:.4f} (target ~0.6678, diff={diff:.4f})")
if evo_rnd:
    diff = abs(evo_rnd - 0.9829)
    status = "PASS" if diff < 0.01 else "WARN"
    print(f"  [{status}] EvoEdit random0 s42: {evo_rnd:.4f} (target ~0.9829, diff={diff:.4f})")

argmax_in_headline = [k for k in registry if registry[k]["metric"] == "argmax" and k.startswith("s")]
if argmax_in_headline:
    print(f"  [WARN] {len(argmax_in_headline)} headline entries use argmax metric")
else:
    print("  [PASS] No argmax metrics in headline positions")

print(f"\nDone. {len(registry)} entries in registry.")

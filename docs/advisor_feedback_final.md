# Advisor Final Assessment Post All Corrections (2026-09-09)

## Verdict: Strong 7, favorable 8. Clear consensus 8 not guaranteed.

## The Biggest Scientific Change

The paper can **no longer lead with** "future-key cosine strongly predicts which edit will be forgotten." Under the corrected continuous margin model:
- Age + installation margin: ~58% R²
- Cosine: <0.1% additional R²
- Cosine significant on 2/3 seeds, consistent direction on all 3

**Old headline:** Exposure is the dominant individual-level predictor of forgetting.
**New headline:** Exposure is a small but consistent conditional risk factor. Small local effects compound into large path-level consequences.

## The Revised Central Claim

> Sequential editors exhibit large causal path dependence: age and installation strength largely determine individual memory robustness, while temporally concentrating identical edit batches can transform small local interference into substantial long-horizon selective forgetting.

## What Survives Strongly

1. **AlphaEdit reproduction at 2K** — 99% efficacy, matches published regime
2. **Age-selective forgetting** — 42.4pp gap at 10K, monotonic, 3 seeds
3. **Fixed-batch path dependence** — 22pp HI-RAND gap, 3 seeds, THE CENTERPIECE
4. **Random ordering is robust** — all random paths cluster at 95-97%
5. **Controlled A/B interventions** — immediate logprob damage from high-overlap batches
6. **Historical displacement** — path-state marker, not individual mediator
7. **Cross-method robustness differences** — AlphaEdit -26pp, EvoEdit -14pp, MEMIT-Seq/PathGuard ~0pp

## What Changes

### Survival model → "Longitudinal Continuous Margin Model"
- NOT a first-failure survival model anymore
- Report as: "What predicts the strength of an installed memory?"
- Do NOT quote AUC 0.926 (that was argmax binary)
- The AUC=1.0 overfitting needs to be documented (likely leakage via checkpoint×age dummies)

### Suffix rescue → Supporting evidence, half a page
- Seed 42: +9.7pp early, seed 2024: +3.7pp, seed 137: negligible
- "Modest improvement in some trajectories when done early, but schedule correction is heterogeneous and much less effective than avoiding a harmful prefix"

### Displacement → Path-level only
- r=-0.717 (p=0.030), partial r=-0.820 (p=0.007) — SURVIVES
- But: 15 paths now available (3 seeds × 5 orderings), use all with seed FE
- Does NOT mediate individual failure
- Verify the partial correlation p with exact sample accounting

### EvoEdit characterization → MUST FIX
- EvoEdit DOES use sequential null-space alignment
- Say "different preservation mechanisms exhibit different robustness"

## Recommended Paper Structure

1. **Introduction:** How does temporal ordering affect which memories survive?
2. **Benchmark-Aligned Setup:** Define prob-pref, NLL margin, argmax. Clean definition, no long narrative about the correction.
3. **Selective Forgetting at Long Horizons:** 5-seed 2K, 3-seed 10K age gap
4. **Local Exposure Effects:** Continuous margin model (age+margin dominate, cosine small but real), A/B interventions, MEMIT-Seq attenuation. Distinguish predictive importance from causal local effect.
5. **Identical Batches, Different Temporal Paths:** THE CENTERPIECE. Fixed HIGH/LOW/RANDOM, all 5 paths × 3 seeds. LOW fails on seed 137. Randomization as strong baseline.
6. **Path-Level State and Limited Intervention:** Displacement vs norm, suffix rescue (half page), scheduler negative result.
7. **Scope and Method Dependence:** GPT-J, zsRE, compact cross-method table. No PathGuard/Poly2 headline claims.
8. **Related Work, Implications, Limitations**
9. **Conclusion:** "The set of edits does not uniquely determine the resulting model; the temporal path through them matters."

## Technical Items

### MEMIT-Seq checkpoint safeguards needed
1. Store `checkpoint_format = full_model | edited_layers` in metadata
2. Store expected parameter-name manifest
3. Reject unexpected missing/extra keys
4. Verify tensor count and total parameter count
5. Hash checkpoint and evaluator version in every result JSON

### Displacement: use all 15 paths
- 3 seeds × {HIGH, LOW, RAND0, RAND1, RAND2}
- Seed fixed effects or within-seed centering
- Permutation of path labels within seed
- Effect size emphasis over nominal significance

### Prob-pref is still binary
- NLL margin is continuous
- prob-pref efficacy (NLL_new < NLL_true) is binary
- Don't call prob-pref a "continuous probability metric"

## Useful Optional Figure
Plot each edit's **margin loss from installation**:
  Δm_i(t) = m_i(t) - m_i(install)
Show degradation by exposure quantile. Directly separates weak installation from subsequent degradation.

## Score

| Dimension | Assessment |
|---|---|
| Contribution | 3-4/4 |
| Soundness after cleanup | 3/4 |
| Presentation potential | 4/4 |
| Most likely overall | **7** |
| Favorable reviewer | **8** |
| Clear consensus 8 | Not guaranteed |

The fixed-batch multi-seed result meets ICLR's standard. But a clear 8 depends on presentation: lead with causal temporal path dependence, not with individual exposure prediction.

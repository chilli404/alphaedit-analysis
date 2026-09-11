# Advisor Feedback: Post-Metric-Correction Assessment (2026-09-09)

## Context

After discovering and correcting the argmax → prob-pref evaluation metric mismatch, we shared the corrected results with the advisor. This document captures their assessment and the action items that follow.

## Advisor's Verdict

The correction was **necessary and correct**. The core contribution survives but the paper story must change slightly.

**Estimated score:**

| State | Score |
|---|---|
| Old paper with argmax headline results | Not submission-ready |
| Corrected results without rerunning first-failure analysis | 6–7 |
| Corrected first-failure model + focused rewrite | Strong 7, plausible 8 |
| Same + clear probability-level causal interventions + rigorous cross-path analysis | 8 defensible |

---

## What Survives Without Change

### 1. Age-selective forgetting (the core phenomenon)
- 42.4pp gap at 10K, monotonic across all seeds
- "AlphaEdit remains highly capable of installing recent edits while selectively losing much older ones"
- 2K numbers now align with official AlphaEdit regime — strengthens credibility

### 2. Fixed-batch path dependence (strongest causal result)
- Mean HI-RAND gap ~22pp
- "The same factual batches can yield substantially different long-horizon retention solely because of temporal sequence"
- Magnitude smaller than argmax but still large and consistent

### 3. Continuous log-probability interventions (unaffected by metric change)
- Different-batch HIGH/LOW A/B damage
- Same-fact/different-prompt A/B damage
- MEMIT-Seq attenuation of immediate damage
- Signed margin/path calculations
- GL2 finite-update validation
- CLaRE branch-ranking comparison

---

## What Must Change

### 1. The ordering narrative
**Old claim:** "Concentrate similar edits → collapse; spread them apart → safe"

**Corrected claim:** "Structured temporal ordering can create harmful paths, while randomization is consistently robust. High temporal concentration is particularly damaging, but globally minimizing exposure is not a universally reliable solution."

**Evidence:** Seed 137: HIGH 80.2%, LOW 80.4%, Random 94.6–95.6%. LOW is NOT automatically safe. Both engineered orderings are much worse than random on this seed.

### 2. Suffix rescue — DEMOTE to appendix
Under prob-pref:
- Seed 2024: ~+3.7pp (marginal)
- Seeds 42, 137: no clear signal
- Late rescue: negligible

**Remove from main story:**
- strong partial rescue
- critical intervention window
- prevention vs cure
- trajectory commitment after 7K
- persistent irrecoverability from suffix switching

**Acceptable appendix framing:** "Suffix reordering affects strict top-1 generation more than benchmark probability preference, so the intervention effect is metric-sensitive."

### 3. Phase-transition language — SOFTEN
prob-pref HIGH falls to ~69-73% (not ~30% as under argmax). Still severe vs 96-97% random, but "phase transition" harder to justify.

**Use:** sharp degradation, destructive path, abrupt crossover, large ordering-induced drop

**Avoid:** formal phase-transition claims unless corrected intermediate trajectories clearly show threshold behavior

### 4. Displacement as headline mechanism — DEMOTE
All path-level displacement correlations need recomputation using corrected endpoint metric. Even if significant, keep as "trajectory-level state marker" not "causal mechanism."

---

## Must-Do Before Submission

### P0: Rebuild first-failure survival model (CPU, no GPU needed)
The old AUC 0.926 used argmax binary labels. Must rebuild using:

Y_i(t) = 1[NLL(y_new) < NLL(y_true)]

Then rerun:
- First-failure censoring
- Nested age/control/exposure models
- Relation-fixed-effects model
- Leave-one-trajectory-out validation
- PR-AUC, log loss, calibration
- Exposure quartile survival curves
- All bootstrap intervals

The failure-curve per-case JSONs already contain NLL values — this is offline analysis.

**Do NOT quote AUC 0.926 until this is complete.**

---

## Two Inconsistencies to Investigate

### 1. MEMIT-Seq value discrepancy
| Source | MEMIT-Seq efficacy |
|---|---|
| Failure curve (seed 42, 5K) | 99.4% |
| Fixed-batch fb ordering (seed 42, 10K) | 97.5-98.8% |
| Original key_clustered/dispersed (seed 42, 5K) | 61.1-63.8% |

The 61-64% values are too different from 99.4%. Check:
- Algorithm configuration (lambda_prev, C0 weight)
- Checkpoint source
- Whether these were actually rescored with prob-pref or are still argmax
- Dataset slices used

**Do not use the 61-64% result until explained.**

### 2. Missing seed-42 early rescue values
The rescue table shows `--` for seed 42 suffix_random and suffix_spread. "Not available" ≠ "no detectable effect."

---

## Cross-Method Interpretation Correction

**Old claim:** "Methods that incorporate sequential history are robust, while AlphaEdit and EvoEdit do not."

**Problem:** EvoEdit explicitly incorporates previous modifications through sequential null-space alignment.

**Corrected claim:** "Different forms of historical protection have different robustness to temporal concentration. MEMIT-Seq and PathGuard are nearly invariant; EvoEdit reduces but does not eliminate the ordering effect; AlphaEdit is the most sensitive."

The one-seed cross-method table is supporting evidence, not a broad ranking conclusion. Keep in appendix or as one compact table in main paper.

---

## Revised Paper Structure (advisor's recommendation)

### 1. Introduction
- Aggregate efficacy hides which edits are lost
- Standard evaluations usually use one temporal order
- This paper studies individual survival and path dependence

### 2. Benchmark-Aligned Setup
- Define primary metric: NLL(y_new) < NLL(y_true)
- Define secondary strict metric: every target token is top-1
- State all headline numbers use probability preference

### 3. Selective Forgetting at Long Horizons
- 2K reproduction
- 10K failure curves
- 42.4pp age gap
- AlphaEdit selective vs MEMIT uniform degradation

### 4. Exposure Predicts First Failure
**Only after rerunning survival analysis with corrected labels**

### 5. Exposure Causes Immediate Damage
- Different-batch same-state A/B
- Same-fact/different-prompt A/B
- CLaRE comparison
- MEMIT-Seq attenuation

### 6. Identical Batches, Different Temporal Paths
- Fixed HIGH, LOW, three random paths
- Emphasize HIGH vs random
- Show all three seeds
- Explicitly show LOW is not universally protective

### 7. Method Dependence and Practical Implications
- Compact cross-method stress test
- Scheduler prevents worst path but doesn't beat random
- Random shuffling as strongest practical recommendation

### 8. Discussion and Limitations
- Three primary seeds
- Engineered paths
- No universal monotonic exposure schedule
- Prob-pref vs strict generation
- No strong suffix rescue
- Cross-method results mainly one seed

---

## Central Paper Claim (advisor's wording)

> Benchmark-aligned evidence that individual memory survival and long-horizon editing performance depend strongly on the temporal path, even when the facts and batches are held fixed.

This is cleaner than the previous rescue/phase-transition narrative.

---

## Source

Advisor feedback received 2026-09-09 in response to the prob-pref corrected results shared from `docs/probpref_results.md`.

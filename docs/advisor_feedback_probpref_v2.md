# Advisor Feedback v2: Detailed Assessment Post Prob-Pref Correction (2026-09-09)

Source: Second advisor review after seeing corrected numbers.

## Key External Validation

Our corrected numbers nearly reproduce published baselines:

| 10K CounterFact, Llama-3-8B | Our corrected | EvoEdit paper |
|---|---|---|
| AlphaEdit efficacy | 66.3% (3-seed mean) | 66.78% |
| MEMIT efficacy | ~47.5-55.6% by seed | 49.73% |
| EvoEdit efficacy | 98.9% (random0, s42) | 98.29% |

This confirms the evaluator is now aligned correctly.

## Important Nuances

### Argmax was not "wrong" — it was the wrong headline metric

- Prob-pref is correct for **CounterFact benchmark comparison**
- Argmax is a valid stricter **behavioral/generation metric**
- For zsRE, EvoEdit uses top-1 accuracy — metric is **benchmark-specific**
- The suffix rescue effect is not an "artifact" — it's a metric-sensitive finding:
  > "Suffix reordering affects strict target-generation behavior more than it affects the standard target-versus-original preference metric."

### The paper should report BOTH metrics

1. Prob-pref as primary benchmark-standard metric
2. Strict target generation as secondary operational metric

## Revised Story: The Strongest Contribution Changed

### OLD strongest result: suffix rescue / phase transition / timing window
### NEW strongest result: cross-algorithm order robustness

> "Standard random or canonical evaluation can make multiple methods look nearly saturated, while a controlled temporal-path stress test reveals large robustness differences."

This suggests **edit order is a hidden benchmark variable**.

## Cross-Algorithm Interpretation Corrections

### EvoEdit DOES incorporate history
EvoEdit uses sequential null-space alignment with accumulated edits (ACL 2026). Saying it "doesn't incorporate history" is wrong.

Correct framing:
> "Different forms of historical protection exhibit very different path robustness. Covariance/history-conditioned MEMIT-Seq and PathGuard remain stable, whereas AlphaEdit's static projector and EvoEdit's evolving-projector mechanism remain substantially order-sensitive."

### PathGuard is NOT SOTA on random-path efficacy
On random0/seed42: EvoEdit (98.9%) > MEMIT-Seq (98.8%) > PathGuard-poly2 (98.4%) > PathGuard (98.2%)

PathGuard's strength is **order-robustness + locality**, not headline efficacy.

### Poly2-hybrid is the strongest standard-metric candidate
99.0% eff, 91.5% para, 62.7% neigh vs published EvoEdit 98.29/91.21/63.91.
Numerically SOTA-comparable but needs multi-seed replication.

## What Must Be Retired

1. Existing PDF's evaluation section (explicitly states greedy/argmax)
2. First-failure AUC 0.926 and OR 0.284 (argmax labels)
3. Suffix rescue as main story
4. Displacement correlations (r=-0.833, partial r=-0.768) — recompute on prob-pref
5. "Poly2 becomes numerically unstable" → say "behavioral collapse" unless condition numbers confirm

## MEMIT-Seq Inconsistency (STILL UNRESOLVED)

- key_clustered/dispersed: 61-64% at 5K
- fb orderings: 97-99% at 10K
- failure curve: 93.6% at 10K

Must publish run manifests: edit-ID hash, batch-membership hash, order hash, method config, C0/ridge weights, evaluator commit, metric field.

## Revised Scoring

| Version | Score |
|---|---|
| Existing PDF with argmax claims | Not submission-ready |
| Corrected tables in old story | 6-7 |
| Full rewrite around age + order robustness | Strong 7 |
| + corrected first-failure + replicated cross-method | 8-capable |
| + held-out Poly2-hybrid/PathGuard validation | Clear 8 candidate |

## Recommended Title Direction

> "Edit Order Is a Hidden Axis of Sequential Knowledge Editing"

## Central Thesis

> Benchmark-standard aggregate scores obscure two forms of fragility: age-selective forgetting within a trajectory and large method-dependent sensitivity to temporal order. Controlled fixed-batch permutations reveal that AlphaEdit and EvoEdit can lose 14-27 points under concentrated paths despite near-saturated random-path performance, whereas MEMIT-Seq and PathGuard remain stable.

## Immediate Priorities

1. Recompute every label-dependent survival/mechanism analysis using prob-pref
2. Report both metrics side by side
3. Replicate cross-algorithm fixed-path on at least one holdout seed
4. Replicate Poly2-hybrid on untouched seeds
5. Remove rescue and old capacity narratives from main paper

## Evidence Sequence for Paper

1. Metric-correct reproduction (matches published AlphaEdit/EvoEdit)
2. Age-conditioned evaluation (42.4pp gap)
3. Fixed-batch temporal intervention (~22pp HI-RAND gap)
4. Random reference paths (tight 95-97% cluster)
5. Cross-method robustness (methods differ sharply under same stress)
6. Mechanism (rerun first-failure + retain logprob A/B)
7. Mitigation (Poly2-hybrid/PathGuard after holdout replication)

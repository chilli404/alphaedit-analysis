# Framing Notes — Advisor's Precise Wording Guidance

---

## Terms to USE vs AVOID

### USE:
- "partial prevention of further failure" (not recovery)
- "persistent prefix-induced disadvantage" (not irreversible damage)
- "time-sensitive intervention window" (not critical window)
- "abrupt regime transition" (not phase transition)
- "diminishing order-only recoverability" (not committed trajectory)
- "historical-key displacement" (not targeted displacement)
- "memory-relevant displacement" (not preferential interference)
- "path-level correlate/marker" (not mediator, for displacement)
- "observed random reference range" (not distribution, for 3 random paths)
- "CLaRE-inspired entanglement baseline" (not "CLaRE comparison")
- "history-aware sequential regularization" (not "C₀ addition")
- "abrupt destructive regime" (not catastrophic failure)
- "descriptive fraction of the gap" (not causal recovery percentage)

### AVOID:
- "irreversible" — can't prove no intervention recovers
- "phase transition" — no formal critical parameter demonstrated
- "critical point" — two branch points don't precisely locate a boundary
- "targeted displacement" — unrelated keys move MORE (ratio 0.75)
- "recovery" / "cure" — failed edits don't regain efficacy
- "committed trajectory" — implies formal irreversibility
- "adds massively" — say "strong path-level association"
- "the mechanism is complete" — say "the multiscale evidence is consistent"
- "only cosine differs" — batches also differ in relations, difficulty, etc.
- "five-path random distribution" — only three are random; HIGH/LOW are engineered

---

## The Multiscale Structure

The paper's central insight is that sequential interference has DIFFERENT signatures at different scales. This is a feature, not a weakness.

| Scale | Supported variable | What it predicts | Evidence |
|---|---|---|---|
| **Individual edit** | Future-key cosine | Which memory first fails | AUC 0.917, OR 0.836/+0.1 |
| **Incoming batch** | Selected key overlap | Immediate behavioral damage | 47/60 trials, p < 0.0001 |
| **Whole trajectory** | Historical-key displacement | Which paths approach collapse | r = −0.833, adds beyond N |
| **Intervention timing** | Prefix state | Whether reordering still helps | +13pp at 5K, none at 7K |

Do NOT force a single mediation chain:
```
cosine → displacement → per-edit damage   ← REJECTED by mediation test
```

Instead:
```
Edit scale: cosine → first failure risk
Path scale: displacement → regime identification  
These are RELATED but NON-EQUIVALENT phenomena.
```

The mediation null is expected under multilevel data. Cite the Arizona State multilevel-mediation reference.

---

## How to Frame Seed 137

Seed 137 is NOT an anomaly or a failure to replicate. It is **informative heterogeneity**.

- Seeds 42/2024: phase transition at 5-6K, catastrophic collapse
- Seed 137: smooth degradation, no differential ordering effect

Explanation: Seed 137's record pool produces lower unsigned displacement and more directional cancellation under the concentrated ordering. The concentrated prefix doesn't create the same accumulated stress.

Frame as: "The path dependence is seed-dependent, motivating analysis of the trajectory-level state variables that distinguish collapsing from surviving paths."

Seed 137 also serves as the **negative control** for the rescue experiment — there's nothing to rescue, and indeed the suffix switch has no effect (±2pp).

---

## How to Frame the Mediation Null

The trial-level mediation test failed (D → damage: r = +0.181). This is NOT a contradiction.

Frame as: "Historical-key displacement is a trajectory-state marker, not a local mediator of individual edit damage. The scale separation is expected: within a single model state, whether a batch displaces historical keys more does not predict whether it damages those keys more, because unsigned displacement discards directional information. Across complete paths, cumulative displacement integrates many factors into a summary of trajectory health."

---

## How to Frame the Rescue

The suffix-switch is **partial prevention with persistent prefix dependence**.

- 5K rescue: +13-15pp, from ~30% to ~45% (but full-random is ~88%)
- 7K rescue: no effect
- Gap closure: ~26% of descriptive gap

Key framing:
1. Future order IS causally consequential (the +13pp is real)
2. But the prefix leaves lasting damage that order-only correction can't undo
3. The intervention window narrows: 5K helps, 7K doesn't
4. Prevention (never concentrating) >> cure (switching after)

The rescue mainly PREVENTS additional failures, doesn't RECOVER lost ones:
- Survivor retention improves (9% → 21% for seed 42)
- Already-failed edits don't spontaneously recover

---

## Statistical Reporting Requirements

1. **Same-fact sign test:** Report **two-sided p = 0.043** unless one-sided was pre-specified
2. **Different-batch A/B:** Report **seed-clustered p = 0.036** (not naive p = 0.000039)
3. **Pooled A/B:** Report **sign test p = 1.2e-05** and **bootstrap CI [−0.005, −0.002]**
4. **Cosine ORs:** Always report per **+0.1 cosine** scale. State the scale explicitly.
5. **LOO-trajectory AUC** is primary for survival model. Edit-disjoint 5-fold is supplementary.
6. **Partial correlation p:** Verify whether 0.016 or 0.026 — document the exact test used.
7. **Random paths:** "Observed reference range" from 3 permutations, not a population distribution.
8. **Seed-specific results:** Always report per-seed effects, not only pooled.

---

## Main Text vs Appendix

### MAIN TEXT (9 pages):
1. Age-selective failure and immediate writability
2. First-failure model as PRIMARY survival analysis
3. Different-batch and same-fact A/B interventions
4. Fixed-batch HIGH/LOW/random path experiment
5. Temporal displacement and abrupt crossover
6. Early-versus-late suffix intervention
7. MEMIT-Seq attenuation
8. Compact GPT-J and zsRE scope results

### APPENDIX:
- Repeated retained/lost model (supplementary to first-failure)
- Alternative exposure metrics
- Approximate kernel η
- Signed-operator derivation
- CLaRE comparison
- Scheduler negative result
- Full seed-level and checkpoint tables
- Complete intervention manifests and hashes
- Mediation test results (with scale-separation explanation)

### REMOVE from old paper:
- Projection-capacity sweep from headline contribution
- "Two geometric bottlenecks"
- P × C₀ capacity narrative
- Claims that scheduling optimizes performance
- Claims that signed hazard is the superior general predictor
- Equal billing for update-space capacity story

---

## Recommended Thesis

> Sequential editing is causally path-dependent. At the edit scale, exposure to compatible future keys predicts which installed memories fail. At the batch scale, increasing overlap causes immediate behavioral damage, including when factual content is held fixed. At the trajectory scale, concentrating identical fixed batches can create an abrupt destructive regime outside ordinary random-order outcomes. Once a harmful prefix accumulates, changing only the remaining order can partially prevent further loss early, but becomes ineffective later. History-aware editing strongly attenuates the same immediate interference effect.

## Recommended Title

"Temporal Paths Determine Which Model Edits Survive"

or

"From Key Exposure to Path Collapse in Sequential Knowledge Editing"

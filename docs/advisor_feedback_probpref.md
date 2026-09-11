# Advisor Feedback on Prob-Pref Correction (2026-09-09)

## Key Verdicts

1. **Correction was necessary and correct.** Core contribution survives.
2. **Age-selective forgetting survives:** 42.4pp gap, monotonic, 3 seeds.
3. **Fixed-batch path result survives:** ~22pp HI-RAND gap, 3 seeds.
4. **A/B interventions unaffected:** continuous logprob, not binary.
5. **Suffix rescue is no longer a central result:** negligible under prob-pref.
6. **Phase-transition language must be softened.**
7. **First-failure survival model MUST be rebuilt** on prob-pref labels.
8. **MEMIT-Seq key_clustered/dispersed numbers (61-64%) need investigation** — too different from fb results (97-99%).
9. **Cross-method interpretation needs correction:** EvoEdit does incorporate history.
10. **Seed 137 LOW ≈ HIGH disproves "spread = safe"** — paper claim must change.

## Action Items

### P0: Must do before submission

- [ ] Rebuild first-failure survival model on prob-pref labels (CPU, offline from stored NLLs)
- [ ] Investigate MEMIT-Seq 61-64% discrepancy (key_clustered/dispersed vs fb orderings)
- [ ] Recompute displacement correlations with corrected retention variable
- [ ] Fix missing seed-42 suffix_random/suffix_spread evals (show as "not available" not "no effect")
- [ ] Correct EvoEdit characterization (it does incorporate history via sequential null-space alignment)

### P1: Paper framing changes

- [ ] Remove suffix rescue from main story → appendix metric comparison
- [ ] Remove phase-transition language → "sharp degradation" / "destructive path"
- [ ] Change "spread = safe" → "randomization is consistently robust; deterministic ordering is not universally safe"
- [ ] Add explicit dual-metric definition in Section 2
- [ ] Move cross-method table to appendix (one seed only)

### P2: Rewrite structure

1. Introduction: individual survival + path dependence
2. Benchmark-Aligned Setup: define both metrics, state prob-pref is primary
3. Selective Forgetting at Long Horizons: 2K reproduction + 10K age gap
4. Exposure Predicts First Failure: **only after survival reanalysis**
5. Exposure Causes Immediate Damage: A/B interventions
6. Identical Batches, Different Temporal Paths: fb experiment, all seeds, LOW not universally safe
7. Method Dependence: compact cross-method stress test
8. Discussion and Limitations

### Score Assessment

| State | Score |
|---|---|
| Old paper with argmax results | Not submission-ready |
| Corrected results, no survival reanalysis | 6-7 |
| Corrected survival model + focused rewrite | Strong 7, plausible 8 |
| + clear probability-level causal + rigorous cross-path | 8 defensible |

### Revised Thesis

> Benchmark-aligned evidence that individual memory survival and long-horizon editing performance depend strongly on the temporal path, even when the facts and batches are held fixed.

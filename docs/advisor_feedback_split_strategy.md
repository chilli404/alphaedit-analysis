# Advisor: Paper Split Strategy (2026-09-09)

## Verdict: Follow the two-paper split

- **ICLR 2027:** Diagnosis, evaluation, age-selective forgetting, temporal-path fragility
- **Later methods paper:** PathGuard (primary) + Poly2-Hybrid (complementary)

## ICLR Paper Scope

### KEEP in main paper
1. Benchmark-aligned 2K reproduction (5 seeds)
2. Age-selective forgetting (3 seeds, 42.4pp gap)
3. Fixed-batch temporal-path benchmark (HIGH/LOW/3×RANDOM, 3 seeds) — THE CENTERPIECE
4. Same-state and same-fact A/B interventions (continuous logprob, unaffected by metric change)
5. Cross-method stress test (AlphaEdit, EvoEdit, MEMIT-Seq — one seed, supporting evidence)
6. Practical benchmark recommendations

### KEEP in appendix
- Full 5-path per-seed curves
- Continuous margin regression details
- Same-fact prompt selection diagnostics
- Displacement vs norm (trajectory-level only)
- Scheduler negative result
- Suffix rescue (half page, metric-sensitive observation)
- CLaRE comparison
- Strict-generation secondary analysis
- Checkpoint manifests

### REMOVE completely
- PathGuard equations, implementation, results
- Poly2-Hybrid equations and results
- Adaptive Signed Repair
- Projection-capacity sweep
- P × C₀ factorial
- "Two geometric bottlenecks" framing
- PathGuard/Poly2 code from supplementary
- Any suggestion a new method has solved the problem

### Safe future-work sentence
> "These findings motivate editors that condition historical protection on temporal exposure and path state; developing and evaluating such methods is left to future work."

## Key Claim Changes

### OLD: "Cosine strongly predicts individual failure"
### NEW: "Age and installation strength dominate individual robustness; exposure exerts a small but directionally consistent local effect that compounds into large path-level differences"

### OLD: "EvoEdit doesn't incorporate history"
### NEW: "Different historical-preservation mechanisms exhibit sharply different robustness to temporal concentration"

## ICLR Paper Structure (9 pages)

1. **Introduction:** Standard single-stream evaluation hides temporal-path fragility
2. **Benchmark-Aligned Setup:** Define prob-pref primary, argmax secondary, age-conditioned retention
3. **Age-Selective Forgetting:** 5-seed 2K + 3-seed 10K + AlphaEdit vs MEMIT
4. **Local Exposure: Modest but Causal:** Margin model + A/B interventions. No claim cosine is dominant predictor.
5. **Edit Order Is a Hidden Benchmark Axis:** CENTERPIECE. Fixed-batch, all paths, all seeds.
6. **Path Fragility Is Method-Dependent:** AlphaEdit/EvoEdit/MEMIT-Seq. One-seed limitation stated.
7. **Implications and Benchmark Recommendations:** Report mean, worst-path, variance, oldest-cohort.
8. **Related Work and Limitations**
9. **Conclusion:** "The edit set does not uniquely determine the edited model; temporal path is an unreported but consequential benchmark variable."

## Future PathGuard Paper

### Thesis
> Robust sequential editing requires controlling geometry at two timescales: protecting historical directions exposed to the incoming batch and stabilizing crowded geometry within the incoming batch.

### Method hierarchy
1. PathGuard: primary algorithm
2. Poly2-Hybrid: complementary current-batch regularizer
3. PathGuard-Hybrid: complete model

### Essential holdout experiment
Freeze everything from seed 42, then run full factorial on untouched seeds:
{linear, Poly2} × {without PathGuard, with PathGuard}

### Claim
> Strong temporal-path robustness (not general SOTA)
Report robustness frontier: mean efficacy vs worst-path efficacy

## Timeline Constraints
- ICLR abstract deadline: September 18, 2026
- ICLR paper deadline: September 25, 2026
- 9 pages main text
- No new authors after abstract deadline
- Do NOT submit PathGuard paper while ICLR is under review

## Score Assessment
| Dimension | Score |
|---|---|
| Contribution | 3-4/4 |
| Soundness | 3/4 |
| Presentation potential | 4/4 |
| Most likely | **7** |
| Favorable reviewer | **8** |

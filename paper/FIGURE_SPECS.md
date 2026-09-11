# Figure Specifications

---

## Figure 1: Evidence Ladder Overview (schematic)

**Type:** Diagram, not data-driven. Illustrate the argument structure.

**Panels:**
- Selective failure → prediction → immediate causality → path causality → prospective control → method boundary

**Visual:** Arrow diagram or stacked panels showing the six-level evidence hierarchy.

**Message:** The paper progresses from observation through prediction to controlled intervention.

---

## Figure 2: Age-Selective Failure

**Panel A: Age-by-checkpoint heatmap (MCF, seed 42)**
- X-axis: checkpoint (2K, 3K, 5K, 7K, 10K)
- Y-axis: per-1K cohort (0–1K through 9K–10K)
- Color: efficacy (green=high, red=low)
- Data: `results/failure_curve_checkpointed/seed42/` per-case JSONs, or pre-aggregated in `results/paper_numbers.json` → `age_selective`

**Panel B: Three-seed cohort retention at 10K**
- X-axis: cohort age (0–1K is oldest)
- Y-axis: efficacy
- Three lines (seeds 42, 2024, 137) plus mean
- Data: same as above

**Panel C (inset): Cross-dataset + cross-model**
- Compact bar or small multiples: MCF (52pp gap), zsRE (10.7pp gap), GPT-J (44.6pp gap)
- Message: pattern generalizes

---

## Figure 3: Controlled Interventions

**Panel A: Different-batch A/B (30 trials)**
- Paired dot plot or violin: HIGH vs LOW logprob damage per trial
- Color by seed
- Annotate: 26/30, p = 0.036 (seed-clustered)
- Data: `results/logit_damage/seed{42,2024,137}/intervention_results.json`

**Panel B: Same-fact A/B (30 trials)**
- Same format as Panel A
- Annotate: 21/30, p = 0.043 (two-sided)
- Smaller effect but same direction
- Data: `results/same_fact_damage/seed{42,2024,137}/intervention_results.json`

**Panel C: MEMIT-Seq attenuation**
- Grouped bar: AlphaEdit vs MEMIT-Seq effect size per seed
- Annotate: 76% attenuation pooled
- Data: `results/logit_damage_memit_seq/seed{42,2024,137}/intervention_results.json`

---

## Figure 4: Fixed-Batch Temporal Paths (CENTERPIECE)

**Panel A: Retention trajectory curves**
- X-axis: edit count (1K → 10K)
- Y-axis: first-1K retention
- 4 curves per seed: Continue-HIGH, suffix-5K, suffix-7K, full-random
- Vertical dashed lines at 5K and 7K branch points
- Before branch: curves are identical (same checkpoint)
- After branch: curves diverge
- Data: `paper/data/temporal_fb_high_exposure_s{42,2024,137}.json` (intermediate checkpoints), plus suffix rescue from `paper/data/suffix_*_eval.json`
- **For seed 42 this is the most dramatic visualization**

**Panel B: Random path reference cloud**
- Scatter: x = realized exposure score, y = endpoint efficacy
- Points: all 15 paths (5 per seed), colored by seed
- HIGH paths are clear outliers below the random cloud
- Data: from the 5-path dose-response table

**Panel C: Historical displacement as early warning**
- Overlay: displacement at midpoint (batch 49) vs retention at 10K
- Show that displacement is already elevated for HIGH paths at 5K, before the behavioral crossover
- Data: `results/signed_displacement_analysis.json`, `results/norm_vs_displacement.json`

---

## Figure 5: Suffix Rescue Detail

**Panel A: 4-curve rescue plot (seed 42)**
- X-axis: edit count (1K → 10K)
- Y-axis: first-1K retention
- Curves: Continue-HIGH, switch-at-5K (random suffix), switch-at-7K (random suffix), full-random
- Vertical lines at 5K and 7K
- Identical before each branch point, then diverge
- Data: temporal checkpoints + rescue evals

**Panel B: Survivor-conditioned analysis**
- Bar chart: survivor retention rate (alive at 5K → still alive at 10K)
- Continue: 9.4% | Rescue: 21.2%
- Message: rescue doubles survivor retention but most still fail
- Data: `results/rescue_analysis.json`

**Panel C: All seeds × both branch points**
- Small multiples grid: 3 seeds × {5K, 7K} × {continue, random, spread}
- Compact table or heatmap format
- Data: suffix rescue results

---

## Notes for all figures:
- Use consistent color scheme across figures (same seed = same color)
- Seed 137 should be visually distinguishable (e.g., dashed line) since it's the non-collapsing control
- Error bars where applicable should be seed-level, not edit-level
- All axis labels should include units and scale

# Results Manifest — Paper Writing Reference

All paths relative to project root. Numbers extracted directly from result files.

---

## 1. Age-Selective Forgetting

**Tests:** Aggregate efficacy hides severe age-dependent memory loss.

**Files:**
- `results/failure_curve_checkpointed/seed{42,2024,137}/` — per-case JSONs at each checkpoint
- `results/paper_numbers.json` → `age_selective` section
- `results/numerical_audit.json` — reconciliation of latest-1K numbers

**Key numbers (3-seed mean at 10K, MCF, Llama-3-8B):**
- Newest 1K cohort: 59.1% efficacy
- Oldest 1K cohort: 7.1% efficacy
- **Age gap: 52.0pp**, monotonic across all 10 cohorts
- Per-seed range: 46.6–60.9pp

**Cross-validation:**
- zsRE (seed 42, 10K): overall 95.5%, age gap 10.7pp
- GPT-J (seed 42, 10K): overall 75.9%, age gap 44.6pp

**Supported claim:** "Aggregate efficacy conceals strong age-dependent memory loss; the gradient is monotonic across seeds, datasets, and architectures."

**AVOID:** Using latest-1K at 10K (59.1%) to claim "local plasticity remains intact." The latest 1K has endured up to 999 subsequent edits. Use immediate installation efficacy for plasticity claims.

---

## 2. First-Failure Survival Model

**Tests:** Key-cosine exposure predicts which edit first fails, under proper censoring.

**Files:**
- `results/survival_table.json`
- `results/figures/paper/interference_panel_results.json`

**Key numbers:**
- M1 (age + checkpoint): LOO-trajectory AUC = 0.838
- M4 (+ cosine + interference): LOO-trajectory AUC = **0.917**
- M5 (+ relation FE): LOO-trajectory AUC = **0.932**
- Cosine OR per +0.1: **0.836** (M4) / **0.882** (M5)
- M5 cosine p = 7.49e-19
- Trajectory bootstrap: OR median 0.149, 95% CI [0.063, 0.216], 100% sign consistent

**Supported claim:** "Checkpoint-censored future-key exposure predicts individual first failure with AUC 0.917 under leave-one-trajectory-out evaluation."

**AVOID:** Calling the repeated-measures AUC (0.944) primary. First-failure is the primary model. Edit-disjoint 5-fold AUC is supplementary.

---

## 3. Pooled A/B Intervention (Different-Batch + Same-Fact)

**Tests:** HIGH-overlap batches cause more immediate damage, combining both intervention types.

**Files:**
- `results/pooled_ab_results.json`
- `paper/data/pooled_ab_results.json` (copy)

**Key numbers (60 trials: 30 different-batch + 30 same-fact):**
- Sign count: **47/60** trials HIGH more damaging
- Sign test p = **1.21e-05**
- Bootstrap CI: **[−0.0051, −0.0020]**, P(negative) = 1.0
- Different-batch mean: −0.0048 (26/30)
- Same-fact mean: −0.0020 (21/30)
- Mean cosine gap: 0.056

**Supported claim:** "Across 60 same-state intervention trials pooling both different-batch and same-fact designs, 47 show greater damage under higher overlap (p < 0.0001, bootstrap CI entirely below zero)."

**AVOID:** Reporting the naive 30-trial t-test (p = 0.000039). Use seed-clustered p = 0.036 for the different-batch experiment alone.

---

## 4. Same-Fact Intervention

**Tests:** Same subjects, same relations, same targets — only prompt-induced key geometry differs.

**Files:**
- `results/same_fact_damage/seed{42,2024,137}/intervention_results.json`
- `paper/data/same_fact_seed{42,2024,137}.json` (copies)

**Key numbers:**
- Seed 42: mean Δ = −0.00246, 7/10 trials
- Seed 2024: mean Δ = −0.00244, 8/10 trials
- Seed 137: mean Δ = −0.00123, 6/10 trials
- Pooled: **21/30 trials**, p = **0.043 two-sided** (0.021 one-sided)

**Supported claim:** "A content-controlled intervention in which only prompt-induced key representation varies produces a smaller but directionally consistent geometric effect (21/30 trials, p = 0.043 two-sided)."

**AVOID:** Reporting one-sided p = 0.021 unless the directional hypothesis was explicitly pre-specified. Default to two-sided.

---

## 5. Fixed-Batch Path Dependence (Dose-Response)

**Tests:** Identical batches yield different outcomes through temporal sequence alone.

**Files:**
- `results/matched_ordering/AlphaEdit/fb_{high,low,random0,random1,random2}_exposure/seed{42,2024,137}/full_eval_seed{N}.json`
- S3-synced to `/tmp/` or `paper/data/`

**Key numbers (overall efficacy at 10K):**

| | HIGH | RAND0 | RAND1 | RAND2 | LOW |
|---|---|---|---|---|---|
| s42 | 0.325 | 0.879 | 0.851 | 0.882 | 0.851 |
| s2024 | 0.294 | 0.876 | 0.876 | 0.874 | 0.861 |
| s137 | 0.676 | 0.819 | 0.820 | 0.853 | 0.697 |

- Seeds 42/2024: HIGH is catastrophic outlier (~30% vs ~87% random)
- Seed 137: HIGH below random but no catastrophic collapse

**Supported claim:** "Reordering identical fixed batches produces stable or catastrophic trajectories; HIGH-exposure orderings fall far below the observed random-order reference range."

**AVOID:** Calling the three random paths a "full permutation distribution." Say "observed random reference range" (three permutations per seed).

---

## 6. Temporal Trajectories (Phase Transition)

**Tests:** Displacement diverges before retention collapses.

**Files:**
- `results/matched_ordering/AlphaEdit/fb_high_exposure/seed{42,2024,137}/full_eval_seed{N}.json` (with intermediate checkpoints)
- `paper/data/temporal_fb_high_exposure_s{42,2024,137}.json`

**Key numbers (first_1k retention under HIGH ordering):**

| Edits | s42 | s2024 | s137 |
|---|---|---|---|
| 1K | 0.968 | 0.984 | 0.259 |
| 3K | 0.883 | 0.913 | 0.374 |
| 5K | 0.800 | 0.752 | 0.455 |
| 6K | 0.589 | 0.518 | 0.434 |
| 8K | 0.140 | 0.361 | 0.402 |
| 10K | 0.075 | — | — |

- Seeds 42/2024: HIGH better than LOW through ~5K, then abrupt crossover and collapse
- Seed 137: HIGH and LOW track within ±5pp throughout — no collapse

**Supported claim:** "The destructive regime emerges abruptly between 5K–7K edits. Historical-key displacement is already elevated at 5K, before the behavioral crossover."

**AVOID:** "Phase transition" — use "abrupt regime transition" or "abrupt crossover." "Critical point" — use "time-sensitive intervention window."

---

## 7. Displacement vs Norm Decomposition

**Tests:** Historical-key displacement adds beyond global update norm.

**Files:**
- `results/norm_vs_displacement.json`
- `paper/data/norm_vs_displacement.json` (copy)

**Key numbers (9 trajectories):**
- D (historical displacement) → retention: r = **−0.833**, p = **0.005**
- N (global norm) → retention: r = −0.583, p = 0.099 (NS)
- D controlling for N: partial r = **−0.768**, p = **0.016**
- N controlling for D: partial r = −0.057, p = 0.884 (nothing left)
- D_hist/D_unrel ratio: **0.75** (historical keys displaced LESS than unrelated)

**Supported claim:** "Historical-key displacement predicts trajectory-level retention substantially better than global update norm and retains a strong association after controlling for norm."

**AVOID:** "Targeted displacement" or "preferential interference with historical keys" — unrelated keys move MORE. Say "historical-key displacement" or "memory-relevant displacement." Do NOT say "global norm plays no role" — it may still operate through displacement as a mediator. Say "D statistically subsumes N in these trajectories." The advisor noted the partial correlation p may be ~0.026 not 0.016 under conventional t-approximation — verify the exact test used and document it.

---

## 8. Signed Displacement (Seed 137 Explanation)

**Tests:** Why seed 137 doesn't collapse despite similar exposure.

**Files:**
- `results/signed_displacement_analysis.json`
- `results/signed_all_trajectories.json`

**Key numbers (under fb_high_exposure):**

| Metric | Seed 42 (collapses) | Seed 137 (survives) |
|---|---|---|
| Unsigned damage | 1.303 | 1.039 |
| Signed damage | 0.212 | 0.093 |
| Cancellation factor | 16.3% | 9.0% |

**Important:** Across ALL 9 trajectories, unsigned displacement (r = −0.833) is the stronger predictor, not cancellation factor (r = −0.483, p = 0.19). Signed direction is a **secondary** within-ordering explanation.

**Supported claim:** "Seed 137's concentrated ordering produces less unsigned displacement and more directional cancellation than seed 42, consistent with the absence of catastrophic collapse."

**AVOID:** Making signed cancellation the headline mechanism. It explains the seed-42-vs-137 pair but does not dominate across all trajectories.

---

## 9. MEMIT-Seq A/B Attenuation

**Tests:** History-aware editing reduces the immediate interference effect.

**Files:**
- `results/logit_damage_memit_seq/seed{42,2024,137}/intervention_results.json`
- `paper/data/memit_seq_seed{42,2024,137}.json`

**Key numbers:**

| Seed | AlphaEdit | MEMIT-Seq | Attenuation |
|---|---|---|---|
| 42 | −0.00656 (9/10) | −0.00027 (5/10) | 95.9% |
| 2024 | −0.00457 (8/10) | −0.00061 (7/10) | 86.7% |
| 137 | −0.00335 (9/10) | −0.00261 (7/10) | 22.3% |
| **Pooled** | **−0.00483 (26/30)** | **−0.00116 (19/30)** | **76.0%** |

- MEMIT-Seq sign test: 19/30, p ≈ 0.10 (not individually significant)

**Supported claim:** "MEMIT-Seq substantially attenuates the immediate exposure effect (~76% reduction in mean effect size), though the attenuated effect does not reach individual significance."

**AVOID:** Attributing attenuation specifically to C₀, history, or projection removal — MEMIT-Seq changes all three jointly. Use "history-aware sequential regularization."

---

## 10. Suffix-Switch Rescue Experiment

**Tests:** Can changing future order from the same pre-collapse state avert collapse?

**Files:**
- `/tmp/rescue_suffix_{random,spread,random_late,spread_late}_s{42,2024,137}.json`
- `paper/data/suffix_*_eval.json` (copies)
- `results/rescue_analysis.json`

**Key numbers (overall / first_1k at 10K):**

| Branch | Continue-HIGH | Random-suffix | Spread-suffix | Full-random |
|---|---|---|---|---|
| **5K s42** | 0.325/0.075 | **0.451/0.170** | 0.346/0.103 | 0.879/0.661 |
| **5K s2024** | 0.294/0.112 | **0.435/0.169** | 0.395/0.182 | 0.876/0.649 |
| 5K s137 | 0.676/0.385 | 0.684/0.387 | 0.666/0.378 | 0.819/0.522 |
| 7K s42 | 0.325/0.075 | 0.259/0.066 | 0.253/0.079 | — |
| 7K s2024 | 0.294/0.112 | 0.394/0.131 | 0.386/0.147 | — |
| 7K s137 | 0.676/0.385 | 0.675/0.375 | 0.691/0.393 | — |

- Early rescue: **+12.6pp** (s42) and **+14.1pp** (s2024) for random-suffix
- Gap closure: ~26% of the descriptive gap to full-random: (45−30)/(88−30) ≈ 0.26
- Late rescue: **no detectable effect** on any seed
- Rescue = prevention of further failure, not recovery (survivor retention: 9% → 21% for s42)
- Exposure contrast is equivalent at both branch points (~1.12×)

**Supported claim:** "Starting from identical high-concentration checkpoints and holding remaining batches fixed, changing only suffix order improves endpoint efficacy when applied at 5K (+13–15pp) but not at 7K. The intervention mainly preserves edits still alive at the branch; it does not restore memories already lost."

**AVOID:**
- "Irreversible damage" → "persistent prefix-induced disadvantage"
- "Critical window" → "time-sensitive intervention window"
- "Recovery" or "cure" → "partial prevention of further failure"
- "Committed trajectory" → "diminishing order-only recoverability"

---

## 11. Random Path Distribution

**Tests:** Are HIGH/LOW paths outside normal random variation?

**Files:** Same as §5 above.

**Key numbers:** Random paths cluster at 0.82–0.88 overall efficacy. HIGH (0.29–0.33) is far below. LOW (0.85) is at the bottom of the random range.

**Supported claim:** "The engineered HIGH-exposure ordering falls well outside the observed range of three independently generated random permutations per seed."

**AVOID:** "Five-path random distribution" — HIGH and LOW are engineered, not random. Say "three random permutations per seed, plus two engineered intervention paths."

---

## 12. CLaRE Comparison + Interference Kernel

**Tests:** How does our predictor compare to CLaRE-style entanglement?

**Files:**
- `results/clare_comparison.json`
- `results/interference_kernel/kernel_vs_cosine_comparison.json`

**CLaRE:**
- A/B ranking accuracy: max_cosine 22/30 = CLaRE 22/30
- Correlation between predictors: r = 0.24

**Kernel η:**
- Baseline AUC: 0.832 | Cosine: 0.845 | η: 0.835 | Both: 0.846
- η adds to cosine: p = 0.017

**Supported claim:** "A CLaRE-inspired entanglement baseline achieves equivalent aggregate branch-ranking accuracy on this task. The approximate solve-aware kernel adds complementary information (p = 0.017) but does not supersede cosine."

**AVOID:** Calling CLaRE implementation the "complete CLaRE procedure" — call it "CLaRE-inspired entanglement baseline." Equal 22/30 is not "statistical equivalence" — it's matched aggregate performance with wide uncertainty.

---

## Scheduler Comparison (APPENDIX)

**Files:** `results/matched_ordering/AlphaEdit/sched_*/seed{42,2024,137}/full_eval_seed{N}.json`

| Strategy | Overall (mean) | First 1K |
|---|---|---|
| high_exposure | 0.432 | 0.191 |
| exposure_only | 0.835 | 0.567 |
| conditioning_only | 0.836 | 0.587 |
| balanced | 0.846 | 0.582 |
| random0 | 0.858 | 0.611 |
| low_exposure | 0.803 | 0.495 |

**Appendix result:** Schedulers prevent catastrophe but don't beat random. Present as honest negative result.

---

## Mediation Test (APPENDIX)

**Files:** `results/mediation_results.json`

- Path level: D predicts collapse (r = −0.833) — marker, not mediator
- Trial level: D does NOT mediate (r = +0.181) — mediation breaks
- Edit level: r = −0.015

**Present as:** "Different scales of sequential interference reveal different signatures. This is expected under multilevel data."

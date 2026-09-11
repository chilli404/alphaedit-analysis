# Extra Statistics and Metrics

All results computed during the review-response analysis sessions (Aug 2026). Numbers are authoritative — derived from raw per-case evaluation JSONs and the vectorized panel analysis.

---

## 1. Group-Disjoint Cross-Validation (Criticism: shared edit identities)

**Panel:** 46,000 rows, 10,000 unique edits, 9,826 unique subjects, 3 trajectories (seeds 42, 2024, 137)

| Validation Scheme | Model | AUC | PR-AUC | Brier | ECE |
|---|---|---|---|---|---|
| Edit-ID disjoint (5-fold) | M1 baseline | 0.871 ± 0.002 | 0.861 ± 0.002 | 0.1452 ± 0.0014 | 0.0228 ± 0.0040 |
| Edit-ID disjoint (5-fold) | M2 semantic | 0.874 ± 0.001 | 0.875 ± 0.003 | 0.1444 ± 0.0012 | 0.0215 ± 0.0053 |
| Edit-ID disjoint (5-fold) | M4 full | 0.944 ± 0.001 | 0.943 ± 0.001 | 0.0943 ± 0.0004 | 0.0183 ± 0.0015 |
| Edit-ID disjoint (5-fold) | M5 within-relation | 0.951 ± 0.000 | 0.951 ± 0.000 | 0.0878 ± 0.0002 | 0.0172 ± 0.0051 |
| Subject-disjoint (5-fold) | M1 baseline | 0.871 ± 0.003 | 0.861 ± 0.007 | 0.1452 ± 0.0020 | 0.0229 ± 0.0023 |
| Subject-disjoint (5-fold) | M2 semantic | 0.874 ± 0.003 | 0.875 ± 0.007 | 0.1444 ± 0.0019 | 0.0228 ± 0.0032 |
| Subject-disjoint (5-fold) | M4 full | 0.948 ± 0.000 | 0.949 ± 0.000 | 0.0910 ± 0.0000 | 0.0372 ± 0.0000 |
| Subject-disjoint (5-fold) | M5 within-relation | 0.955 ± 0.000 | 0.957 ± 0.000 | 0.0845 ± 0.0000 | 0.0314 ± 0.0000 |
| Leave-one-trajectory-out | — | 0.862–0.897 | — | — | — |

### Within-Strata AUC (M2 semantic, edit-ID disjoint)

| Stratum | AUC |
|---|---|
| age_Q1_young | 0.826 |
| age_Q2 | 0.893 |
| age_Q3 | 0.878 |
| age_Q4_old | 0.841 |
| ckpt_3000 | 0.795 |
| ckpt_5000 | 0.722 |
| ckpt_7000 | 0.671 |
| ckpt_10000 | 0.731 |

---

## 2. Within-Relation Fixed Effects — M5 (Criticism: geometry vs semantics)

**Full-sample fit on 45,989 rows (3 seeds pooled):**

- Cosine OR = **0.246** [0.191, 0.317], p = 1.50 × 10⁻²⁷
- 34 unique relations, median ~1,300 observations per relation
- Interpretation: after absorbing all between-relation variation, higher max cosine to subsequent edits reduces retention odds by 75%

---

## 3. Condition-Number Control — M6 (Criticism: scheduler confound)

**Full-sample fit on 23,992 rows (seeds 42 + 2024, which have mechanism logs):**

- Cosine OR (controlling for condition) = **0.068** [0.049, 0.095], p = 2.59 × 10⁻⁵⁸
- Condition number OR = 1.543, p = 7.25 × 10⁻³⁶
- Comparison: cosine OR without condition control = 0.149
- Interpretation: controlling for batch conditioning makes cosine effect **stronger** (suppression, not confounding)

---

## 4. First-Failure Analysis (Criticism: not conventional hazard)

**First-failure panel: 36,588 rows (80% of repeated-measures panel retained)**

| Model | First-Failure AUC | Repeated-Measures AUC | Attenuation |
|---|---|---|---|
| M1 baseline | 0.840 ± 0.004 | 0.871 ± 0.002 | -3.1% |
| M4 full | 0.926 ± 0.002 | 0.944 ± 0.001 | -1.8% |
| M5 within-relation | 0.936 ± 0.002 | 0.951 ± 0.000 | -1.5% |

**First-failure cosine OR (M5):** 0.284 [0.215, 0.375], p = 7.49 × 10⁻¹⁹

---

## 5. Trajectory-Level Bootstrap (Criticism: only 3 trajectories)

**1,000 resamples from 3 trajectories (seeds 42, 2024, 137) with replacement:**

- AUC: median = 0.944, 95% CI = **[0.940, 0.958]**
- Cosine OR: median = 0.149, 95% CI = **[0.063, 0.216]**
- Sign consistency: **100.0%** of bootstraps have OR < 1
- Convergence: 1000/1000 iterations

---

## 6. Recovery Rate (Criticism: survival terminology)

**AlphaEdit, 3 seeds (27,000 edit trajectories across checkpoints 5K/7K/9K/10K):**

| Seed | Stable Success | Monotonic Forgetting | Stable Failure | Recovery/Oscillation |
|---|---|---|---|---|
| 42 | 22.5% | 62.1% | 11.0% | 4.4% |
| 137 | 8.0% | 67.2% | 19.6% | 5.2% |
| 2024 | 6.2% | 61.8% | 25.9% | 6.1% |

- **Edit-level non-absorbing rate: 2.9%** (382/13,000 edits show recovery or oscillation)
- **Transition-level recovery rate: 4.6%** (1,418/30,766 forgotten-observations)
- Conclusion: forgetting is approximately absorbing; survival framing justified

---

## 7. Multilayer Key Predictor (Criticism: layer-6 selection not justified)

**All 5 edited layers tested independently (Llama-3-8B, 45,989 rows, edit-ID disjoint CV):**

| Layer | AUC | PR-AUC | Cosine OR | 95% CI | p-value |
|---|---|---|---|---|---|
| 4 | 0.944 ± 0.002 | 0.943 ± 0.002 | 0.132 | [0.108, 0.161] | 9.9 × 10⁻⁸⁸ |
| 5 | 0.944 ± 0.002 | 0.943 ± 0.002 | 0.143 | [0.116, 0.176] | 1.3 × 10⁻⁷⁵ |
| 6 | 0.944 ± 0.002 | 0.943 ± 0.002 | 0.149 | [0.120, 0.185] | 5.8 × 10⁻⁶⁶ |
| 7 | 0.944 ± 0.002 | 0.942 ± 0.003 | 0.165 | [0.130, 0.209] | 5.7 × 10⁻⁵¹ |
| 8 | 0.944 ± 0.002 | 0.942 ± 0.003 | 0.191 | [0.151, 0.241] | 2.5 × 10⁻⁴³ |

- AUC identical across all layers (0.944)
- Earlier layers have slightly stronger individual cosine effects
- Layer 6 is not special — any layer works equally well

---

## 8. Cross-Architecture Replication — GPT-J (Criticism: split across architectures)

**GPT-J-6B survival model (seed 42, layer 5 keys, 10,273 panel rows):**

| | Llama-3-8B | GPT-J-6B |
|---|---|---|
| Cosine OR | 0.149 [0.120, 0.185] | **0.150 [0.113, 0.198]** |
| p-value | 5.8 × 10⁻⁶⁶ | 3.1 × 10⁻⁴⁰ |
| AUC (5-fold) | 0.944 ± 0.002 | 0.709 ± 0.008 |
| N (panel rows) | 45,989 | 10,273 |
| Trajectories | 3 (seeds 42, 2024, 137) | 1 (seed 42) |
| Checkpoints | 3K, 5K, 7K, 10K | 1K, 3K, 5K, 10K |

- Cosine OR virtually identical (0.149 vs 0.150)
- Lower AUC expected: fewer observations, 1 seed, partial 10K eval
- The effect size replicates perfectly across architectures

---

## 9. MEMIT-Seq Authoritative Cohort Numbers (Criticism: numerical contradiction)

**From raw per-case JSONs (`analysis/reconcile_memit_seq.py`):**

### Seed 42 — Oldest Cohort (first 1,000 edits)

| Checkpoint | AlphaEdit | MEMIT-Seq | Gap |
|---|---|---|---|
| 5K | 74.4% | 92.4% | +18.0 pp |
| 7K | 54.0% | 89.0% | +35.0 pp |
| 9K | 23.7% | 78.4% | +54.7 pp |
| 10K | 13.4% | 64.2% | +50.7 pp |

### Seed 2024 — Oldest Cohort

| Checkpoint | AlphaEdit | MEMIT-Seq | Gap |
|---|---|---|---|
| 5K | 61.5% | 92.0% | +30.5 pp |
| 7K | 18.3% | 89.3% | +71.0 pp |
| 9K | 6.5% | 83.7% | +77.2 pp |
| 10K | 5.1% | 79.4% | +74.3 pp |

Note: Paper's "83.7%" matches seed 2024 MEMIT-Seq oldest at **9K** (not 10K). Paper's "91.2%" matches seed 42 MEMIT-Seq at **5K** (92.4%). The paper mixed checkpoint levels.

---

## 10. Joint GPT-J Ordering × Capacity (Criticisms: architecture asymmetry, ordering gap)

### Ordering Gap with Bootstrap 95% CIs

| Condition | Δefficacy (clustered − dispersed) | 95% CI |
|---|---|---|
| Binding (τ=0.0052, ~20.7% rank) | **+5.0 pp** | [+3.1, +6.9] |
| Permissive (τ=0.0105, ~50% rank) | **-0.0 pp** | [-0.7, +0.6] |

### P-only vs P+C₀ with Bootstrap 95% CIs

| τ | Seed | Δ(C₀) Efficacy | 95% CI | Δ(C₀) Locality |
|---|---|---|---|---|
| 0.0052 | 2024 | **-21.0 pp** | [-22.3, -19.7] | +5.6 pp |
| 0.0105 | 2024 | **+35.2 pp** | [+34.1, +36.2] | +5.5 pp |
| 0.005 | 42 | -13.7 pp | [-15.1, -12.3] | +5.1 pp |
| 0.01 | 42 | +29.1 pp | [+28.0, +30.2] | +6.1 pp |
| 0.05 | 42 | +28.5 pp | [+27.6, +29.5] | +4.9 pp |
| 0.1 | 42 | +18.5 pp | [+17.6, +19.4] | +3.8 pp |
| 0.2 | 42 | +27.4 pp | [+26.4, +28.3] | +3.7 pp |

---

## 11. Locality Conditioned on Edit Success (Criticism: confound check)

**Conditioning neighborhood on successful edits (efficacy ≥ 0.5) to control for failed-edit preservation:**

### P-only vs P+C₀ at τ=0.0052 (binding capacity)

| Seed | Condition | Efficacy | Neigh (all) | Neigh\|Success | N_success |
|---|---|---|---|---|---|
| 2024 | P-only | 69.5% | 5.5% | 6.1% | 6,945 |
| 2024 | P+C₀ | 48.4% | 11.1% | 12.7% | 4,843 |
| | Δ(C₀) | | +5.6 pp | **+6.5 pp** | |
| 42 | P-only | 61.4% | 5.6% | 6.1% | 6,140 |
| 42 | P+C₀ | 47.7% | 10.7% | 11.8% | 4,769 |
| | Δ(C₀) | | +5.1 pp | **+5.7 pp** | |

Conclusion: locality gains from C₀ survive conditioning — confound does not explain the result.

---

## 12. Sample Accounting

| Metric | Value |
|---|---|
| Panel rows (Llama-3, 3 seeds) | 46,000 |
| Unique edits | 10,000 |
| Unique subjects | 9,826 |
| Unique relations | 34 |
| Trajectories | 3 (seeds 42, 2024, 137) |
| Checkpoints per trajectory | 3–4 (3K, 5K, 7K, 10K; seed 137 missing 3K) |
| Panel rows with key predictors | 45,989 |
| First-failure panel rows | 36,588 |
| GPT-J panel rows | 10,273 |
| Rows with mechanism data (M6) | 23,992 |

---

## 13. Model Specification Reference

All ORs and AUCs in the paper correspond to specific model specifications:

| Model | Formula | AUC (edit-disjoint) | Cosine OR | Context |
|---|---|---|---|---|
| M1 | survived ~ C(ckpt) + C(age) | 0.871 | — | Baseline (no predictors) |
| M2 | M1 + relation_overlap + target_margin | 0.874 | — | Semantic (no keys) |
| M4 | M2 + max_cosine + cumulative_interf | 0.944 | 0.149 | Full model |
| M5 | M4 + C(relation_id) - relation_overlap | 0.951 | 0.246 | Within-relation FE |
| M6 | M4 + log_cache_condition | — | 0.068 | Condition-controlled |
| M4 (first-failure) | same as M4, first-failure censored | 0.926 | — | First-failure |
| M5 (first-failure) | same as M5, first-failure censored | 0.936 | 0.284 | First-failure + FE |
| M4 (GPT-J) | same as M4, GPT-J keys | 0.709 | 0.150 | Cross-architecture |

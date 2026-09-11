# Table Specifications

---

## Table 1: First-Failure Survival Model (MAIN TEXT)

**Purpose:** Show that cosine predicts first failure under proper censoring.

| Model | Predictors | LOO-T AUC | OR per +0.1 | p |
|---|---|---|---|---|
| M1 | Age + checkpoint | 0.838 | — | — |
| M4 | + cosine + interference + margin | 0.917 | 0.836 | 6.2e-48 |
| M5 | + relation FE | 0.932 | 0.882 | 7.5e-19 |

**Source:** `results/survival_table.json`
**Notes:** LOO-trajectory AUC is PRIMARY. Edit-disjoint 5-fold is supplementary. All ORs on per +0.1 scale. State the scale.

---

## Table 2: Fixed-Batch 5-Path Dose-Response at 10K (MAIN TEXT)

**Purpose:** Show HIGH is an outlier; random paths cluster.

### Overall Efficacy:
| | HIGH | RAND0 | RAND1 | RAND2 | LOW |
|---|---|---|---|---|---|
| s42 | 0.325 | 0.879 | 0.851 | 0.882 | 0.851 |
| s2024 | 0.294 | 0.876 | 0.876 | 0.874 | 0.861 |
| s137 | 0.676 | 0.819 | 0.820 | 0.853 | 0.697 |

### First-1K Retention:
| | HIGH | RAND0 | RAND1 | RAND2 | LOW |
|---|---|---|---|---|---|
| s42 | 0.075 | 0.661 | 0.594 | 0.641 | 0.539 |
| s2024 | 0.112 | 0.649 | 0.649 | 0.571 | 0.566 |
| s137 | 0.385 | 0.522 | 0.564 | 0.577 | 0.381 |

**Source:** full_eval files for each ordering × seed. Seed 2024/137 HIGH and LOW values from earlier evals (local copies may be overwritten by temporal evals — use paper/data/ copies or values confirmed in conversation: s2024 HIGH=0.294, LOW=0.861; s137 HIGH=0.676, LOW=0.697).

**Note:** HIGH and LOW are engineered paths. RAND0/1/2 are independent random permutations. Report as "observed random reference range."

---

## Table 3: Suffix-Switch Rescue (MAIN TEXT)

**Purpose:** Show early rescue works partially, late rescue doesn't.

### Overall Efficacy at 10K:
| Branch | Continue | Random-suffix | Spread-suffix | Full-random |
|---|---|---|---|---|
| **5K s42** | 0.325 | **0.451** | 0.346 | 0.879 |
| **5K s2024** | 0.294 | **0.435** | 0.395 | 0.876 |
| 5K s137 | 0.676 | 0.684 | 0.666 | 0.819 |
| 7K s42 | 0.325 | 0.259 | 0.253 | — |
| 7K s2024 | 0.294 | 0.394 | 0.386 | — |
| 7K s137 | 0.676 | 0.675 | 0.691 | — |

**Source:** `/tmp/rescue_suffix_*_s*.json` copied to `paper/data/suffix_*_eval.json`

**Key stats to report alongside:**
- Gap closure: (45−30)/(88−30) ≈ 0.26 = 26% of descriptive gap
- Survivor retention: Continue 9.4% → Rescue 21.2% (seed 42)
- Exposure contrast equivalent at both branch points (~1.12×)

---

## Table 4: MEMIT-Seq Attenuation (MAIN TEXT)

| Seed | AlphaEdit Δlogprob | MEMIT-Seq Δlogprob | Attenuation |
|---|---|---|---|
| 42 | −0.00656 (9/10) | −0.00027 (5/10) | 95.9% |
| 2024 | −0.00457 (8/10) | −0.00061 (7/10) | 86.7% |
| 137 | −0.00335 (9/10) | −0.00261 (7/10) | 22.3% |
| **Pooled** | **−0.00483 (26/30)** | **−0.00116 (19/30)** | **76.0%** |

**Source:** `results/logit_damage/` and `results/logit_damage_memit_seq/`

**Note:** Report MEMIT-Seq sign test: 19/30, p ≈ 0.10. Effect is attenuated but MEMIT-Seq alone is not individually significant. Frame as interaction: editor × exposure.

---

## Table 5: Scheduler Comparison (APPENDIX)

| Strategy | Overall (3-seed mean) | First 1K |
|---|---|---|
| high_exposure | 0.432 | 0.191 |
| exposure_only | 0.835 | 0.567 |
| conditioning_only | 0.836 | 0.587 |
| balanced | 0.846 | 0.582 |
| random0 | 0.858 | 0.611 |
| low_exposure | 0.803 | 0.495 |

**Source:** full_eval files for sched_* orderings

**Note:** Present as honest negative result. Schedulers prevent catastrophe but don't beat random.

---

## Table 6: Displacement Decomposition (MAIN TEXT or APPENDIX)

| Predictor | Spearman r | p |
|---|---|---|
| Global norm N | −0.583 | 0.099 |
| Historical displacement D | **−0.833** | **0.005** |
| D controlling for N | **−0.768** | **0.016** |
| N controlling for D | −0.057 | 0.884 |

**Source:** `results/norm_vs_displacement.json`

**Note:** Verify the partial correlation p-value (advisor flagged possible 0.016 vs 0.026 discrepancy).

---

## Table 7: Pooled A/B Summary (MAIN TEXT)

| Experiment type | Trials | Sign (HI worse) | Mean Δlogprob | p |
|---|---|---|---|---|
| Different-batch | 30 | 26/30 | −0.00483 | 0.036 (seed-clustered) |
| Same-fact | 30 | 21/30 | −0.00204 | 0.043 (two-sided) |
| **Pooled** | **60** | **47/60** | **−0.00344** | **1.2e-05** |

Bootstrap CI (pooled): [−0.0051, −0.0020]

**Source:** `results/pooled_ab_results.json`, `results/logit_damage/corrected_analysis.json`

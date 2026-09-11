# Complete Results — Probability-Preference Metric (v2)

Generated 2026-09-09 after correcting the evaluation metric from argmax to prob-pref.

## Why Re-evaluation Was Necessary

The entire previous analysis used **argmax metrics** — checking if every token in the target is the model's top-1 prediction. Published papers (AlphaEdit, EvoEdit, REVIVE) use **probability-preference metrics** — checking if the target has higher probability than the alternative (pairwise NLL comparison).

The vendor code (`vendor/AlphaEdit/experiments/evaluate.py`) saves both in per-case JSONs:
- `rewrite_prompts_correct` — argmax (what we read)
- `rewrite_prompts_probs` — NLL values with `target_new` and `target_true` (what papers use)

Our `analysis/loaders.py` only read the `_correct` fields. The vendor's own `summarize.py` reads `_probs`.

**Impact:** Neighborhood scores were 5-16% (argmax) vs 52-72% (prob-pref) — a +50pp systematic gap. Efficacy shifted +3-40pp depending on the experiment. All reported numbers were on the wrong metric.

**What we rescored offline (no GPU):** Failure curve per-case JSONs (787K files), GPT-J per-case files (133K), matched ordering per-case files (194K), polykernel seqreg (135K). These already had `_probs` stored — we re-read the NLL values without model inference.

**What required GPU re-evaluation (63 SkyPilot jobs):** Matched ordering `full_eval_seed*.json` files were produced by `eval_matched_ordering.py` which computed only argmax summaries and did not store raw NLL values. We loaded the model, applied each checkpoint's weights, and ran inference on all 10K records to produce `full_eval_seed*_v2.json` with both metrics. This covered AlphaEdit (all orderings, 3 seeds), MEMIT-Seq, PathGuard, and EvoEdit on fb orderings (seed 42).

---

## Table 1: MVE Reproduction @ 2K edits (5 seeds, Llama-3-8B)

| Algorithm | Seed | Efficacy | Neighborhood | Paraphrase | 1st 1K | Latest 1K |
|---|---|---|---|---|---|---|
| AlphaEdit | 42 | 99.2% | 69.0% | 92.8% | 98.5% | 99.9% |
| AlphaEdit | 2024 | 99.2% | 67.9% | 94.5% | 98.4% | 99.9% |
| AlphaEdit | 137 | 98.7% | 68.5% | 94.1% | 97.5% | 99.9% |
| AlphaEdit | 7 | 99.0% | 68.7% | 93.5% | 98.0% | 100.0% |
| AlphaEdit | 99 | 99.1% | 68.3% | 93.8% | 98.4% | 99.7% |
| **AlphaEdit mean** | | **99.0%** | **68.5%** | **93.7%** | **98.2%** | **99.9%** |
| MEMIT | 42 | 65.3% | 51.7% | 63.5% | 48.2% | 82.4% |
| MEMIT | 2024 | 65.3% | 50.7% | 63.9% | 49.5% | 81.1% |
| MEMIT | 137 | 64.1% | 51.0% | 62.4% | 49.7% | 78.6% |
| MEMIT | 7 | 64.6% | 51.9% | 64.4% | 46.6% | 82.6% |
| MEMIT | 99 | 65.1% | 51.7% | 64.5% | 47.9% | 82.3% |
| **MEMIT mean** | | **64.9%** | **51.4%** | **63.7%** | **48.4%** | **81.4%** |

---

## Table 2: Failure Curve — AlphaEdit vs MEMIT (Llama-3-8B, MCF)

| Algorithm | Seed | Edits | Efficacy | Neighborhood | Paraphrase | 1st 1K | Latest 1K | Age Gap |
|---|---|---|---|---|---|---|---|---|
| AlphaEdit | 42 | 2K | 99.2% | 69.0% | 92.8% | 98.5% | 99.9% | +1.4pp |
| AlphaEdit | 42 | 5K | 97.5% | 61.3% | 92.5% | 92.2% | 99.9% | +7.7pp |
| AlphaEdit | 42 | 10K | 72.6% | 52.7% | 60.1% | 57.8% | 97.7% | +39.9pp |
| AlphaEdit | 2024 | 2K | 99.2% | 67.9% | 94.5% | 98.4% | 99.9% | +1.5pp |
| AlphaEdit | 2024 | 5K | 96.2% | 59.5% | 92.4% | 88.6% | 99.8% | +11.2pp |
| AlphaEdit | 2024 | 10K | 62.5% | 52.4% | 57.3% | 50.3% | 94.9% | +44.6pp |
| AlphaEdit | 137 | 5K | 96.4% | 61.8% | 91.1% | 89.2% | 99.9% | +10.7pp |
| AlphaEdit | 137 | 10K | 63.7% | 52.4% | 57.0% | 51.8% | 94.5% | +42.7pp |
| **AE 10K mean** | | **10K** | **66.3%** | **52.5%** | **58.1%** | **53.3%** | **95.7%** | **+42.4pp** |
| MEMIT | 42 | 2K | 61.5% | 52.7% | 61.2% | 49.0% | 73.9% | +24.9pp |
| MEMIT | 42 | 10K | 47.5% | 52.4% | 47.9% | 47.0% | 49.4% | +2.4pp |
| MEMIT | 2024 | 10K | 49.8% | 50.3% | 49.5% | 50.6% | 49.2% | -1.4pp |
| MEMIT | 137 | 10K | 55.6% | 50.0% | 53.4% | 49.0% | 71.5% | +22.5pp |

---

## Table 3: Method Comparison @ 10K (seed 42, Llama-3-8B)

| Rank | Algorithm | Efficacy | Neighborhood | Paraphrase | 1st 1K | Latest 1K |
|---|---|---|---|---|---|---|
| 1 | MEMIT-Seq-poly2-hybrid | 99.0% | 62.7% | 91.5% | 95.8% | 99.9% |
| 2 | MEMIT-Seq-poly3-hybrid | 98.3% | 61.0% | 91.7% | 94.0% | 99.9% |
| 3 | AlphaEdit-C0-15000 | 97.1% | 59.2% | 87.5% | 89.6% | 100.0% |
| 4 | MEMIT-Seq (plain) | 93.6% | 61.0% | 87.2% | 85.1% | 99.7% |
| 5 | MEMIT-Seq-poly4-hybrid | 90.5% | 57.2% | 85.5% | 76.4% | 99.9% |
| 6 | AlphaEdit | 72.6% | 52.7% | 60.1% | 57.8% | 97.7% |
| 7 | MEMIT-Seq-ld10.0 | 67.9% | 52.4% | 58.3% | 51.1% | 97.2% |
| 8 | MEMIT-Seq-poly2 (no hybrid) | 54.9% | 50.7% | 50.5% | 49.4% | 82.9% |
| 9 | MEMIT | 47.5% | 52.4% | 47.9% | 47.0% | 49.4% |

---

## Table 4: GPT-J-6B @ 10K (MCF)

| Rank | Algorithm | Seed | Efficacy | Neighborhood | Paraphrase |
|---|---|---|---|---|---|
| 1 | AlphaEdit-C0-15000 | 2024 | 99.4% | 66.5% | 92.2% |
| 2 | MEMIT-Seq | 42 | 99.3% | 64.7% | 93.5% |
| 3 | AlphaEdit-C0-15000 | 42 | 99.1% | 66.8% | 91.0% |
| 4 | MEMIT-Seq-poly2-hybrid | 42 | 98.3% | 64.2% | 92.5% |
| 5 | AlphaEdit | 2024 | 96.8% | 62.3% | 85.4% |
| 6 | AlphaEdit | 42 | 95.7% | 61.6% | 85.8% |
| 7 | MEMIT | 2024 | 51.0% | 51.5% | 51.5% |
| 8 | MEMIT-Seq-poly2 (no hybrid) | 42 | 50.4% | 49.6% | 50.4% |
| 9 | MEMIT | 42 | 33.1% | 73.6% | 29.6% |

---

## Table 5: 5-Path Fixed-Batch Dose-Response @ 10K (AlphaEdit)

**Overall Efficacy:**

| Seed | HIGH | RAND0 | RAND1 | RAND2 | LOW | HI-LO | HI-RAND(mean) |
|---|---|---|---|---|---|---|---|
| 42 | 69.5% | 97.0% | 95.8% | 96.8% | 95.6% | -26.1pp | -27.0pp |
| 2024 | 72.7% | 97.1% | 96.8% | 96.6% | 95.7% | -23.0pp | -24.1pp |
| 137 | 80.2% | 94.6% | 95.0% | 95.6% | 80.4% | -0.2pp | -14.8pp |
| **Mean** | **74.1%** | **96.2%** | **95.9%** | **96.3%** | **90.6%** | **-16.4pp** | **-22.0pp** |

**First 1K Retention:**

| Seed | HIGH | RAND0 | RAND1 | RAND2 | LOW |
|---|---|---|---|---|---|
| 42 | 48.0% | 89.7% | 85.2% | 88.4% | 80.9% |
| 2024 | 50.6% | 89.0% | 88.3% | 82.7% | 82.5% |
| 137 | 63.3% | 83.5% | 84.0% | 83.9% | 61.8% |

---

## Table 6: Cross-Algorithm Ordering Sensitivity (seed 42, 10K)

**Efficacy:**

| Algorithm | fb_HIGH | fb_LOW | fb_RAND | HI-LO Gap | Ordering-immune? |
|---|---|---|---|---|---|
| PathGuard | 98.0% | 97.6% | 98.2% | +0.4pp | Yes |
| MEMIT-Seq | 97.5% | 97.9% | 98.8% | -0.4pp | Yes |
| EvoEdit | 84.6% | 98.0% | 98.9% | -13.5pp | No |
| AlphaEdit | 69.5% | 95.6% | 97.0% | -26.1pp | No |

**Neighborhood:**

| Algorithm | fb_HIGH | fb_LOW | fb_RAND |
|---|---|---|---|
| PathGuard-poly2-hybrid | -- | 72.0% | 72.5% |
| PathGuard | 67.0% | 70.5% | 70.3% |
| MEMIT-Seq | 64.6% | 69.3% | 68.2% |
| EvoEdit | 59.7% | 69.8% | 69.0% |
| AlphaEdit | 57.4% | 63.1% | 63.9% |

**First 1K Retention:**

| Algorithm | fb_HIGH | fb_LOW | fb_RAND |
|---|---|---|---|
| PathGuard | 93.1% | 89.3% | 95.9% |
| MEMIT-Seq | 91.8% | 90.5% | 96.5% |
| EvoEdit | 61.5% | 91.2% | 96.6% |
| AlphaEdit | 48.0% | 80.9% | 89.7% |

---

## Table 7: Key Clustered vs Dispersed

| Algorithm | Ordering | Seed | N | Efficacy | Neighborhood | Paraphrase | 1st 1K |
|---|---|---|---|---|---|---|---|
| AlphaEdit | clustered | 42 | 10K | 93.8% | 59.5% | 87.1% | 83.5% |
| AlphaEdit | clustered | 2024 | 10K | 94.3% | 59.8% | 86.8% | 85.0% |
| AlphaEdit | clustered | 137 | 5K | 98.4% | 67.5% | 91.4% | 96.5% |
| AlphaEdit | dispersed | 42 | 10K | 72.3% | 54.1% | 63.5% | 53.8% |
| AlphaEdit | dispersed | 137 | 5K | 91.3% | 60.1% | 83.4% | 76.4% |
| MEMIT-Seq | clustered | 42 | 5K | 61.1% | 73.5% | 55.2% | 66.7% |
| MEMIT-Seq | dispersed | 42 | 5K | 63.8% | 69.5% | 58.0% | 60.3% |

---

## Table 8: Scheduler Comparison @ 10K (AlphaEdit)

| Strategy | s42 | s2024 | s137 | Mean | Mean 1st 1K |
|---|---|---|---|---|---|
| fb_high_exposure | 69.5% | 72.7% | 80.2% | 74.1% | 54.0% |
| sched: exposure_only | 95.3% | 95.7% | 94.3% | 95.1% | 82.0% |
| sched: conditioning_only | 95.8% | 96.2% | 95.2% | 95.7% | 86.0% |
| sched: balanced | 94.8% | 97.1% | 95.6% | 95.8% | 85.7% |
| **fb_random0** | **97.0%** | **97.1%** | **94.6%** | **96.2%** | **87.4%** |
| fb_low_exposure | 95.6% | 95.7% | 80.4% | 90.6% | 75.1% |

Schedulers prevent catastrophe (74% -> 95%) but do not beat random (96.2%).

---

## Table 9: Suffix Rescue @ 10K (AlphaEdit)

| Branch | s42 Eff | s42 1st1K | s2024 Eff | s2024 1st1K | s137 Eff | s137 1st1K |
|---|---|---|---|---|---|---|
| Continue HIGH | 69.5% | 48.0% | 72.7% | 50.6% | 80.2% | 63.3% |
| suffix random | -- | -- | 76.4% | 54.2% | 80.5% | 63.1% |
| suffix spread | -- | -- | 75.6% | 55.3% | 79.8% | 62.1% |
| suffix random (late) | 68.3% | 48.9% | 74.8% | 52.1% | 80.1% | 63.8% |
| suffix spread (late) | 67.8% | 47.6% | 75.0% | 54.0% | 80.7% | 63.2% |
| Full random | 97.0% | 89.7% | 97.1% | 89.0% | 94.6% | 83.5% |

Under prob-pref, the suffix rescue effect is negligible. Seed 2024 early rescue shows +3.7pp, late rescue negligible. This contrasts with the argmax metric where the effect appeared larger — argmax amplified small probability differences near the decision boundary into binary flips.

---

## Table 10: Polykernel Degradation Curves (seed 42, Llama-3-8B)

**Efficacy:**

| Edits | poly2-hybrid | poly3-hybrid | poly4-hybrid | poly2 | MEMIT-Seq | AlphaEdit |
|---|---|---|---|---|---|---|
| 2K | 99.6% | 99.6% | 99.5% | 99.5% | 99.5% | 99.2% |
| 3K | 99.5% | 99.4% | 99.4% | 99.2% | 99.4% | 98.9% |
| 4K | 99.5% | 99.4% | 99.5% | 98.7% | 99.6% | 98.3% |
| 5K | 99.5% | 99.4% | 99.2% | 98.3% | 99.4% | 97.5% |
| 6K | 99.5% | 99.4% | 98.9% | 96.8% | 99.3% | 96.1% |
| 7K | 99.3% | 99.2% | 98.4% | 93.2% | 99.0% | 92.9% |
| 8K | 99.1% | 98.8% | 97.5% | 83.2% | 98.3% | 87.1% |
| 9K | 99.0% | 98.6% | 95.6% | 63.7% | 96.6% | 79.0% |
| 10K | **99.0%** | **98.3%** | 90.5% | 54.9% | 93.6% | 72.6% |

**Neighborhood @ 10K:** poly2-hybrid 62.7%, poly3-hybrid 61.0%, poly4-hybrid 57.2%, poly2 50.7%, MEMIT-Seq 61.0%, AlphaEdit 52.7%

**1st 1K @ 10K:** poly2-hybrid 95.8%, poly3-hybrid 94.0%, poly4-hybrid 76.4%, poly2 49.4%, MEMIT-Seq 85.1%, AlphaEdit 57.8%

The hybrid linear term stabilizes long-horizon editing. Without it (poly2), catastrophic collapse occurs between 7K-9K. With it, poly2-hybrid and poly3-hybrid maintain >98% through 10K.

---

## Table 11: REVIVE Tau Sweep (seed 42)

| Config | Edits | Efficacy | Neighborhood |
|---|---|---|---|
| poly2-REVIVE tau=0.05 | 1K | 99.6% | 77.5% |
| poly2-REVIVE tau=0.1 | 1K | 99.5% | 76.8% |
| poly2-REVIVE tau=0.2 | 1K | 99.6% | 76.0% |
| poly2-REVIVE tau=0.3 | 1K | 99.8% | 75.4% |
| poly2-REVIVE tau=0.4 | 1K | 99.5% | 75.1% |
| poly2-REVIVE tau=0.2 | 5K | 92.4% | 56.6% |
| poly2-hybrid-REVIVE tau=0.2 | 5K | 99.1% | 57.9% |

At 1K all tau values produce near-identical results. At 5K the hybrid term matters more than the REVIVE filter (99.1% vs 92.4%).

---

## Table 12: Best Overall Results by Model

**Llama-3-8B @ 10K (non-ordering, MCF, seed 42):**

| Rank | Algorithm | Efficacy | Neighborhood | Paraphrase | 1st 1K |
|---|---|---|---|---|---|
| 1 | MEMIT-Seq-poly2-hybrid | 99.0% | 62.7% | 91.5% | 95.8% |
| 2 | MEMIT-Seq-poly3-hybrid | 98.3% | 61.0% | 91.7% | 94.0% |
| 3 | AlphaEdit-C0-15000 | 97.1% | 59.2% | 87.5% | 89.6% |
| 4 | MEMIT-Seq (plain) | 93.6% | 61.0% | 87.2% | 85.1% |
| 5 | MEMIT-Seq-poly4-hybrid | 90.5% | 57.2% | 85.5% | 76.4% |
| 6 | AlphaEdit | 72.6% | 52.7% | 60.1% | 57.8% |
| 7 | MEMIT | 47.5% | 52.4% | 47.9% | 47.0% |

**GPT-J-6B @ 10K (MCF):**

| Rank | Algorithm | Seed | Efficacy | Neighborhood | Paraphrase |
|---|---|---|---|---|---|
| 1 | AlphaEdit-C0-15000 | 2024 | 99.4% | 66.5% | 92.2% |
| 2 | MEMIT-Seq | 42 | 99.3% | 64.7% | 93.5% |
| 3 | AlphaEdit-C0-15000 | 42 | 99.1% | 66.8% | 91.0% |
| 4 | MEMIT-Seq-poly2-hybrid | 42 | 98.3% | 64.2% | 92.5% |
| 5 | AlphaEdit | 2024 | 96.8% | 62.3% | 85.4% |
| 6 | AlphaEdit | 42 | 95.7% | 61.6% | 85.8% |
| 7 | MEMIT | 2024 | 51.0% | 51.5% | 51.5% |
| 8 | MEMIT | 42 | 33.1% | 73.6% | 29.6% |

**Llama-3-8B matched ordering @ 10K (fb_random0, seed 42):**

| Rank | Algorithm | Efficacy | Neighborhood | 1st 1K |
|---|---|---|---|---|
| 1 | EvoEdit | 98.9% | 69.0% | 96.6% |
| 2 | MEMIT-Seq | 98.8% | 68.2% | 96.5% |
| 3 | PathGuard-poly2-hybrid | 98.4% | 72.5% | 96.5% |
| 4 | PathGuard | 98.2% | 70.3% | 95.9% |
| 5 | AlphaEdit | 97.0% | 63.9% | 89.7% |

---

## MEMIT-Seq Failure Curves (seed 42, Llama-3-8B, 2K-10K)

### MEMIT-Seq (plain, lp1.0-ld0.0)

| Edits | Efficacy | Neighborhood | 1st 1K | Latest 1K |
|---|---|---|---|---|
| 2K | 99.5% | 79.1% | 99.1% | 99.8% |
| 3K | 99.4% | 75.0% | 98.7% | 99.7% |
| 5K | 99.4% | 68.5% | 98.4% | 99.9% |
| 7K | 99.0% | 64.7% | 96.3% | 99.9% |
| 10K | 93.6% | 61.0% | 85.1% | 99.7% |

### MEMIT-Seq-poly2-hybrid

| Edits | Efficacy | Neighborhood | 1st 1K | Latest 1K |
|---|---|---|---|---|
| 2K | 99.6% | 81.4% | 99.2% | 100.0% |
| 3K | 99.5% | 78.3% | 99.1% | 99.7% |
| 5K | 99.5% | 71.5% | 98.5% | 99.9% |
| 7K | 99.3% | 66.4% | 97.4% | 100.0% |
| 10K | 99.0% | 62.7% | 95.8% | 99.9% |

### MEMIT-Seq-poly2 (no hybrid)

| Edits | Efficacy | Neighborhood | 1st 1K | Latest 1K |
|---|---|---|---|---|
| 2K | 99.5% | 81.1% | 99.0% | 100.0% |
| 3K | 99.2% | 77.5% | 98.3% | 99.6% |
| 5K | 98.3% | 69.8% | 94.5% | 99.9% |
| 7K | 93.2% | 61.9% | 83.8% | 99.8% |
| 10K | 54.9% | 50.7% | 49.4% | 82.9% |

### MEMIT-Seq-poly3-hybrid

| Edits | Efficacy | Neighborhood | 1st 1K | Latest 1K |
|---|---|---|---|---|
| 2K | 99.6% | 79.1% | 99.4% | 99.7% |
| 3K | 99.4% | 74.8% | 99.1% | 99.6% |
| 5K | 99.4% | 68.3% | 98.3% | 100.0% |
| 7K | 99.2% | 63.7% | 96.9% | 99.9% |
| 10K | 98.3% | 61.0% | 94.0% | 99.9% |

### MEMIT-Seq-poly4-hybrid

| Edits | Efficacy | Neighborhood | 1st 1K | Latest 1K |
|---|---|---|---|---|
| 2K | 99.5% | 77.5% | 99.0% | 100.0% |
| 3K | 99.4% | 73.0% | 98.7% | 99.8% |
| 5K | 99.2% | 65.5% | 97.1% | 99.9% |
| 7K | 98.4% | 61.1% | 93.9% | 100.0% |
| 10K | 90.5% | 57.2% | 76.4% | 99.9% |

---

## Summary of Key Paper Numbers

| Metric | Value |
|---|---|
| Age-selective gap (3-seed mean, 10K) | 42.4pp (oldest 53.3% vs newest 95.7%) |
| Overall efficacy at 10K (AlphaEdit, 3-seed) | 66.3% |
| Fixed-batch HI-LO gap (3-seed mean) | -16.4pp |
| Fixed-batch HI-RAND gap (3-seed mean) | -22.0pp |
| Best method at 10K (seed 42) | poly2-hybrid: 99.0% eff, 62.7% neigh, 95.8% 1st-1K |
| Cross-alg: AlphaEdit ordering gap | -26.1pp |
| Cross-alg: EvoEdit ordering gap | -13.5pp |
| Cross-alg: MEMIT-Seq ordering gap | -0.4pp |
| Cross-alg: PathGuard ordering gap | +0.4pp |
| Scheduler best | random (96.2%) > balanced (95.8%) > conditioning (95.7%) |
| Suffix rescue effect | negligible under prob-pref |

---

## Argmax vs Prob-Pref Comparison

| Metric | Old (argmax) | New (prob-pref) | Shift |
|---|---|---|---|
| Overall eff @10K (3-seed) | 21.6% | 66.3% | +44.7pp |
| Oldest 1K @10K | 7.1% | 53.3% | +46.2pp |
| Newest 1K @10K | 71.5% | 95.7% | +24.2pp |
| Age gap | 64.4pp | 42.4pp | -22.0pp |
| Neighborhood @10K | ~5% | ~52% | +47pp |
| fb_high s42 | 32.5% | 69.5% | +37pp |
| fb_low s42 | 85.1% | 95.6% | +10.5pp |
| HI-LO gap s42 | -52.6pp | -26.1pp | shrinks |
| AlphaEdit @2K eff | 95.4% | 99.0% | +3.6pp |
| MEMIT @2K eff | 24.5% | 64.9% | +40.4pp |

The qualitative story is preserved: age-selective forgetting, ordering sensitivity, and the cross-algorithm ranking all hold. The absolute numbers change substantially, bringing our results in line with published literature.

---

## Data Sources

- Failure curve rescores: `results/failure_curve_checkpointed/*/probpref_summary.json` (offline, 787K per-case files)
- GPT-J rescores: `results/failure_curve_gptj/*/probpref_summary.json` (offline, 133K per-case files)
- Matched ordering v2: `results/matched_ordering/*/full_eval_seed*_v2.json` (56 GPU-evaluated files)
- Polykernel rescores: `results/polykernel_seqreg/*/probpref_summary.json` (offline, 135K per-case files)
- REVIVE rescores: `results/failure_curve_checkpointed/seed42/*/MEMIT-Seq-poly2-REVIVE-*/probpref_summary.json`

## Experiments NOT yet on prob-pref

- zsRE failure curve: per-case files lack `_probs` fields entirely (needs GPU re-run)
- A/B interventions (logit_damage, same_fact_damage): unaffected — these measure continuous logprob, not binary success
- Displacement / mediation analysis: unaffected — continuous metrics

## Known Issues

### MEMIT-Seq key_clustered/key_dispersed v2 results are INVALID

The v2 GPU re-eval of MEMIT-Seq on key_clustered/key_dispersed produced 61-64% efficacy,
compared to 97-99% on fb orderings. Root cause: **checkpoint format mismatch**.

- key_clustered checkpoints on S3: **2.2 GB** model_weights.pt (full model, July 22 run)
- fb ordering checkpoints on S3: **1.1 GB** model_weights.pt (edited layers only, September 4 run)

The eval script loads stored weights on top of a fresh base model. The 2.2GB file overwrites
ALL parameters (including pretrained layers), producing a corrupted model state.

**Action:** Discard MEMIT-Seq key_clustered/key_dispersed v2 results.
Use v1 argmax results (95.7% clustered, 92.2% dispersed at 10K) until checkpoints are fixed.

### Binary first-failure survival model is unstable under prob-pref

Under prob-pref, 83% of edits survive (vs ~30% under argmax). Failures concentrate in the
oldest cohorts. Age alone explains nearly all variance — cosine adds almost nothing (mean
difference only 0.021). The logistic model produces singular matrices, infinite ORs, or
AUC=1.0 overfitting.

**Solution:** Continuous NLL margin model (see analysis/margin_survival_model.py):
  margin_i(t) = NLL(target_true) - NLL(target_new)
Predicts continuous margin decline rather than binary first-failure.

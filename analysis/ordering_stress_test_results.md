# Ordering Stress Test Results

## Setup

All experiments: Llama-3-8B-Instruct, 10,000 sequential edits in batches of $B=100$, MultiCounterFact dataset. Each ordering permutes the same 10K facts into a fixed stream before editing begins.

**Orderings tested:**
- **Random** (`fb_random0`): uniform random permutation (control)
- **High exposure** (`fb_high_exposure`): facts sharing subject-relation clusters are packed together — maximises interference between consecutive batches
- **Low exposure** (`fb_low_exposure`): related facts are spread as far apart as possible — minimises local interference

**Algorithms:**
- **AlphaEdit** — null-space projected editing (ICLR 2025 Outstanding Paper)
- **MEMIT-Aligned** — MEMIT with sequential key regularisation ($\lambda_{\text{prev}}=1.0$, $\lambda_\delta=0$)
- **RECT-Aligned** — MEMIT with rectification error correction
- **EvoEdit** — evolutionary optimisation-based editing

## Metrics

Two scoring protocols applied to every edited fact:

- **Probability-preference (PP)**: the new target has lower NLL than the original answer, i.e. $\text{NLL}(t_{\text{new}}) < \text{NLL}(t_{\text{true}})$. This is the metric used in published tables for AlphaEdit, REVIVE, and NSE.
- **Argmax (AM)**: the new target is the greedy-decoded output at every token position. Strictly harder — the model must not just *prefer* the edit, it must *commit* to it as the dominant prediction.

Both are computed over all 10,000 edited facts at the final checkpoint. Additionally we report **First-1K** (facts 0–999, the earliest edits) and **Latest-1K** (facts 9001–10000) to measure temporal degradation.

---

## 1 — How badly does ordering hurt each algorithm?

Read each row left-to-right: **Random** is the control, **High** and **Low** show how much ordering changes the outcome. The $\Delta$ columns are the drop from random.

### PP Efficacy (the published metric)

| Algorithm | Random | High | $\Delta_{\text{high}}$ | Low | $\Delta_{\text{low}}$ |
|---|:---:|:---:|:---:|:---:|:---:|
| AlphaEdit | 96.2 | 74.1 | **−22.1** | 90.6 | −5.6 |
| MEMIT-Aligned | 98.7 | 97.1 | −1.6 | 86.1 | **−12.6** |
| RECT-Aligned | 95.7 | 95.6 | −0.1 | 94.2 | −1.5 |
| EvoEdit | 98.6 | 92.8 | −5.8 | 97.9 | −0.7 |

### AM Efficacy (the strict metric)

| Algorithm | Random | High | $\Delta_{\text{high}}$ | Low | $\Delta_{\text{low}}$ |
|---|:---:|:---:|:---:|:---:|:---:|
| AlphaEdit | 85.8 | 45.3 | **−40.5** | 80.3 | −5.5 |
| MEMIT-Aligned | 94.9 | 90.4 | −4.5 | 78.7 | **−16.2** |
| RECT-Aligned | 87.9 | 87.8 | −0.1 | 86.3 | −1.6 |
| EvoEdit | 94.6 | 73.9 | **−20.7** | 93.4 | −1.2 |

**Pattern:** AlphaEdit and EvoEdit collapse under high exposure (−22 to −41 points). MEMIT-Aligned is uniquely vulnerable to low exposure instead (−13 to −16). RECT-Aligned is nearly immune to both ($|\Delta| \leq 1.6$).

---

## 2 — How much do early edits degrade?

The first 1,000 facts edited are measured at the final 10K checkpoint. **F1K** = first-1K efficacy, **L1K** = latest-1K efficacy. The gap $\Delta = \text{L1K} - \text{F1K}$ measures forgetting — larger is worse.

### Under random ordering (control)

| Algorithm | PP F1K | PP L1K | $\Delta_{\text{PP}}$ | AM F1K | AM L1K$^*$ | $\Delta_{\text{AM}}$ |
|---|:---:|:---:|:---:|:---:|:---:|:---:|
| EvoEdit | 94.4 | 97.6 | +3.2 | 82.3 | ~95 | +13 |
| MEMIT-Aligned | 95.3 | 99.7 | +4.4 | 83.9 | ~95 | +11 |
| AlphaEdit | 87.4 | 99.6 | +12.2 | 61.1 | ~86 | +25 |
| RECT-Aligned | 89.0 | 98.0 | +9.0 | 74.2 | ~88 | +14 |

### Under high exposure (worst case)

| Algorithm | PP F1K | PP L1K | $\Delta_{\text{PP}}$ | AM F1K | AM L1K$^*$ | $\Delta_{\text{AM}}$ |
|---|:---:|:---:|:---:|:---:|:---:|:---:|
| RECT-Aligned | 89.7 | 97.1 | +7.4 | 74.4 | ~88 | +14 |
| MEMIT-Aligned | 88.2 | 99.1 | +10.9 | 69.4 | ~90 | +21 |
| EvoEdit | 78.6 | 96.2 | +17.6 | 45.9 | ~74 | +28 |
| AlphaEdit | 54.0 | 96.2 | **+42.2** | 20.3 | ~45 | **+25** |

$^*$AM L1K approximated from overall AM Eff since latest-1K AM not separately tracked in all runs.

**Pattern:** Sorted by severity. Under random, all methods have modest F1K drops in PP (3–12 pts). Under high exposure, AlphaEdit's first-1K AM falls to **20.3%** — four in five of the earliest edits are effectively erased. RECT-Aligned's F1K is barely affected by the ordering switch.

---

## 3 — What does the strict metric reveal that PP hides?

The gap between PP and AM for the same experiment. Larger gap = the model "prefers" the edit but doesn't actually produce it.

### All 10K facts

| Algorithm | Ordering | PP Eff | AM Eff | Gap |
|---|---|:---:|:---:|:---:|
| AlphaEdit | High | 74.1 | 45.3 | **28.8** |
| EvoEdit | High | 92.8 | 73.9 | **18.9** |
| RECT-Aligned | Low | 94.2 | 86.3 | 7.9 |
| MEMIT-Aligned | Random | 98.7 | 94.9 | 3.8 |

### First-1K facts only

| Algorithm | Ordering | PP F1K | AM F1K | Gap |
|---|---|:---:|:---:|:---:|
| AlphaEdit | High | 54.0 | 20.3 | **33.7** |
| EvoEdit | High | 78.6 | 45.9 | **32.7** |
| AlphaEdit | Low | 75.1 | 49.5 | 25.6 |
| AlphaEdit | Random | 87.4 | 61.1 | 26.3 |
| EvoEdit | Low | 89.5 | 73.7 | 15.8 |
| MEMIT-Aligned | High | 88.2 | 69.4 | 18.8 |
| MEMIT-Aligned | Low | 81.3 | 66.3 | 15.0 |
| RECT-Aligned | Low | 80.0 | 62.0 | 18.0 |
| RECT-Aligned | Random | 89.0 | 74.2 | 14.8 |
| MEMIT-Aligned | Random | 95.3 | 83.9 | 11.4 |
| EvoEdit | Random | 94.4 | 82.3 | 12.1 |
| RECT-Aligned | High | 89.7 | 74.4 | 15.3 |

**Pattern:** The PP–AM gap is largest exactly where degradation is worst (AlphaEdit + high exposure first-1K: 33.7 pts). PP reports 54% efficacy for early edits; AM says 20%. The published metric flatters methods that shift probability slightly toward the target without making it dominant.

---

## 4 — Neighbourhood preservation

Neighbourhood measures whether *unedited related facts* are preserved. Higher is better.

| Algorithm | Random | High | Low | Best |
|---|:---:|:---:|:---:|:---:|
| **PP Neighbourhood** |
| RECT-Aligned | **73.3** | **71.6** | **74.5** | ← |
| MEMIT-Aligned | 67.7 | 64.8 | 69.0 | |
| EvoEdit | 68.0 | 62.6 | 69.8 | |
| AlphaEdit | 62.6 | 59.9 | 64.9 | |
| **AM Neighbourhood** |
| RECT-Aligned | **17.6** | **17.9** | **18.4** | ← |
| MEMIT-Aligned | 14.1 | 13.1 | 14.5 | |
| EvoEdit | 13.2 | 10.0 | 13.7 | |
| AlphaEdit | 11.9 | 8.8 | 13.4 | |

**Pattern:** RECT-Aligned wins neighbourhood across every ordering and both metrics. AlphaEdit has the worst neighbourhood — its null-space projection, designed to preserve existing knowledge, performs worst on this exact criterion. AM neighbourhood is uniformly low (8–18%) — at argmax level, all methods severely damage related facts after 10K edits.

---

## 5 — Algorithm ranking summary

Ranked by robustness. Reading: how does each method perform across the full stress-test matrix?

| Criterion | Best | 2nd | 3rd | Worst |
|---|---|---|---|---|
| Ordering robustness (max $|\Delta|$) | RECT (1.6) | MEMIT-A (16.2) | EvoEdit (20.7) | AlphaEdit (40.5) |
| Worst-case AM efficacy | RECT (86.3) | MEMIT-A (78.7) | EvoEdit (73.9) | AlphaEdit (45.3) |
| First-1K AM (high exp) | RECT (74.4) | MEMIT-A (69.4) | EvoEdit (45.9) | AlphaEdit (20.3) |
| Neighbourhood (PP) | RECT (71–75) | MEMIT-A (65–69) | EvoEdit (63–70) | AlphaEdit (60–65) |
| Best-case PP efficacy | MEMIT-A (98.7) | EvoEdit (98.6) | AlphaEdit (96.2) | RECT (95.7) |

RECT-Aligned dominates on robustness, preservation, and neighbourhood. MEMIT-Aligned wins on peak efficacy. AlphaEdit — the method with the strongest theoretical preservation guarantee — ranks last on every robustness criterion.

---

## Status

- AlphaEdit: 3 seeds (42, 137, 2024) per ordering ✅
- MEMIT-Aligned: 3 seeds per ordering ✅
- RECT-Aligned: 2 seeds (42, 2024) per ordering ✅
- EvoEdit: 2 seeds for low, 1 seed (s2024) for random/high (s42 being recalculated) ⏳
- EvoEdit GPT-J: 2 orderings × 1 seed — results under verification (may have evaluated with wrong model) ⚠️
- NSE / REVIVE+NSE: invalid due to v_star cache mismatch — vendor trajectory test running ⏳
- REVIVE+AlphaEdit: ordering runs in progress (~50/100 batches) ⏳

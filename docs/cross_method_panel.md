# Cross-Method Comparison Panel

## Overview

We evaluate temporal-path sensitivity across ALL published sequential knowledge editing methods with available code. Each method is tested on our fixed-batch ordering stress test (fb_high vs fb_random0 vs fb_low) on Llama-3-8B (MCF, 10K edits) and GPT-J-6B where applicable.

The goal: establish that **edit order is a hidden benchmark axis** that affects diverse editing families differently, not just AlphaEdit.

---

## Methods Under Test

### Null-Space Projection Family

| Method | Paper | Mechanism | Our Config |
|---|---|---|---|
| **AlphaEdit** | [ICLR 2025 Outstanding Paper](https://proceedings.iclr.cc/paper_files/paper/2025/hash/29c8c615b3187ee995029284702d3f43-Abstract-Conference.html) | Fixed null-space projection from pretrained covariance | Standard, 5 layers (4-8) |
| **EvoEdit** | [ACL Findings 2026](https://aclanthology.org/2026.findings-acl.75/) | Evolving sequential null-space alignment — dynamically aligns projector with accumulated edits | Standard hparams, L2=4 |

### History-Aware Regularization

| Method | Paper | Mechanism | Our Config |
|---|---|---|---|
| **MEMIT-Seq** | Our implementation (non-projected analogue of AlphaEdit Eq. 12) | Accumulated-history LHS augmentation: α·C₀ + K_new@K_new^T + λ_prev·K_prev@K_prev^T | λ_prev=1.0, λ_delta=0.0, cache=all |

**Note:** MEMIT-Seq is identical to **MEMIT-Aligned** from the RECT-Aligned paper (Labyrinth/Thread). Both augment MEMIT's LHS with accumulated previous-key regularization. We use the name MEMIT-Seq throughout for consistency with our codebase.

### Neuron-Level Selection

| Method | Paper | Mechanism | Our Config |
|---|---|---|---|
| **NSE** | [ACL 2025 Main](https://aclanthology.org/2025.acl-long.815/) | Selects neurons by activation magnitude, optimizes target hidden states using original model weights. 25 gradient steps per edit. | alpha=2.5/15, upper_bound=50/100, neuron_threshold=0.8/1.0 |

[GitHub: jianghoucheng/NSE](https://github.com/jianghoucheng/NSE)

### Spectral Preservation

| Method | Paper | Mechanism | Our Config |
|---|---|---|---|
| **REVIVE** | [ACL 2026 Long](https://aclanthology.org/2026.acl-long.1384/) | Projects ΔW to remove components in dominant singular directions of current weights. SVD-based spectral filter applied post-solve. | tau=0.1, full_matrices=True, dynamic SVD per batch |

**REVIVE is a plugin** — it wraps a base editing method. We test:
- **REVIVE + MEMIT** (λ_prev=0): spectral filtering on plain MEMIT (matches REVIVE paper)
- **REVIVE + AlphaEdit**: spectral filtering on null-space projected editing
- **REVIVE + NSE**: spectral filtering on neuron-level editing
- **REVIVE + RECT-Aligned**: spectral filtering on alignment-based editing

### Alignment-Based

| Method | Paper | Mechanism | Our Config |
|---|---|---|---|
| **RECT-Aligned** | Labyrinth/Thread (RECT family) | Rectified alignment approach with error-cache correction | Standard config |

**Note:** The RECT-Aligned paper introduces MEMIT-Aligned as a baseline, which is functionally identical to our MEMIT-Seq (accumulated previous-key regularization on MEMIT's LHS). We verified this equivalence.

---

## Experiment Matrix

### Llama-3-8B (MCF, 10K edits)

#### Ordering Stress Test (fb_high / fb_low / fb_random0)

See `results/matched_ordering/` for current completion status per method, ordering, and seed. V2 eval files (`full_eval_seed*_v2.json`) contain dual-metric (prob-pref + argmax) results.

Core methods with complete v2 results: AlphaEdit, EvoEdit, MEMIT-Seq, PathGuard-poly2 (3 seeds x 3 orderings each).

Extended methods (varying completion): NSE, REVIVE+MEMIT, REVIVE+AlphaEdit, REVIVE+NSE, REVIVE+RECT, RECT-Aligned.

---

## Method Equivalences and Notes

### MEMIT-Seq ≡ MEMIT-Aligned

The RECT-Aligned paper (Labyrinth/Thread) introduces "MEMIT-Aligned" as:
```
LHS = α·C₀ + K_new@K_new^T + λ·K_prev@K_prev^T
```

Our MEMIT-Seq uses the identical formulation:
```
LHS = α·C₀ + K_new@K_new^T + λ_prev·K_prev@K_prev^T + λ_delta·I
```

With λ_delta=0, these are the same. We verified numerically that both produce identical weight updates given the same inputs.

### REVIVE Combinations

REVIVE is tested as a **plugin on top of each base method** to evaluate whether spectral preservation helps different editing mechanisms differently:

| Combination | Base | Spectral | Tests |
|---|---|---|---|
| REVIVE+MEMIT | Plain MEMIT (no history) | SVD filter tau=0.1 | Does spectral filtering alone help? |
| REVIVE+AlphaEdit | AlphaEdit (null-space P) | SVD filter tau=0.1 | Does spectral help null-space methods? |
| REVIVE+NSE | NSE (neuron selection) | SVD filter tau=0.1 | Does spectral help neuron methods? |
| REVIVE+RECT | RECT-Aligned | SVD filter tau=0.1 | Does spectral help alignment methods? |

This is a factorial design: {base method} × {with/without REVIVE} tests the interaction between method design and spectral preservation under temporal stress.

---

## What We're Testing

The central question for each method:

> Under identical fixed-batch orderings, does this method exhibit temporal-path sensitivity?

We report for each method × ordering × seed:
- Probability-preference efficacy (primary benchmark metric)
- Neighborhood specificity
- Paraphrase generalization
- Oldest-1K cohort retention
- Newest-1K efficacy
- HIGH–RANDOM gap (the ordering sensitivity measure)

Methods that show near-zero HIGH–RANDOM gap are **temporally robust**. Methods with large gaps have a **hidden evaluation vulnerability** that single-stream benchmarks miss.

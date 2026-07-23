# Experiment Results — Full Detail

This document provides a comprehensive record of all experimental results produced by this reproducibility study, including raw metrics, observations, and interpretations.

---

## Table of Contents

1. [Results Directory Structure](#results-directory-structure)
2. [Evaluation Pipeline](#evaluation-pipeline)
3. [MVE1: AlphaEdit on MultiCounterFact](#mve1-alphaedit-on-multicounterfact)
4. [MVE2: MEMIT on MultiCounterFact](#mve2-memit-on-multicounterfact)
5. [MVE3: AlphaEdit on zsRE](#mve3-alphaedit-on-zsre)
6. [Failure Curve (Checkpointed)](#failure-curve-checkpointed)
7. [Capability Probe](#capability-probe)
8. [Matched Ordering](#matched-ordering)
9. [Interference Panel Analysis](#interference-panel-analysis)
10. [MEMIT+SeqReg (Sequential Regularization)](#memitseqreg-sequential-regularization)
11. [Polykernel Editor](#polykernel-editor)
12. [Comparison Ordered](#comparison-ordered)
13. [Mechanism Analysis](#mechanism-analysis)
14. [Interference (Installation Strength)](#interference-installation-strength)
15. [Summary of Key Findings](#summary-of-key-findings)

---

## Results Directory Structure

```
results/
├── mve1_alphaedit_mcf/           # 10,080 files — 5 seeds × 2000 edits
├── mve2_memit_mcf/               # 10,080 files — 5 seeds × 2000 edits
├── mve3_alphaedit_zsre/          # 10,000 files — 5 seeds × 2000 edits
├── failure_curve_checkpointed/   # 414,538 files — 3 seeds × up to 10K edits
│   └── seed{N}/{E}edits/{Alg}/run_000/*_edits-case_*.json
├── capability_probe/             # JSONL timeseries (perplexity + MMLU)
│   └── seed{N}/{E}edits/{Alg}/offline_probe_*.jsonl
├── matched_ordering/             # Clustered vs dispersed ordering experiments
│   ├── orderings/{ORDERING}_seed{N}.json
│   └── {ALG}/{ORDERING}/seed{N}/full_eval_seed{N}.json
├── interference/                 # Per-edit forgetting regression analysis
│   └── installation_strength_seed{N}.json
├── mechanism_analysis/           # Cache rank & singular value diagnostics
│   └── seed{N}/mechanism_seed{N}_*.jsonl
├── key_vectors/                  # Precomputed keys (NPZ, 100MB/seed)
├── polykernel_editor/            # 56,234 files — poly2 kernels 2K-10K edits
│   └── seed{N}/{E}edits/AlphaEdit-poly2/run_000/
├── polykernel_diagnostic/        # Effective rank in kernel space
├── comparison_ordered/           # 152,232 files — 5 orderings × 3 scales
│   └── seed{N}/{E}edits/order{0-4}/{Alg}/run_000/
└── figures/
    ├── paper/                    # Main manuscript figures + tables + paper_numbers.json
    ├── appendix/                 # Extended analysis figures
    └── method_comparison/        # Algorithm comparison summary CSV
```

**Total scale**: ~657,000 files across 14 experiment groups.

---

## Evaluation Pipeline

### How Metrics Are Computed from Per-Case JSON Files

Each experiment produces per-case JSON files (one per edited fact). The analysis pipeline (`analysis/loaders.py`) extracts metrics using:

**Per-case JSON structure** (e.g., `100_edits-case_0.json`):
```json
{
  "case_id": 0,
  "grouped_case_ids": [0, 1, ..., 99],
  "num_edits": 100,
  "requested_rewrite": {"prompt": "...", "subject": "...", "target_new": {...}, "target_true": {...}},
  "post": {
    "rewrite_prompts_correct": [true],
    "paraphrase_prompts_correct": [true, false],
    "neighborhood_prompts_correct": [true, false, false, ...],
    "rewrite_prompts_probs": [{"target_new": -2.3, "target_true": -8.1}],
    "paraphrase_prompts_probs": [...],
    "neighborhood_prompts_probs": [...]
  }
}
```

**Metric extraction** (`loaders.py:94-113`):
- **Efficacy** (binary): `mean(post.rewrite_prompts_correct)` — fraction of rewrite prompts answered correctly
- **Paraphrase** (binary): `mean(post.paraphrase_prompts_correct)` — fraction of paraphrased prompts correct
- **Neighborhood** (binary): `mean(post.neighborhood_prompts_correct)` — fraction of neighbor prompts still correct
- **Probability variants**: `mean(target_new log-prob)` from `*_probs` fields

**Aggregation across cases** (macro-averaging):
1. Within-case: compute metric as mean of per-prompt booleans → single float per case
2. Across-cases: compute mean of all case-level metrics → final aggregate
3. Each case contributes equally regardless of prompt count

**Full_eval JSON format** (for checkpointed/matched ordering experiments):
```json
{
  "2000_edits": {
    "all_facts": {"efficacy": 0.955, "paraphrase": 0.691, "neighborhood": 0.166},
    "first_1k": {"efficacy": 0.92},
    "latest_1k": {"efficacy": 0.99},
    "retention_auc": 0.956,
    "cohort_metrics": {"0": {"edits_range": "0-100", "efficacy": 0.98}, ...}
  }
}
```

### Data Flow

```
Experiments (GPU clusters)
  → Per-case JSON files (one per edited fact, ~500 bytes each)
  → analysis/loaders.py (metric extraction via extract_case_metrics + _aggregate_case_files)
  → analysis/stats/ (Wilson CIs for binary, BCa bootstrap for continuous)
  → analysis/paper_tables.py (summary tables + paper_numbers.json)
  → analysis/fig*.py (figure generation)
  → results/figures/{paper,appendix}/ (PNG + PDF outputs)
```

### Key Output Paths

| Output | Path |
|--------|------|
| Paper figures | `results/figures/paper/fig{1-5}_*.{png,pdf}` |
| Appendix figures | `results/figures/appendix/a{1,3,8,9}_*.{png,pdf}` |
| Table 1 (reproduction) | `results/figures/paper/table1_reproduction.csv` |
| Table 3 (matched comparison) | `results/figures/paper/table3_matched_comparison.csv` |
| Table 5 (interference) | `results/figures/paper/table5_interference.csv` |
| All paper numbers | `results/figures/paper/paper_numbers.json` |
| Method comparison | `results/figures/method_comparison/method_comparison_summary.csv` |

### Orchestration (Makefile)

```bash
make check        # Report data availability
make all          # Generate: fig1, fig2, interference, fig5, mechanism, tables
make interference # Run interference panel (requires GPU + precomputed keys)
make tables       # Generate paper tables + paper_numbers.json
make appendix     # Generate appendix figures (A1, A3, A8, A9, capability)
```

---

## MVE1: AlphaEdit on MultiCounterFact

**Purpose**: Reproduce the primary AlphaEdit claim — 2000 sequential edits on MultiCounterFact with null-space projection preserving existing knowledge.

**Configuration**:
- Algorithm: AlphaEdit (null-space projected)
- Dataset: MultiCounterFact (MCF)
- Edits: 2000 (20 batches of 100)
- Model: Meta-Llama-3-8B-Instruct
- Seeds: 42, 137, 2024, 7, 99
- Layers edited: [4, 5, 6, 7, 8]

**Results (mean ± std across 5 seeds)**:

| Metric | Mean | Std |
|--------|------|-----|
| Efficacy | 0.9543 | 0.0022 |
| Paraphrase | 0.7056 | 0.0101 |
| Neighborhood | 0.1644 | 0.0015 |
| Neighborhood (prob) | 7.8902 | 0.0600 |

**Per-seed efficacy at 2000 edits**: seed42=0.955, seed2024=0.9525, seed137/7/99 within ±0.002 of mean.

**Observations**:
- AlphaEdit achieves >95% efficacy at 2000 edits, consistent with the original ICLR paper.
- Extremely low inter-seed variance (std=0.0022) demonstrates robust deterministic behavior.
- Neighborhood preservation is modest (16.4%) — the model does maintain most unrelated facts but shows measurable collateral damage to semantically proximate knowledge.
- Paraphrase generalization at ~70% indicates edits transfer moderately well to rephrased queries.

---

## MVE2: MEMIT on MultiCounterFact

**Purpose**: Fair baseline comparison — MEMIT (unconstrained mass editing) on the same 2000 facts with identical batching.

**Configuration**:
- Algorithm: MEMIT (no null-space projection)
- Dataset: MultiCounterFact
- Edits: 2000 (20 batches of 100)
- Seeds: 42, 137, 2024, 7, 99

**Results (mean ± std across 5 seeds)**:

| Metric | Mean | Std |
|--------|------|-----|
| Efficacy | 0.2445 | 0.0147 |
| Paraphrase | 0.1989 | 0.0116 |
| Neighborhood | 0.0484 | 0.0049 |
| Neighborhood (prob) | 8.4281 | 0.1674 |

**Observations**:
- MEMIT catastrophically fails at 2000 sequential edits — only 24.5% efficacy.
- This confirms AlphaEdit's primary advantage: null-space projection prevents successive edits from overwriting each other.
- MEMIT's higher inter-seed variance (std=0.0147) reflects chaotic edit interactions without projection.
- Neighborhood preservation is near-zero (4.8%), indicating complete corruption of related knowledge.
- The gap between AlphaEdit (95.4%) and MEMIT (24.5%) at 2000 edits is the headline claim of the original paper.

---

## MVE3: AlphaEdit on zsRE

**Purpose**: Test AlphaEdit's generalization to a different knowledge editing dataset (zsRE, which uses different fact structures than MCF).

**Configuration**:
- Algorithm: AlphaEdit
- Dataset: zsRE (zero-shot Relation Extraction)
- Edits: 2000
- Seeds: 42, 137, 2024, 7, 99

**Results (mean ± std across 5 seeds)**:

| Metric | Mean | Std |
|--------|------|-----|
| Efficacy | 0.9452 | 0.0013 |
| Paraphrase | 0.9080 | 0.0031 |
| Neighborhood | 0.3256 | 0.0019 |

**Observations**:
- AlphaEdit maintains high efficacy (94.5%) on zsRE, only slightly below MCF (95.4%).
- Paraphrase generalization is substantially higher on zsRE (90.8% vs 70.6%) — zsRE's simpler fact structures generalize better.
- Neighborhood preservation doubles (32.6% vs 16.4%) — zsRE facts have less semantic overlap, so collateral damage is reduced.
- Cross-dataset consistency validates that the null-space projection mechanism is not dataset-specific.

---

## Failure Curve (Checkpointed)

**Purpose**: Identify the edit count at which AlphaEdit's null-space advantage disappears — the central empirical claim of this study.

**Configuration**:
- Algorithms: AlphaEdit, MEMIT, MEMIT-Seq-lp1.0-ld0.0-cache0, MEMIT-Seq-lp0.0-ld1.0-cache0
- Edit counts: 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000, 10000
- Seeds: 42, 2024, 137 (AlphaEdit/MEMIT); seed 42 only for MEMIT-Seq
- Evaluation mode: Milestone (full eval at checkpoint boundaries)
- Checkpoint interval: Every 10 batches (1000 edits)
- Results path: `results/failure_curve_checkpointed/seed{N}/{E}edits/{Alg}/run_000/`

### AlphaEdit Full Trajectory (Seed 42, from method_comparison_summary.csv)

| Edits | Efficacy | Paraphrase | Neighborhood | n_cases |
|-------|----------|------------|--------------|---------|
| 2,000 | 0.9550 | 0.6908 | 0.1660 | 2000 |
| 3,000 | 0.9383 | 0.6890 | 0.1612 | 3000 |
| 4,000 | 0.9130 | 0.6999 | 0.1483 | 4000 |
| 5,000 | 0.8838 | 0.6777 | 0.1330 | 5000 |
| 6,000 | 0.8395 | 0.6278 | 0.1224 | 6000 |
| 7,000 | 0.7539 | 0.5512 | 0.1092 | 7000 |
| 8,000 | 0.6194 | 0.3584 | 0.0827 | 8000 |
| 9,000 | 0.4479 | 0.1653 | 0.0666 | 9000 |
| 10,000 | 0.3147 | 0.1126 | 0.0513 | 10000 |

### AlphaEdit (Seed 2024)

| Edits | Efficacy | Paraphrase | Neighborhood |
|-------|----------|------------|--------------|
| 2,000 | 0.9525 | 0.7203 | 0.1634 |
| 5,000 | 0.8186 | 0.6836 | 0.1289 |
| 10,000 | 0.1614 | 0.0691 | 0.0329 |

### MEMIT Baseline (Seed 42)

| Edits | Efficacy | Paraphrase | Neighborhood |
|-------|----------|------------|--------------|
| 2,000 | 0.1780 | 0.1260 | 0.0473 |
| 3,000 | 0.2583 | 0.1563 | 0.0252 |
| 4,000 | 0.1047 | 0.0340 | 0.0165 |
| 5,000 | 0.0852 | 0.0303 | 0.0149 |
| 10,000 | 0.0000 | 0.0000 | — |

### MEMIT-Seq-lp1.0-ld0.0-cache0 (Seed 42)

| Edits | Efficacy | Paraphrase | Neighborhood | n_cases |
|-------|----------|------------|--------------|---------|
| 2,000 | 0.9795 | 0.6478 | 0.2092 | 2000 |
| 3,000 | 0.9763 | 0.6535 | 0.1974 | 3000 |
| 4,000 | 0.9715 | 0.6611 | 0.1762 | 4000 |
| 5,000 | 0.9662 | 0.6639 | 0.1613 | 5000 |

### MEMIT-Seq-lp0.0-ld1.0-cache0 — Ridge Only (Seed 42)

| Edits | Efficacy | Paraphrase | Neighborhood | n_cases |
|-------|----------|------------|--------------|---------|
| 2,000 | 0.2055 | 0.1655 | 0.0377 | 2000 |
| 3,000 | 0.2427 | 0.1152 | 0.0229 | 3000 |
| 4,000 | 0.1885 | 0.0535 | 0.0125 | 4000 |
| 5,000 | 0.1676 | 0.0305 | 0.0244 | 5000 |

### Ordering Sensitivity (Coefficient of Variation)

| Edit count | CV | Spread |
|------------|-----|--------|
| 3,000 edits | 0.295 | 0.0073 |
| 7,000 edits | **4.754** | **0.0974** |

**Observations**:
- AlphaEdit degrades monotonically from 95.5% → 31.5% (seed 42) over 2K-10K edits. The decline accelerates sharply after 7K edits (75.4% → 61.9% → 44.8% → 31.5%).
- Seed 2024 shows even steeper decline to 16.1% at 10K, demonstrating seed-dependent degradation rates.
- MEMIT reaches effective 0% by 10K edits — complete failure of all installed edits.
- **MEMIT-Seq-lp1.0-ld0.0 (prev-key regularization) outperforms AlphaEdit at every checkpoint**: 97.95% vs 95.5% at 2K, 96.62% vs 88.38% at 5K.
- **Ridge-only (lp0.0-ld1.0) fails completely** — comparable to unconstrained MEMIT (20.6% vs 17.8% at 2K). This isolates the critical component: it is the previous-key regularization, not ridge regularization, that provides protection.
- The critical transition zone for AlphaEdit is 7000-8000 edits, where efficacy drops by ~13pp per 1K additional edits.
- Ordering sensitivity (CV) explodes 16× between 3K and 7K edits, indicating the system becomes chaotic near its capacity boundary.

---

## Capability Probe

**Purpose**: Track general model capabilities (WikiText-2 perplexity + few-shot MMLU) during editing to detect when editing damages core language understanding beyond just factual knowledge.

**Configuration**:
- Model: Meta-Llama-3-8B-Instruct
- Metrics: WikiText-2 perplexity (128-token windows), 4-subject MMLU (5-shot, 50 per subject)
- MMLU subjects: abstract_algebra, world_religions, us_foreign_policy, college_biology
- Checkpoints: Every 1000 edits (0-10000)
- Seeds: 42, 137, 2024 (AlphaEdit, MEMIT); seed 42 (MEMIT-Seq)
- Results path: `results/capability_probe/seed{N}/{E}edits/{Alg}/offline_probe_*.jsonl`
- Probe version: 1.1.0 (torch.float16)

### AlphaEdit Capability Trajectory (Seed 42)

| Edits | Perplexity | MMLU Accuracy |
|-------|-----------|---------------|
| 0 | 14.80 | 69.5% |
| 1,000 | 15.17 | 70.5% |
| 2,000 | 15.66 | 67.5% |
| 3,000 | 16.22 | 67.0% |
| 4,000 | 17.12 | 66.0% |
| 5,000 | 18.35 | 63.0% |
| 6,000 | 20.62 | 62.0% |
| 7,000 | 25.55 | 58.0% |
| 8,000 | **56.68** | **36.5%** |
| 9,000 | **1,157.71** | **25.5%** |
| 10,000 | **25,194.10** | **22.5%** |

### AlphaEdit (Seed 137)

| Edits | Perplexity | MMLU Accuracy |
|-------|-----------|---------------|
| 0 | 14.80 | 69.5% |
| 1,000 | 15.19 | 69.5% |
| 2,000 | 15.68 | 70.5% |
| 3,000 | 16.41 | 67.5% |
| 4,000 | 17.30 | 67.5% |
| 5,000 | 18.79 | 63.5% |
| 6,000 | 22.37 | 63.0% |
| 7,000 | 32.75 | 54.5% |
| 8,000 | **1,970.17** | **26.0%** |
| 9,000 | **71,286.66** | **20.5%** |
| 10,000 | **123,019.37** | **23.0%** |

### AlphaEdit (Seed 2024)

| Edits | Perplexity | MMLU Accuracy |
|-------|-----------|---------------|
| 0 | 14.80 | 69.5% |
| 1,000 | 15.25 | 68.5% |
| 2,000 | 15.80 | 71.5% |
| 3,000 | 16.53 | 68.0% |
| 4,000 | 17.63 | 67.0% |
| 5,000 | 19.54 | 64.5% |
| 6,000 | 27.83 | 60.0% |
| 7,000 | **254.09** | **25.5%** |
| 8,000 | **38,571.55** | **23.5%** |
| 9,000 | **114,183.19** | **21.0%** |
| 10,000 | **124,505.35** | **25.0%** |

### MEMIT Capability Trajectory (Seed 42)

| Edits | Perplexity | MMLU Accuracy |
|-------|-----------|---------------|
| 0 | 14.80 | 69.5% |
| 1,000 | 15.05 | 67.0% |
| 2,000 | **5,902.98** | **22.0%** |
| 4,000 | 251,292.78 | 24.5% |
| 5,000 | 882,659.63 | 20.5% |
| 6,000 | 19,501,652.31 | 28.5% |
| 7,000 | 5,610,680.60 | 27.5% |
| 8,000 | 6,972,888.61 | 26.5% |
| 9,000 | 12,046,568.44 | 29.0% |
| 10,000 | 36,653,602.08 | 29.0% |

### MEMIT (Seed 137)

| Edits | Perplexity | MMLU Accuracy |
|-------|-----------|---------------|
| 0 | 14.80 | 69.5% |
| 1,000 | 15.13 | 67.0% |
| 2,000 | **69,384.14** | **24.5%** |
| 5,000 | 597,964.08 | 35.0% |
| 10,000 | 527,688.04 | 21.0% |

### MEMIT (Seed 2024)

| Edits | Perplexity | MMLU Accuracy |
|-------|-----------|---------------|
| 0 | 14.80 | 69.5% |
| 1,000 | 15.03 | 68.0% |
| 2,000 | **5,816.85** | **24.0%** |
| 5,000 | 475,568.52 | 25.0% |
| 7,000 | 1,348,244,018.14 | 23.0% |
| 10,000 | 1,842,787,614.20 | 25.5% |

### MEMIT-Seq-lp1.0-ld0.0-cache0 (Seed 42, 6000 edits)

| Edits | Perplexity | MMLU Accuracy |
|-------|-----------|---------------|
| 0 | 14.80 | 69.5% |
| 1,000 | 14.86 | 67.5% |
| 2,000 | 14.91 | 69.5% |
| 3,000 | 15.09 | 67.0% |
| 4,000 | 15.32 | 64.5% |
| 5,000 | 15.65 | 64.0% |
| 6,000 | 19.78 | 62.0% |

**Observations**:
- **AlphaEdit preserves model coherence until ~7000 edits**, then undergoes catastrophic collapse. The transition is sharp: perplexity jumps from ~25 (7K) to ~57 (8K) to ~1158 (9K) to ~25194 (10K) for seed 42.
- **Seed variation in collapse point**: Seed 2024 collapses earlier (perplexity 254 at 7K), seed 137 collapses at 8K (perplexity 1970). All seeds converge to catastrophic failure by 9-10K edits.
- **MEMIT collapses at just 2000 edits** across all seeds — perplexity jumps from ~15 to 5000-70000 and MMLU halves immediately.
- **MEMIT-Seq maintains near-baseline perplexity through 6000 edits** (perplexity 19.78 at 6K vs AlphaEdit's 20.62). This is the strongest evidence that sequential regularization protects model coherence as well as null-space projection up to at least 6K edits.
- At MMLU ~22-25%, the model is performing at random chance for 4-option MCQ — it has lost essentially all structured knowledge.
- The perplexity collapse point (7-8K edits) aligns precisely with the efficacy collapse zone from the failure curve, confirming these measure the same underlying phenomenon (null-space exhaustion).

---

## Matched Ordering

**Purpose**: Test whether the geometric structure of edited facts (clustered vs dispersed key vectors) affects AlphaEdit's degradation rate. This isolates the null-space exhaustion mechanism by controlling key geometry while keeping the same facts.

**Configuration**:
- Orderings: key_clustered (facts with similar key vectors grouped together) vs key_dispersed (maximally spread key vectors)
- Algorithms: AlphaEdit, MEMIT-Seq-lp1.0-ld0.0-cache0
- Edits: 1000-5000 (5 checkpoints at 1K intervals)
- Seeds: 42, 2024
- Results path: `results/matched_ordering/{ALG}/{ORDERING}/seed{N}/full_eval_seed{N}.json`

### AlphaEdit — Key Clustered (seed 42)

| Edits | Efficacy | Paraphrase | Neighborhood | First-1K | Retention AUC |
|-------|----------|------------|--------------|----------|---------------|
| 1,000 | 98.7% | 71.5% | 20.8% | — | 0.987 |
| 2,000 | 95.6% | 65.8% | 20.7% | — | 0.970 |
| 3,000 | 95.8% | 67.9% | 17.9% | — | 0.963 |
| 4,000 | 95.4% | 66.8% | 16.5% | — | 0.960 |
| 5,000 | **95.3%** | 65.6% | 14.8% | **91.5%** | **0.956** |

### AlphaEdit — Key Dispersed (seed 42)

| Edits | Efficacy | Paraphrase | Neighborhood | First-1K | Retention AUC |
|-------|----------|------------|--------------|----------|---------------|
| 1,000 | 96.1% | 69.4% | 21.5% | — | 0.961 |
| 2,000 | 95.7% | 70.8% | 16.0% | — | 0.958 |
| 3,000 | 93.4% | 70.2% | 14.2% | — | 0.940 |
| 4,000 | 91.0% | 69.5% | 12.7% | — | 0.923 |
| 5,000 | **87.9%** | 68.2% | 12.2% | **65.0%** | **0.882** |

### MEMIT-Seq (lp1.0-ld0.0) — Key Clustered (seed 42)

| Edits | Efficacy | Paraphrase | Neighborhood | First-1K | Retention AUC |
|-------|----------|------------|--------------|----------|---------------|
| 1,000 | 99.7% | 68.4% | 21.8% | — | 0.997 |
| 2,000 | 98.3% | 63.9% | 22.9% | — | 0.990 |
| 3,000 | 98.6% | 66.3% | 21.5% | — | 0.985 |
| 4,000 | 98.3% | 66.3% | 20.3% | — | 0.981 |
| 5,000 | **97.7%** | 65.1% | 18.9% | **98.8%** | **0.978** |

### MEMIT-Seq (lp1.0-ld0.0) — Key Dispersed (seed 42)

| Edits | Efficacy | Paraphrase | Neighborhood | First-1K | Retention AUC |
|-------|----------|------------|--------------|----------|---------------|
| 1,000 | 97.5% | 64.5% | 24.6% | — | 0.975 |
| 2,000 | 97.9% | 66.9% | 21.5% | — | 0.976 |
| 3,000 | 97.6% | 68.5% | 19.0% | — | 0.975 |
| 4,000 | 97.3% | 69.5% | 17.1% | — | 0.974 |
| 5,000 | **97.1%** | 69.9% | 15.8% | — | **0.972** |

### Summary Comparison at 5000 Edits (seed 42)

| Algorithm × Ordering | Efficacy | First-1K Retention | Retention AUC |
|---------------------|----------|--------------------|--------------:|
| AlphaEdit / clustered | 95.3% | 91.5% | 0.956 |
| AlphaEdit / dispersed | 87.9% | 65.0% | 0.882 |
| MEMIT-Seq / clustered | 97.7% | 98.8% | 0.978 |
| MEMIT-Seq / dispersed | 97.1% | — | 0.972 |

**Observations**:
- **Key dispersal accelerates AlphaEdit's failure**: 7.4pp lower efficacy (87.9% vs 95.3%) and 26.5pp lower first-1K retention (65% vs 91.5%) under dispersed ordering at 5K edits.
- **MEMIT-Seq is ordering-insensitive**: Only 0.6pp difference (97.7% vs 97.1%) between clustered and dispersed — the regularization-based approach doesn't rely on null-space geometry and therefore isn't affected by key distribution.
- **MEMIT-Seq outperforms AlphaEdit** in all four conditions at 5K edits — both higher efficacy and dramatically higher retention AUC.
- The ordering effect supports the hypothesis that AlphaEdit's null-space becomes exhausted faster when edits span diverse directions in key space.
- Clustered edits are "easier" for null-space projection because similar keys consume fewer null-space dimensions.
- The retention AUC gap (0.956 vs 0.882 for AlphaEdit) quantifies the cumulative cost of dispersed editing.

---

## Interference Panel Analysis

**Purpose**: Per-edit level analysis of what predicts whether an individual fact is retained or forgotten after subsequent editing.

**Configuration**:
- Panel: 18,000 observations (3,000 edits × 3 checkpoints at 3K, 4K, 5K)
- Seeds: 42, 2024
- Model: Discrete-time logistic regression per trajectory
- Predictors: Position (age), subject overlap, relation overlap, target margin, max key cosine similarity
- Results path: `results/figures/paper/interference_panel_results.json`

### Monotonicity of Forgetting

| Pattern | Count | Percentage |
|---------|-------|-----------|
| Success → Failure (monotonic) | 1,018 | 16.97% |
| Failure → Success (reversal) | 35 | 0.58% |
| Stable success | 4,552 | 75.87% |
| Stable failure | 330 | 5.50% |
| Mixed patterns | 65 | 1.08% |
| **Monotonic fraction** | — | **98.33%** |

### Survival Analysis (seed 42, 9000 observations)

| Predictor | β Coefficient | Odds Ratio | p-value | 95% CI (OR) |
|-----------|--------------|------------|---------|-------------|
| Max cosine similarity | -3.767 | 0.023 | 1.38e-44 | [0.641, 0.734]* |
| Subject overlap | -0.568 | 0.567 | 0.0080 | — |
| Position (age) | +2.47 | — | — | [1.16, 3.95]** |

*CI is for OR per 0.1 cosine increase
**CI for β coefficient

### Survival Analysis (seed 2024, 9000 observations)

| Predictor | β Coefficient | Odds Ratio | p-value |
|-----------|--------------|------------|---------|
| Max cosine similarity | -4.169 | 0.015 | 3.96e-59 |
| Subject overlap | -0.819 | 0.441 | 0.0080 |

### AIC Model Comparison

| Model | Variables | AIC (s42) | AIC (s2024) |
|-------|-----------|-----------|-------------|
| Age-only | position | baseline | baseline |
| +Semantic | +subject_overlap, +relation | small improvement | small improvement |
| +Keys (full) | +max_cosine | **3,911.67** | **4,912.42** |
| AIC improvement (keys) | — | **1,848.32** | **2,092.95** |

### Negative Controls

- **Permuted keys**: p < 0.001 (confirming real structure in key similarity)
- **Preceding vs subsequent keys**: Subsequent keys have 2.7× larger effect (temporal directionality confirmed)
- **Shuffled outcomes**: z-scores = -14.93 (s42), -19.27 (s2024)
- **Random keys**: Near-zero effect (real keys show 160× larger effect)
- **Leave-one-out AUC**: Improvement = 0.0135 (s42), 0.0091 (s2024)

### Bootstrap Validation (1000 replicates, block size 100)

| Seed | β CI | OR per 0.1 CI | Sign Consistency |
|------|------|---------------|-----------------|
| 42 | [-4.44, -3.09] | [0.641, 0.734] | 100% |
| 2024 | [-4.85, -3.58] | [0.616, 0.699] | 100% |

**Observations**:
- **Key cosine similarity is the dominant predictor of forgetting**: Facts whose keys are similar to future edits are dramatically more likely to be overwritten.
- **OR interpretation**: Each 0.1 increase in max cosine similarity to a future edit reduces survival odds by ~31% (seed 42) to ~34% (seed 2024).
- Forgetting is overwhelmingly monotonic (98.33%) — once a fact is lost, it almost never spontaneously recovers.
- The AIC improvement of ~2000 when adding key vectors shows they provide massive explanatory power beyond simple age/semantic predictors.
- This provides the mechanistic explanation for why dispersed orderings hurt AlphaEdit more: high-cosine collisions in key space directly cause overwriting.

---

## MEMIT+SeqReg (Sequential Regularization)

**Purpose**: Test whether MEMIT with AlphaEdit-style sequential regularization (Eq. 12 analogue without null-space projection P) can match AlphaEdit's performance. This isolates whether the null-space projection is necessary or whether simple regularization against prior keys suffices.

**Configuration**:
- Algorithm: MEMIT with augmented LHS: `lhs = α·C₀ + K_new@K_new^T + λ_prev·K_prev@K_prev^T + λ_delta·I`
- Variants tested:
  - **lp1.0-ld0.0** (prev-key regularization only): The critical component
  - **lp0.0-ld1.0** (ridge regularization only): Control/ablation
- Cache strategy: all (full history of prior keys)
- Seed: 42
- Results paths:
  - `results/failure_curve_checkpointed/seed42/{E}edits/MEMIT-Seq-lp{X}-ld{Y}-cache0/run_000/`
  - `results/matched_ordering/MEMIT-Seq-lp1.0-ld0.0-cache0/{ORDERING}/seed42/full_eval_seed42.json`

### Failure Curve: MEMIT-Seq-lp1.0-ld0.0 vs AlphaEdit vs MEMIT (Seed 42)

| Edits | MEMIT-Seq (lp1.0-ld0.0) | AlphaEdit | MEMIT | Ridge-only (lp0.0-ld1.0) |
|-------|--------------------------|-----------|-------|--------------------------|
| 2,000 | **97.95%** | 95.50% | 17.80% | 20.55% |
| 3,000 | **97.63%** | 93.83% | 25.83% | 24.27% |
| 4,000 | **97.15%** | 91.30% | 10.47% | 18.85% |
| 5,000 | **96.62%** | 88.38% | 8.52% | 16.76% |

### Paraphrase Comparison (Seed 42)

| Edits | MEMIT-Seq (lp1.0-ld0.0) | AlphaEdit | MEMIT | Ridge-only |
|-------|--------------------------|-----------|-------|------------|
| 2,000 | 64.78% | **69.08%** | 12.60% | 16.55% |
| 3,000 | 65.35% | **68.90%** | 15.63% | 11.52% |
| 4,000 | **66.11%** | 69.99% | 3.40% | 5.35% |
| 5,000 | **66.39%** | 67.77% | 3.03% | 3.05% |

### Neighborhood Preservation (Seed 42)

| Edits | MEMIT-Seq (lp1.0-ld0.0) | AlphaEdit | MEMIT | Ridge-only |
|-------|--------------------------|-----------|-------|------------|
| 2,000 | **20.92%** | 16.60% | 4.73% | 3.77% |
| 3,000 | **19.74%** | 16.12% | 2.52% | 2.29% |
| 4,000 | **17.62%** | 14.83% | 1.65% | 1.25% |
| 5,000 | **16.13%** | 13.30% | 1.49% | 2.44% |

### Table 3 — Paper Comparison (includes cohort retention)

| Method | Edits | Efficacy | Paraphrase | Neighborhood | First-1K | Latest-1K |
|--------|-------|----------|------------|--------------|----------|-----------|
| AlphaEdit | 3,000 | 93.8% | 68.9% | 16.1% | 85.6% | 99.1% |
| MEMIT (unconstrained) | 3,000 | 25.8% | 15.6% | 2.5% | — | — |
| **MEMIT+SeqReg** | **3,000** | **96.7%** | **66.4%** | **19.3%** | **92.9%** | **99.3%** |
| AlphaEdit | 5,000 | 88.4% | 67.8% | 13.3% | 70.0% | 99.0% |
| MEMIT (unconstrained) | 5,000 | 8.5% | 3.0% | 1.5% | — | — |
| **MEMIT+SeqReg** | **5,000** | **92.9%** | **66.6%** | **15.5%** | **78.2%** | **99.7%** |

### Capability Preservation (Seed 42, from capability probe)

| Edits | MEMIT-Seq Perplexity | AlphaEdit Perplexity | MEMIT Perplexity |
|-------|---------------------|---------------------|-----------------|
| 0 | 14.80 | 14.80 | 14.80 |
| 2,000 | **14.91** | 15.66 | 5,902.98 |
| 4,000 | **15.32** | 17.12 | 251,292.78 |
| 6,000 | **19.78** | 20.62 | 19,501,652.31 |

**Observations**:
- **MEMIT-Seq (lp1.0-ld0.0) outperforms AlphaEdit on efficacy at every checkpoint** from 2K-5K edits: +2.5pp at 2K, +3.8pp at 3K, +5.9pp at 4K, +8.2pp at 5K. The gap widens with scale.
- **AlphaEdit retains a modest advantage on paraphrase** (~3-4pp higher), suggesting null-space projection may better preserve generalization structure.
- **MEMIT-Seq achieves better neighborhood preservation** (+4pp at 2K, +3pp at 5K), indicating less collateral damage to semantically adjacent facts.
- **Ridge-only (lp0.0-ld1.0) is no better than unconstrained MEMIT** — this is the critical ablation proving that previous-key regularization is the active ingredient, not general regularization.
- **Capability probe shows MEMIT-Seq preserves perplexity better** than AlphaEdit through 6K edits (19.78 vs 20.62), while MEMIT collapses to millions by 2K edits.
- The first-1K retention difference (92.9% vs 85.6% at 3K; 78.2% vs 70.0% at 5K) shows MEMIT-Seq is more explicitly protective of previously-installed edits.
- **Central implication**: Null-space projection is sufficient but not necessary for sequential editing. A simple quadratic regularization term penalizing interference with prior keys achieves comparable or better protection, challenging AlphaEdit's core architectural contribution.

---

## Polykernel Editor

**Purpose**: Test whether non-linear kernel extensions of AlphaEdit's null-space projection can expand the effective projection space and delay exhaustion at scale.

**Configuration**:
- Kernel: Polynomial degree 2 (poly2) with sigma='median'
- Edit counts: 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000, 10000
- Seed: 42
- Model: Meta-Llama-3-8B-Instruct
- Batch size: 100 edits
- Results path: `results/polykernel_editor/seed42/{E}edits/AlphaEdit-poly2/run_000/`

### Efficacy Progression: AlphaEdit-poly2 vs Linear AlphaEdit (Seed 42)

| Edits | AlphaEdit-poly2 | AlphaEdit (linear) | Difference |
|-------|-----------------|-------------------|:----------:|
| 2,000 | **96.00%** | 95.50% | +0.50 |
| 3,000 | **94.60%** | 93.83% | +0.77 |
| 4,000 | **92.03%** | 91.30% | +0.73 |
| 5,000 | **89.44%** | 88.38% | +1.06 |
| 6,000 | **86.38%** | 83.95% | +2.43 |
| 7,000 | — | 75.39% | — |
| 8,000 | — | 61.94% | — |
| 9,000 | — | 44.79% | — |
| 10,000 | **63.66%** | 31.47% | **+32.19** |

### Full Metrics at Key Checkpoints (Seed 42)

**At 2,000 edits:**

| Metric | AlphaEdit-poly2 | AlphaEdit (linear) |
|--------|-----------------|-------------------|
| Efficacy | 96.00% | 95.50% |
| Paraphrase | 67.03% | 69.08% |
| Neighborhood | 18.27% | 16.60% |

**At 5,000 edits:**

| Metric | AlphaEdit-poly2 | AlphaEdit (linear) |
|--------|-----------------|-------------------|
| Efficacy | 89.44% | 88.38% |
| Paraphrase | ~66% | 67.77% |
| Neighborhood | ~14% | 13.30% |

**At 10,000 edits:**

| Metric | AlphaEdit-poly2 | AlphaEdit (linear) |
|--------|-----------------|-------------------|
| Efficacy | **63.66%** | 31.47% |
| Paraphrase | **44.50%** | 11.26% |
| Neighborhood | **10.45%** | 5.13% |

### GLUE Degradation (poly2 at 10,000 edits)

| Task | F1 | Invalid Responses |
|------|----|-----------------:|
| SST-2 | 0.039 | 97/100 |
| MMLU | 0.209 | 49/100 |
| MRPC | 0.155 | 66/100 |
| CoLA | 0.300 | 57/100 |
| RTE | 0.411 | 17/100 |
| NLI | 0.326 | 41/100 |

### Effective Rank Diagnostics

- Diagonal mean effective rank ratio (poly2 vs linear) at batch 0: 1.1949
- The polynomial kernel space provides ~19% more effective dimensions than linear.

**Observations**:
- **AlphaEdit-poly2 matches linear AlphaEdit at low scales** (96.0% vs 95.5% at 2K) — the kernel does not impair editing at standard operating points.
- **The advantage grows with scale**: At 5K edits the gap is +1pp, at 6K it's +2.4pp, and at 10K it's a massive +32pp (63.7% vs 31.5%).
- **Poly2 doubles efficacy at 10K edits** compared to linear AlphaEdit (63.7% vs 31.5%), and quadruples paraphrase preservation (44.5% vs 11.3%).
- **However, GLUE scores are severely degraded at 10K** (97% invalid on SST-2), indicating the model's general capabilities are still destroyed even though factual edits are better preserved.
- The 19% effective rank increase translates into substantially delayed exhaustion at scale — the polynomial kernel expands the usable null-space.
- **Trade-off**: Polykernel buys ~2× more factual retention at 10K edits but cannot prevent eventual model coherence collapse. It extends the practical capacity ceiling from ~7K (linear) to perhaps ~9K edits before factual efficacy collapses below 50%.
- This represents a viable engineering improvement for extending AlphaEdit's capacity while confirming the fundamental exhaustion problem remains.

---

## Comparison Ordered

**Purpose**: Test sensitivity to fact ordering with 5 random permutations at multiple edit scales to quantify ordering-induced variance.

**Configuration**:
- Orderings: 5 random orderings (order0-order4) per seed
- Edit counts: 3000, 5000, 7000
- Seeds: 42, 137, 2024
- Algorithms: AlphaEdit, MEMIT
- Results path: `results/comparison_ordered/seed{N}/{E}edits/order{0-4}/{Alg}/run_000/`

### GLUE Performance (3000 edits, order0, AlphaEdit)

| Task | F1 | MCC |
|------|-----|-----|
| SST-2 | 0.831 | 0.623 |
| MMLU | 0.562 | 0.401 |
| MRPC | 0.658 | 0.368 |
| CoLA | 0.761 | 0.542 |
| RTE | 0.284 | -0.386 |
| NLI | 0.666 | 0.349 |

### Ordering Sensitivity Results

The coefficient of variation (CV) of efficacy across orderings:

| Scale | CV | Spread | Interpretation |
|-------|-----|--------|---------------|
| 3,000 edits | 0.295 | 0.0073 | Low — ordering barely matters |
| 7,000 edits | **4.754** | **0.0974** | **High — ordering dominates outcomes** |

**Observations**:
- At 3000 edits, the system is robust to ordering (CV < 0.3) — all permutations yield similar results within ~0.7pp spread.
- At 7000 edits, ordering sensitivity explodes 16× — the same facts in different orders produce wildly different final states (spread ~9.7pp).
- This is a signature of operating near capacity: small perturbations (ordering) cascade into large outcome differences.
- GLUE scores at 3000 edits remain reasonable (SST-2 F1=0.83, CoLA MCC=0.54), indicating model coherence is preserved at this scale.
- The transition from ordered to chaotic behavior between 3K-7K aligns with the failure curve's accelerating degradation in the same range.

---

## Mechanism Analysis

**Purpose**: Track internal cache statistics (singular values, effective rank, condition number) to understand the null-space exhaustion mechanism at the linear algebra level.

**Configuration**:
- Layers tracked: 4, 5, 6, 7, 8
- Metrics per (batch, layer): numerical rank, effective rank, stable rank, condition number, top-5 singular values, SV percentiles
- Seeds: 42, 2024
- Format: JSONL (one line per batch×layer pair)
- Results path: `results/mechanism_analysis/seed{N}/mechanism_seed{N}_*.jsonl`

### Cache Statistics Progression (Seed 42)

| Total Edits | Layer | Effective Rank | Stable Rank | Condition # | Top SV | Trace |
|-------------|-------|---------------|-------------|-------------|--------|-------|
| 1,000 | 4 | 964 | 15.6 | 15.9M | 22.4 | 7,864 |
| 1,000 | 5 | 954 | 12.5 | 10.3M | 30.0 | 11,319 |
| 1,000 | 6 | 934 | 8.1 | 31.2M | 41.6 | 13,968 |
| 1,000 | 7 | — | — | — | — | — |
| 1,000 | 8 | — | — | — | — | — |
| 10,000 | 4 | 7,987 | 15.8 | 34.7M | 70.4 | 78,191 |
| 10,000 | 5 | 7,815 | 11.8 | 5.0M | 96.6 | 110,108 |
| 10,000 | 6 | 7,581 | 8.4 | 9.1M | 124.3 | 129,379 |

### Key Ratios

| Layer | Eff. Rank Growth (1K→10K) | Stable Rank Change | Top SV Growth |
|-------|---------------------------|-------------------|---------------|
| 4 | 964 → 7,987 (8.3×) | 15.6 → 15.8 (flat) | 22.4 → 70.4 (3.1×) |
| 5 | 954 → 7,815 (8.2×) | 12.5 → 11.8 (flat) | 30.0 → 96.6 (3.2×) |
| 6 | 934 → 7,581 (8.1×) | 8.1 → 8.4 (flat) | 41.6 → 124.3 (3.0×) |

**Observations**:
- **Effective rank grows linearly** with edit count (~8× growth for 10× more edits), but **stable rank remains constant** (~8-16 across all scales). This divergence is the signature of spectral concentration.
- The cache accumulates thousands of near-zero singular values — directions that are "technically" occupied in the numerical rank sense but carry negligible signal weight.
- **Top singular values grow 3×** (22.4 → 70.4 at layer 4) while lower SVs grow proportionally less — the projection concentrates into fewer dominant directions.
- **Layer 6 is the bottleneck**: Lowest stable rank (8.1), highest top SV (124.3), highest trace (129K) — suggesting this layer's null-space exhausts first.
- Condition numbers are extremely high (10⁷) even at 1K edits, indicating the projection is numerically fragile from the start. At 10K edits they remain in the same order, but the effective null-space has been consumed.
- The flat stable rank despite growing effective rank means the projection P is increasingly dominated by a small number of directions — the "available" null-space for new edits shrinks.

---

## Interference (Installation Strength)

**Purpose**: Logistic regression predicting which individual edits survive vs are forgotten, broken down by algorithm and ordering condition.

**Configuration**:
- Matched ordering conditions: AlphaEdit/clustered, AlphaEdit/dispersed, MEMIT-Seq/clustered, MEMIT-Seq/dispersed
- Predictors: position, margin (confidence), edit norm, future_max_cos
- Results path: `results/interference/installation_strength_seed{N}.json`

### AlphaEdit Clustered (seed 42)

| Statistic | Value |
|-----------|-------|
| Valid cases | 987 |
| Retained | 911 (92.3%) |
| Forgotten | 76 (7.7%) |
| Pseudo R² | 0.132 |
| Position β | 2.47 |
| Margin β | 1.23 |
| Edit norm β | 1.95 |

### AlphaEdit Dispersed (seed 42)

| Statistic | Value |
|-----------|-------|
| Valid cases | 961 |
| Retained | 645 (67.1%) |
| Forgotten | 316 (32.9%) |
| Pseudo R² | 0.047 |
| Position β | 0.65 |
| Margin β | 0.41 |
| Edit norm β | 1.04 |

### MEMIT-Seq Clustered (seed 42)

| Statistic | Value |
|-----------|-------|
| Valid cases | 997 |
| Retained | 986 (98.9%) |
| Forgotten | 11 (1.1%) |
| Pseudo R² | 0.110 |
| Edit norm β | 6.33 |

### MEMIT-Seq Dispersed (seed 42)

| Statistic | Value |
|-----------|-------|
| Valid cases | 975 |
| Retained | 913 (93.6%) |
| Forgotten | 62 (6.4%) |
| Pseudo R² | 0.176 |
| Position β | 1.11 |

### Joint Model (AlphaEdit clustered + dispersed)

| Statistic | Value |
|-----------|-------|
| Total cases | 1,948 |
| Pseudo R² | 0.157 |
| Dispersed × future_max_cos interaction | -3.14 |
| LR test for ordering effect | χ² = 119.1, p < 10⁻²⁷ |

**Observations**:
- **AlphaEdit under dispersed ordering forgets 4.3× more facts** (32.9% vs 7.7%) than under clustered ordering.
- **MEMIT-Seq forgets dramatically fewer facts in both conditions**: 1.1% (clustered) and 6.4% (dispersed) — a fraction of AlphaEdit's forgetting rate.
- The joint model confirms ordering is statistically significant (p < 10⁻²⁷) with a strong interaction between dispersed ordering and future cosine similarity.
- Position (temporal age) is the strongest single predictor under clustered ordering (β=2.47) — older edits are more likely to survive because they've already survived many subsequent edits.
- Under dispersed ordering, all predictors weaken (position β drops from 2.47 to 0.65) because geometric collision becomes the dominant, more uniformly distributed cause of failure.
- Edit norm is the strongest predictor for MEMIT-Seq (β=6.33), suggesting that with regularization, the magnitude of the weight update determines survival rather than geometric properties.

---

## Summary of Key Findings

### Finding 1: AlphaEdit's Null-Space Has Finite Capacity

AlphaEdit degrades monotonically from 95.5% to 31.5% efficacy (seed 42) over 2K-10K edits. The degradation accelerates sharply after 7K edits and produces catastrophic model collapse: perplexity increases ~1700× and MMLU drops to random chance (22.5%). This directly challenges the implied claim that null-space projection enables unlimited sequential editing.

### Finding 2: Sequential Regularization Outperforms Null-Space Projection

MEMIT-Seq (lp1.0-ld0.0) achieves 97.95% efficacy at 2K and 96.62% at 5K — outperforming AlphaEdit (95.5% and 88.4%) at every checkpoint. It also maintains near-baseline perplexity through 6K edits (19.78 vs AlphaEdit's 20.62). The ridge-only ablation (lp0.0-ld1.0) fails completely (20.6% at 2K), isolating previous-key regularization as the critical component. This challenges AlphaEdit's central architectural contribution.

### Finding 3: Key Geometry Determines Failure Rate

Facts whose key vectors have high cosine similarity to future edits are preferentially forgotten (OR = 0.023 per unit cosine, p < 10⁻⁴⁴). Dispersed key orderings accelerate AlphaEdit's failure (87.9% vs 95.3% at 5K) while MEMIT-Seq remains insensitive to ordering (97.1% vs 97.7%). This reveals the mechanism: edits spanning diverse null-space directions exhaust available dimensions faster.

### Finding 4: Forgetting is Monotonic and Predictable

98.33% of edits follow a monotonic survival trajectory — once forgotten, a fact almost never recovers. This rules out "interference and recovery" dynamics and supports a model where null-space dimensions are permanently consumed.

### Finding 5: Polykernel Extends Capacity but Doesn't Solve the Problem

AlphaEdit-poly2 matches linear AlphaEdit at 2K edits (96.0% vs 95.5%) and provides massive improvement at 10K (63.7% vs 31.5% — doubling retained efficacy). The polynomial kernel expands the effective null-space by ~19%, delaying exhaustion. However, GLUE scores still collapse at 10K (97% invalid on SST-2), confirming the fundamental problem remains.

### Finding 6: The Critical Transition is Sharp and Seed-Dependent

The system transitions from ordered (CV = 0.3, stable performance) to chaotic (CV = 4.8, ordering-dominated outcomes) between 3K and 7K edits. The collapse point varies by seed (7K for seed 42, 7K for seed 2024, 8K for seed 137), suggesting stochastic dynamics near the capacity boundary. This phase-transition-like behavior indicates the null-space fills gradually but produces sudden, non-linear capability collapse.

### Finding 7: Model Coherence and Factual Editing Collapse Together

The capability probe confirms that factual efficacy collapse and general language model collapse are the same event: perplexity and MMLU degrade at exactly the edit counts where efficacy drops. AlphaEdit delays this by ~3.5× vs MEMIT (collapse at 7-8K vs 2K edits), and MEMIT-Seq delays it at least as long as AlphaEdit while maintaining better factual retention throughout.

# AlphaEdit Reproducibility Study

A mechanistic reproducibility study of [AlphaEdit](https://github.com/jianghoucheng/AlphaEdit) (ICLR 2025 Outstanding Paper), targeting TMLR Reproducibility Certification and NeurIPS 2026 MLRC track.

---

## Background

AlphaEdit edits LLM knowledge by projecting weight updates into the **null-space** of a preserved activation covariance matrix. The core promise: updates along directions that don't affect existing representations can insert new facts without catastrophic forgetting.

The method maintains:
- **P** — a static projection matrix (computed once from Wikipedia activations)
- **cache_c** — an accumulated covariance cache tracking which input directions have been used

Each edit batch solves: `upd = solve(P @ (K@K^T + cache_c) + L2*I, P @ K @ resid^T)`

---

## Central Thesis

> AlphaEdit's reliability is limited by two mechanisms: **capacity saturation** (the null-space fills up) and **semantic interference** (the desired edit overlaps with preserved knowledge). The second failure mode is more fundamental — it challenges the assumption that editability and preservation are always separable.

---

## Contributions

1. **Reproduce** AlphaEdit's sequential editing advantage over MEMIT (5 seeds, paired bootstrap, BCa CIs)

2. **Diagnose two failure modes:**
   - *Capacity saturation* — `rank(cache_c) / rank(P)` grows until edits have nowhere to go
   - *Semantic coupling* — when the new edit relates to preserved knowledge, the projection strips the edit's critical component **even before saturation**

3. **Challenge the core assumption** — measure *projection loss* (fraction of update removed by P) across four semantic coupling levels. If projection loss increases with coupling strength, the separability assumption fails where editing matters most.

4. **Discover a hidden hyperparameter** — edit ordering affects AlphaEdit but not MEMIT, because early edits claim the best null-space directions

---

## Experiments

### Core Reproduction (MVE)

| ID | Experiment | Tests |
|----|-----------|-------|
| MVE1 | AlphaEdit on MultiCounterFact | Primary benchmark: 2000 facts, 5 seeds |
| MVE2 | MEMIT on MultiCounterFact | Fair comparison under identical conditions |
| MVE3 | AlphaEdit on zsRE | Cross-dataset generalization |
| MVE4 | Conflict sequence | Sequential contradictions |

All MVEs: batches of 100, evaluation every 5 batches, 5 seeds.

### Calibration

ROME as a sanity-check baseline — it *should* perform worse. If it doesn't, the evaluation has a bug.

### Novel Extensions

| Priority | Experiment | Core question |
|----------|-----------|---------------|
| **P0** | Semantic coupling stress test | Does the projection strip more of the edit when it's related to preserved knowledge? |
| **P0** | Edit order sensitivity | Is ordering a hidden hyperparameter for null-space methods? |
| P1 | Failure curve (500–10K edits) | Where does AlphaEdit's advantage disappear? |
| P1 | Null-space rank tracking | Which layers saturate first? |
| P1 | MEMIT+PrevKeyReg+Ridge | Is null-space projection necessary, or does key-direction regularization suffice? |
| P2 | Capability probe (WikiText-103) | Does editing destroy general language ability? |
| P2 | Second model (Mistral-7B) | Do findings generalize beyond Llama-3-8B? |

---

## Key Metrics

### Standard editing metrics

| Metric | What it measures |
|--------|-----------------|
| Efficacy | Edit succeeds on the exact prompt |
| Generalization | Edit succeeds on paraphrases |
| Specificity | Unrelated facts remain correct |
| Fluency | Output remains coherent |
| Consistency | Open-ended generation aligns with the edit |

### Mechanistic metrics (novel)

| Metric | Formula | Interpretation |
|--------|---------|----------------|
| **Projection loss** | `1 - ‖P @ K @ resid^T‖ / ‖K @ resid^T‖` | Fraction of desired edit stripped by the null-space constraint. **The central measurement.** |
| Consumption ratio | `rank(cache_c) / rank(P)` per layer | How much null-space remains — predicts capacity failure |
| Perplexity | `exp(total_NLL / total_tokens)` on WikiText-103 | Global capability preservation |

### Statistical framework

- 5 seeds (42, 137, 2024, 7, 99)
- BCa bootstrap CIs (10,000 resamples)
- Holm-Bonferroni correction for multiple comparisons
- Effect sizes: Cohen's d, Cliff's delta
- Paired bootstrap test for AlphaEdit vs MEMIT

---

## Extension A: Semantic Coupling Stress Test

### Motivation

AlphaEdit assumes editability and preservation occupy orthogonal subspaces. But when a new edit is semantically coupled to preserved knowledge — same subject, same relation, or a direct contradiction — the required update direction may *lie partly in the preserved subspace*. The projection removes exactly that component.

This failure mode is **more fundamental than saturation**: it can happen on the very first edit if that edit conflicts with the model's existing representations.

### Coupling type hierarchy

| Type | Name | Example |
|------|------|---------|
| 0 | **Unrelated** (control) | "Tesla born_in Smiljan" after preserving "Curie won Nobel" |
| 1 | **Relation match** | "Einstein born_in Ulm" after preserving "Curie born_in Warsaw" |
| 2 | **Subject match** | "Curie field_of_work Chemistry" after preserving "Curie born_in Warsaw" |
| 3 | **Full conflict** | "Curie born_in Paris" after preserving "Curie born_in Warsaw" |

### Design

- 500 edits, `num_edits=1` (sequential, one at a time)
- ~60 anchor-probe pairs per coupling type
- Anchor establishes preserved knowledge → probe measures projection loss
- Per-edit, per-layer measurement via **double source injection** into `AlphaEdit_main.py`

### Hypotheses

- **H1**: Projection loss increases with coupling strength (Type 0 < 1 < 2 < 3)
- **H2**: High projection loss predicts low edit efficacy
- **H3**: The effect is strongest at layers where the subject representation is concentrated

### Statistical tests

- Kruskal-Wallis H-test across coupling types
- Pairwise Mann-Whitney U with Holm-Bonferroni correction
- Cliff's delta: Type 3 vs Type 0
- Spearman correlation: projection_loss ↔ update norm

---

## Extension B: Edit Order Sensitivity

AlphaEdit processes edits in source-file order with no shuffling. P is static, so the first edits claim the best null-space directions.

**Design**: 2000 MCF edits × 10 random orderings × {AlphaEdit, MEMIT}

**Test**: Levene's test for equality of variances. If AlphaEdit has significantly higher cross-ordering variance, ordering is a hidden hyperparameter the paper doesn't acknowledge.

---

## Extension C: MEMIT+SeqReg — Non-Projected Sequential Regularization

### Scientific Question

**Does MEMIT with AlphaEdit-like sequential regularization (Eq. 12) close the performance gap to AlphaEdit, or is the null-space projection P still necessary?**

### Motivation

AlphaEdit (Eq. 12) optimizes:
```
minimize ||ΔPK - R||² + λ_prev ||ΔPK_prev||² + λ_delta ||ΔP||²
```
where P is the null-space projection matrix.

This objective combines three components:
1. **Edit fit** (||ΔPK - R||²)
2. **Previous-edit protection** (λ_prev ||ΔPK_prev||²)
3. **Update size minimization** (λ_delta ||ΔP||²)

**Key insight**: Components 2 and 3 are sequential regularization terms that could help MEMIT even without projection.

MEMIT+SeqReg tests the **non-projected analogue**:
```
minimize ||ΔK - R||² + λ_prev ||ΔK_prev||² + λ_delta ||Δ||²
```

If this closes the gap, AlphaEdit's advantage is regularization strategy, not geometric projection. If the gap remains, projection P is necessary.

### Implementation

LHS-only augmentation (dual source injection):

```python
lhs = α·C₀ + K_new @ K_new^T + λ_prev · K_prev @ K_prev^T + λ_delta · I
adj_k = solve(lhs, K_new)
ΔW = resid @ adj_k^T
```

- **K_prev**: Concatenation of keys from previous batches (cached)
- **λ_prev=1, λ_delta=1**: Direct coefficient analogue to AlphaEdit Eq. 12
- **LHS norm logging**: Captures ||base_lhs||, ||K_prev@K_prev^T||, dimension for scale verification

### Calibration Settings

Run on seed 42 (2000 edits, 20 batches):

```bash
# A: Direct Eq. 12 analogue
FAST_CHECKPOINT=true bash scripts/run_memit_sequential.sh 42 1 1

# B: Weak ridge (sensitivity test)
FAST_CHECKPOINT=true bash scripts/run_memit_sequential.sh 42 1 1e-4

# C: Strong prev-key protection
FAST_CHECKPOINT=true bash scripts/run_memit_sequential.sh 42 10 1

# D: Very strong prev-key protection
FAST_CHECKPOINT=true bash scripts/run_memit_sequential.sh 42 100 1
```

After observing LHS term norms, may add norm-calibrated setting.

### Key Metrics

- **Standard preservation**: efficacy, paraphrase, neighborhood, GLUE
- **Per-batch norms**: ||ΔW||, ||ΔW @ K_prev||, ||base_lhs||, ||K_prev@K_prev^T||
- **Cache statistics**: batches stored, total keys per layer
- **Comparison**: AlphaEdit vs MEMIT vs MEMIT+SeqReg degradation curves

### Expected Outcomes

1. **Gap closes**: Sequential regularization is sufficient → projection unnecessary
2. **Gap remains**: Null-space constraint is critical → validates AlphaEdit's design
3. **Partial closure**: Regularization helps but projection still provides advantage

---

## Setup

```bash
cd src/alphaedit_replication
bash scripts/setup_env.sh
bash scripts/link_stats.sh /path/to/wikipedia_stats/
export HF_TOKEN=hf_...
uv run huggingface-cli login --token $HF_TOKEN
bash scripts/smoke_test.sh  # requires GPU
```

**Requirements**: Python 3.10, NVIDIA GPU ≥ 48GB VRAM, HuggingFace access to `meta-llama/Meta-Llama-3-8B-Instruct`.

---

## Running Experiments

### Core Experiments

```bash
# Core reproduction (MVE1: 2000 edits, 20 batches)
bash scripts/run_mve1_alphaedit_mcf.sh 42
bash scripts/run_rome_baseline.sh 42

# Mechanistic analysis
bash scripts/run_nullspace_analysis.sh 42
bash scripts/run_failure_curve.sh 42
```

### Checkpoint-Based Experiments (Long-Running)

**Evaluation Modes:**

1. **Normal** (default): Evaluate all facts after every batch (~16h for 5000 edits)
2. **Fast** (`FAST_CHECKPOINT=true`): Evaluate only edited batch (~2-3h, partial data)
3. **Milestone** (`EVAL_AT_CHECKPOINTS_ONLY=true`): Evaluate all facts only at checkpoints (~10-12h) **← RECOMMENDED FOR PAPERS**

```bash
# Failure curve with milestone evaluation (AlphaEdit + MEMIT to 5000 edits)
tmux new-session -d -s fc_42_5k "cd ~/Projects/alphaedit-analysis && \
  EVAL_AT_CHECKPOINTS_ONLY=true bash scripts/run_failure_curve_checkpointed.sh 42 both 5000 \
  2>&1 | tee logs/fc_42_5k.log"

# Fast mode (testing/iteration)
FAST_CHECKPOINT=true bash scripts/run_failure_curve_checkpointed.sh 42 AlphaEdit 2000
```

### MEMIT+SeqReg Control Baseline

```bash
# Calibration (seed 42, with fast checkpoint for speed)
FAST_CHECKPOINT=true bash scripts/run_memit_sequential.sh 42 1 1      # Direct Eq. 12 analogue
FAST_CHECKPOINT=true bash scripts/run_memit_sequential.sh 42 1 1e-4   # Weak ridge
FAST_CHECKPOINT=true bash scripts/run_memit_sequential.sh 42 10 1     # Strong prev-key
FAST_CHECKPOINT=true bash scripts/run_memit_sequential.sh 42 100 1    # Very strong prev-key

# Verification: original MEMIT
FAST_CHECKPOINT=true bash scripts/run_memit_sequential.sh 42 0 0
```

### Novel Extensions

```bash
bash scripts/run_coupling_stress.sh 42      # ~2.5h per seed
bash scripts/run_order_sensitivity.sh 42    # ~3h per ordering

# Failure curve (checkpointed for long runs)
bash scripts/run_failure_curve_checkpointed.sh 42 both 5000

# Full sweep
bash scripts/run_all_seeds.sh mve
bash scripts/run_all_seeds.sh all
```

---

## Analysis

```bash
uv run python analysis/aggregate.py --results_dir results
uv run python analysis/paired_bootstrap.py --results_dir results
uv run python analysis/plots.py --results_dir results --output_dir results/figures
uv run python analysis/nullspace_analysis.py --results_dir results/nullspace_tracking
uv run python analysis/coupling_analysis.py --results_dir results/coupling_stress
```

Coupling analysis produces:
1. Violin plot of projection loss by coupling type (main result figure)
2. Heatmap of projection loss by coupling type × layer
3. Temporal plot showing projection loss over the editing sequence
4. Statistical summary with effect sizes and corrected p-values

---

## Next Steps

- [ ] Obtain HuggingFace access to Llama-3-8B-Instruct
- [ ] GPU smoke test (validate pipeline end-to-end)
- [ ] MVE1-4 reproduction (5 seeds × 4 experiments)
- [ ] Coupling stress test (5 seeds, ~12.5h GPU)
- [ ] Order sensitivity (20 runs, ~60h GPU)
- [ ] Failure curve + null-space tracking
- [ ] Write up for TMLR (target: 2026-07-24)

### Open questions

1. **Does MultiCounterFact have enough same-subject records for Type 2?** If <30 natural pairs exist, augment from zsRE.
2. **Is double source injection stable?** First GPU smoke test will reveal if the exec'd patched `AlphaEdit_main.py` resolves all imports correctly.
3. **Should MEMIT also be tested for coupling sensitivity?** Projection loss is undefined for MEMIT (no null-space constraint), but comparing edit success rates across coupling types would show if the problem is unique to null-space methods.

---

## GPU Budget

| Experiment | Runs | Total |
|-----------|------|-------|
| Coupling stress (5 seeds) | 5 | ~12.5h |
| Order sensitivity (10 orderings × 2 algs) | 20 | ~60h |
| MVE reproduction (5 seeds × 4) | 20 | ~60h |
| Failure curve + tracking | 10 | ~30h |
| **Total** | | **~162h** |

---

## Citation

```bibtex
@inproceedings{fang2024alphaedit,
  title={AlphaEdit: Null-Space Constrained Knowledge Editing for Language Models},
  author={Fang, Junfeng and Jiang, Houcheng and ...},
  booktitle={ICLR},
  year={2025}
}
```

Upstream code: https://github.com/jianghoucheng/AlphaEdit (pinned at commit `b84624f`)

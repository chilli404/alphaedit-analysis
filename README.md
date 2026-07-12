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

```bash
# Core reproduction
bash scripts/run_mve1_alphaedit_mcf.sh 42
bash scripts/run_rome_baseline.sh 42

# Mechanistic analysis
bash scripts/run_nullspace_analysis.sh 42
bash scripts/run_failure_curve.sh 42

# Novel extensions
bash scripts/run_coupling_stress.sh 42      # ~2.5h per seed
bash scripts/run_order_sensitivity.sh 42    # ~3h per ordering

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

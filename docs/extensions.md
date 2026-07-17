# Novel Extensions — Detailed Design

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

**Design**: 2000 MCF edits × 5 random orderings × {AlphaEdit, MEMIT}

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

### Key Metrics

- **Standard preservation**: efficacy, paraphrase, neighborhood, GLUE
- **Per-batch norms**: ||ΔW||, ||ΔW @ K_prev||, ||base_lhs||, ||K_prev@K_prev^T||
- **Cache statistics**: batches stored, total keys per layer
- **Comparison**: AlphaEdit vs MEMIT vs MEMIT+SeqReg degradation curves

### Expected Outcomes

1. **Gap closes**: Sequential regularization is sufficient → projection unnecessary
2. **Gap remains**: Null-space constraint is critical → validates AlphaEdit's design
3. **Partial closure**: Regularization helps but projection still provides advantage

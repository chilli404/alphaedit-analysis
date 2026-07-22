# Novel Extensions — Detailed Design

## Extension A: MEMIT+SeqReg — Non-Projected Sequential Regularization

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

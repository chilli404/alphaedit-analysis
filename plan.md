# REVIVE Implementation Plan

## 1. Existing Update Path (Summary)

The MEMIT-Seq Poly2 hybrid update path through `polykernel_seqreg_runner.py`:

1. **Runner** builds an inline Python script via `build_polykernel_seqreg_script()`
2. **Script patches** `memit_main.py` via source injection:
   - Replaces `SOLVE_ANCHOR` (the `torch.linalg.solve(...)` call) with kernel-augmented solve
   - Injects log+cache code before `DELTAS_ANCHOR` (`deltas[weight_name] = (`)
3. **The patched `execute_memit`** computes per-layer updates:
   - `adj_k = torch.linalg.solve(lhs, layer_ks)` → produces key solution
   - `upd_matrix = resid @ adj_k.T` → raw update (line 201)
   - `upd_matrix = upd_matrix_match_shape(upd_matrix, weights[weight_name].shape)` → orientation fix
   - `weights[weight_name][...] = weights_copy[weight_name] + upd_matrix.float()` → temporary application (line 212)
   - `deltas[weight_name] = (adj_k.detach().cpu(), resid.detach().cpu())` → stored factors (line 213)
   - After all layers: `weights_copy` is restored (line 226-228)
4. **`apply_memit_to_model`** reconstructs deltas: `key_mat @ val_mat.T` and applies `w[...] += upd_matrix.float()`

**REVIVE injection point**: Between `upd_matrix_match_shape` and the weight application (line 211-212). This ensures:
- The temporary weight update uses the filtered version (cross-layer residual accuracy)
- We also store filtered factors OR store the full filtered matrix

**Approach**: Store `upd_matrix` directly in deltas instead of `(adj_k, resid)` factors, and patch `apply_memit_to_model` to handle this. This is the cleanest approach since we already patch the entire function.

## 2. Files to Create/Modify

### New Files:
- `src/revive/__init__.py` — Package marker
- `src/revive/revive_filter.py` — Core REVIVE SVD filter logic (standalone, testable)
- `src/revive/svd_cache.py` — Persistent SVD caching with validation
- `tests/test_revive.py` — Unit tests (8 tests specified in requirements)
- `tests/test_revive_integration.py` — Integration smoke test

### Modified Files:
- `src/polykernel/polykernel_seqreg_runner.py` — Add REVIVE args, inject filter code
- `scripts/run_polykernel_seqreg.sh` — Add REVIVE env vars
- `analysis/plots/method_comparison.py` — Add polykernel_seqreg discovery path

## 3. Implementation Strategy

### 3A. Core REVIVE Filter (`src/revive/revive_filter.py`)

```python
def compute_protected_rank(S: Tensor, tau: float) -> int:
    """Find smallest k where cumsum(S)[:k] / sum(S) >= tau."""

def revive_filter(delta_w_raw: Tensor, U: Tensor, S: Tensor, Vh: Tensor, tau: float) -> Tensor:
    """
    Filter update to remove components in protected spectral subspace.

    For rectangular W0 (m×n) with compact SVD:
      U: [m, r], S: [r], Vh: [r, n], r = min(m,n)

    Implementation: P_u_tail @ delta_w_raw @ P_v_tail
    where P_u_tail = U_tail @ U_tail.T, P_v_tail = V_tail @ V_tail.T

    Memory-efficient: never materializes full m×m or n×n matrices.
    Uses: (U_tail.T @ delta_w_raw @ V_tail) then reconstructs.
    """
```

**Rectangular matrix handling**: For Llama-3 `down_proj` (4096×14336), compact SVD gives `U:[4096,4096]`, `S:[4096]`, `Vh:[4096,14336]`. The compact SVD spans the full row space but NOT the full column space. Components of `delta_w_raw` in the null space of `U.T` (orthogonal complement of left singular vectors) are NOT representable in the SVD basis and would be zeroed by a naive `U @ A_safe @ Vh` reconstruction.

**Correct approach**: Use the projection formulation:
- `delta_w_safe = U_tail @ (U_tail.T @ delta_w_raw @ V_tail) @ V_tail.T`

This correctly:
1. Projects out the top-k left singular directions
2. Projects out the top-k right singular directions
3. Preserves components in the tail subspace
4. For square matrices: equivalent to zeroing A[:k,:] and A[:,:k]
5. For rectangular (m<n): U spans all of R^m, so P_u_tail properly removes top-k left directions

For Llama-3 down_proj: m=4096, n=14336, r=min(m,n)=4096. U is [4096,4096] (square, full), Vh is [4096,14336] (truncated). V_tail = Vh[k:,:].T has shape [14336, 4096-k]. The right-side projection P_v_tail does NOT project onto the full n-dimensional space but only onto the subspace spanned by the tail right singular vectors. Components in the orthogonal complement of all right singular vectors (the null space of W0.T) pass through neither projection — they're implicitly removed.

**This matches the paper's intent**: protect the dominant spectral structure of W0 by removing update components aligned with its principal directions.

### 3B. SVD Cache (`src/revive/svd_cache.py`)

- Cache layout: `{cache_dir}/{model_slug}/{param_slug}/svd.pt`
- Stores: U, S, Vh, metadata (shape, model_id, fingerprint, dtype, version)
- Fingerprint: SHA256 of first 1000 elements of weight (fast, stable)
- Validation on load: shape, model_id, fingerprint, finite values, descending S

### 3C. Source Injection in polykernel_seqreg_runner.py

Add a new injection between `upd_matrix_match_shape` and the weight assignment:

**New anchor** (line 205-211 of vendor memit_main.py):
```python
WEIGHT_UPDATE_ANCHOR = '        # Update model weights and record desired changes in `delta` variable\n        with torch.no_grad():\n            weights[weight_name][...] = weights_copy[weight_name] + upd_matrix.float()'
```

**Injected code** (when REVIVE enabled):
```python
        # === REVIVE: filter update through pretrained spectral subspace ===
        if _revive_enabled:
            upd_matrix = _revive_apply(upd_matrix, weight_name, layer)
        # === END REVIVE ===
```

**Also patch `apply_memit_to_model`**: Change deltas storage from `(adj_k, resid)` factors to the full `upd_matrix` tensor, and change the apply loop to use it directly.

### 3D. Original Weight Capture

Before the editing loop in evaluate.py, capture original weights for all edited layers. Inject code that:
1. Reads `hparams.layers` to determine which layers are edited
2. For each layer, resolves the weight name and clones the original parameter
3. Computes (or loads from cache) the SVD
4. Stores the SVD factors in a dict keyed by weight_name

This goes into the `_ckpt_load_injection` site (before the editing loop).

### 3E. CLI Arguments

Add to argparse:
- `--revive` (store_true)
- `--revive_tau` (float, default=0.2)
- `--revive_svd_device` (str, default="cpu")
- `--revive_svd_dtype` (str, default="float32")
- `--revive_cache_dir` (str, default=None → auto-resolve)
- `--revive_log_interval` (int, default=1)
- `--revive_mode` (str, default="hard", choices=["hard"])

### 3F. Variant Naming

When REVIVE enabled:
```
MEMIT-Seq-poly2-hybrid-REVIVE-tau{tau}-lp{lp}-ld{ld}-cache{cache}
```

Checkpoint dir also encodes this so incompatible checkpoints can't be resumed.

### 3G. Logging

One JSONL record per layer per batch when REVIVE is active. Logged in `_memit_log` alongside existing SeqReg entries with a `"phase": "revive"` marker.

### 3H. Method Comparison Loader Fix

The loader at `analysis/plots/method_comparison.py`:
1. Already scans `results/failure_curve_checkpointed/` which is where polykernel_seqreg writes
2. The `_discover_polykernel_seq_variants` regex needs updating to also match REVIVE variants
3. Add a new discovery path for `results/polykernel_seqreg/` as a fallback
4. Add REVIVE color/marker to the plot config

## 4. Orientation Audit

From the vendor code:
- `adj_k` shape: `[d_model, n_edits]` (layer_ks shape, from solve)
- `resid` shape: `[d_out, n_edits]`
- `upd_matrix = resid @ adj_k.T` → `[d_out, d_model]`
- For Llama-3: `down_proj.weight` is `[4096, 14336]` (output × input)
- `upd_matrix_match_shape` handles possible transposition

REVIVE filter receives `upd_matrix` AFTER `upd_matrix_match_shape`, so it's guaranteed to match the weight shape. Assertion: `assert upd_matrix.shape == weights[weight_name].shape`

## 5. Experiment Commands

### Stage A: Validation (1 batch, verbose)
```bash
REVIVE=true REVIVE_TAU=0.2 FAST_CHECKPOINT=true TARGET_EDITS=100 \
NO_KERNEL_PREV=true bash scripts/run_polykernel_seqreg.sh 42 1.0 0.0
```

### Stage B: 100-edit smoke
```bash
REVIVE=true REVIVE_TAU=0.2 NO_KERNEL_PREV=true TARGET_EDITS=100 \
EVAL_AT_CHECKPOINTS_ONLY=true bash scripts/run_polykernel_seqreg.sh 42 1.0 0.0
```

### Stage C: 1K calibration sweep
```bash
for TAU in 0.05 0.10 0.20 0.30 0.40; do
  REVIVE=true REVIVE_TAU=$TAU NO_KERNEL_PREV=true TARGET_EDITS=1000 \
  EVAL_AT_CHECKPOINTS_ONLY=true bash scripts/run_polykernel_seqreg.sh 42 1.0 0.0
done
```

### Stage D: 5K comparison
```bash
REVIVE=true REVIVE_TAU=<best> NO_KERNEL_PREV=true TARGET_EDITS=5000 \
EVAL_AT_CHECKPOINTS_ONLY=true bash scripts/run_polykernel_seqreg.sh 42 1.0 0.0
```

### Stage E: 10K final
```bash
REVIVE=true REVIVE_TAU=<best> NO_KERNEL_PREV=true TARGET_EDITS=10000 \
EVAL_AT_CHECKPOINTS_ONLY=true bash scripts/run_polykernel_seqreg.sh 42 1.0 0.0
```

## 6. Test Commands

```bash
uv run pytest tests/test_revive.py -v
uv run pytest tests/test_revive_integration.py -v
```

## 7. Implementation Order

1. Create `src/revive/revive_filter.py` — pure math, no dependencies
2. Create `src/revive/svd_cache.py` — caching logic
3. Create `tests/test_revive.py` — verify filter correctness
4. Modify `src/polykernel/polykernel_seqreg_runner.py`:
   a. Add CLI args
   b. Add REVIVE injection code (new anchor + replacement)
   c. Update variant naming
   d. Update checkpoint dir resolution
   e. Pass REVIVE state to exec namespaces
5. Modify `scripts/run_polykernel_seqreg.sh` — env var support
6. Create `tests/test_revive_integration.py` — script compilation test
7. Modify `analysis/plots/method_comparison.py` — discovery fix

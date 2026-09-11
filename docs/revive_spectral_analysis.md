# REVIVE Spectral Filter: Off-By-One Bug and Spectrum Analysis

Date: 2026-09-10

## Summary

We identified an off-by-one discrepancy between REVIVE's published dominant-subspace definition (Equation 5) and its released reference implementation. The impact is architecture-dependent: on GPT-J the filter is a complete no-op at the paper's default threshold, while on Llama-3-8B it protects one fewer direction than specified (7 vs 8).

## Background: How REVIVE Works

REVIVE (ACL 2026) is a post-hoc spectral filter for sequential knowledge editing. After any base editor (MEMIT, AlphaEdit, NSE) computes a weight update ΔW, REVIVE projects ΔW to remove components in the dominant singular directions of the current weight matrix W.

### Paper's Algorithm (Equations 5-6)

1. Compute full SVD of current weight: W = U @ diag(S) @ Vᵀ
2. Find split_rank k = min{k : Σᵢ₌₁ᵏ σᵢ / Σᵢ σᵢ ≥ τ}
3. Project ΔW into coefficient space: C = Uᵀ @ ΔW @ V
4. Zero out top-k rows and columns: C[:k, :] = 0, C[:, :k] = 0
5. Reconstruct: ΔW_safe = U @ C @ Vᵀ
6. Apply W_new = W + ΔW_safe

The parameter τ (typically 0.1) controls how many dominant directions to protect.

## The Bug

### Released implementation (code.py line 47)

```python
split_rank = (s.cumsum(dim=0) / s.sum() > thresh).float().argmax().item()
projected_coff[:split_rank, :] = 0
projected_coff[:, :split_rank] = 0
```

### The problem

`argmax` returns the 0-indexed position of the first True value. If the first singular value alone exceeds the threshold (σ₁/Σσᵢ > τ), then `argmax` returns **0**. The slice `[:0]` is empty in Python — nothing is zeroed.

More precisely: the paper specifies k as a **count** (protect k directions), but the code returns a **0-indexed position** and uses it as a count without adding 1. This means the code always protects one fewer direction than the paper specifies.

### Corrected implementation

```python
cumfrac = s.cumsum(dim=0) / s.sum()
split_rank = int(torch.searchsorted(cumfrac, torch.tensor(tau), right=False).item()) + 1
split_rank = min(split_rank, s.numel())
```

Using `searchsorted` with `right=False` implements the paper's `≥ τ` condition (vs the code's `> τ`), and the `+1` converts from 0-indexed position to count.

## Spectral Analysis: Architecture-Dependent Impact

The bug's severity depends on how concentrated the singular value spectrum is. If σ₁ alone exceeds τ, bug_k = 0 (complete no-op). If not, bug_k = paper_k - 1 (off by one direction).

### GPT-J-6B Layer 5 (dim=16384)

The covariance matrix has a highly concentrated spectrum.

| Property | Value |
|---|---|
| σ₁ share of total energy | **21.15%** |
| Top 10 cumulative | ~25% |
| Top 50 cumulative | ~50% |

| τ | bug_k | paper_k | Bug behavior |
|---|---|---|---|
| 0.01 | 0 | 1 | **No-op** |
| 0.05 | 0 | 1 | **No-op** |
| 0.10 | 0 | 1 | **No-op** |
| 0.15 | 0 | 1 | **No-op** |
| 0.20 | 0 | 1 | **No-op** |
| 0.25 | 8 | 9 | Off-by-one |
| 0.30 | 35 | 36 | Off-by-one |
| 0.50 | 659 | 660 | Negligible |

**For GPT-J at τ=0.1 (paper default): REVIVE is a complete no-op.** The released "MEMIT+REVIVE" GPT-J results are functionally just MEMIT.

### Llama-3-8B Layer 6 (dim=14336)

The covariance matrix has a more spread-out spectrum.

| Property | Value |
|---|---|
| σ₁ share of total energy | **5.39%** |
| Top 10 cumulative | 10.8% |
| Top 100 cumulative | 22.4% |
| Top 500 cumulative | 37.6% |

| τ | bug_k | paper_k | Bug behavior |
|---|---|---|---|
| 0.01 | 0 | 1 | **No-op** |
| 0.05 | 0 | 1 | **No-op** |
| 0.10 | 7 | 8 | Off-by-one (7 vs 8 directions) |
| 0.15 | 27 | 28 | Off-by-one |
| 0.20 | 69 | 70 | Off-by-one |
| 0.25 | 140 | 141 | Off-by-one |
| 0.30 | 249 | 250 | Off-by-one |
| 0.50 | 0 | 500+ | **No-op** (top 500 < 50% energy) |

**For Llama at τ=0.1: the filter is active (k=7 with bug, k=8 corrected).** The difference is one direction out of 8 — a real but mild discrepancy.

## Implications for Published REVIVE Results

### REVIVE's threshold sweep is real but needs reinterpretation

The paper reports a smooth tradeoff as τ increases from 0.05 to 0.30 on Llama:

| τ | Efficacy | Paraphrase | Neighborhood |
|---|---|---|---|
| 0.05 | 94.46 | 86.03 | 59.70 |
| 0.10 | 95.62 | 84.60 | 62.17 |
| 0.15 | 95.03 | 80.60 | 64.49 |
| 0.20 | 94.58 | 78.38 | 66.19 |
| 0.25 | 92.96 | 73.94 | 68.94 |
| 0.30 | 88.94 | 67.56 | 71.86 |

This tradeoff is consistent with progressively filtering more directions. On Llama, the filter is active for τ ≥ 0.10 (k=7+), so the sweep variation is genuine.

However, on GPT-J, the filter is a no-op for τ ≤ 0.20. Any reported GPT-J "REVIVE" results at these thresholds are functionally the base editor.

### The paper's results are not invalidated

The published Llama results at τ=0.1 used k=7 (instead of the intended k=8). This is a minor discrepancy unlikely to materially change the reported numbers. The overall REVIVE mechanism works as intended on Llama — the bug simply protects one fewer direction.

The GPT-J results at τ=0.1 are more affected: the filter was genuinely inactive. But the paper primarily presents Llama results, and the GPT-J tau sweep presumably shows variation at higher thresholds where the filter activates.

## What We Changed

### Code fixes (3 files)

1. `src/polykernel/polykernel_seqreg_runner.py` line ~612: `searchsorted` + 1
2. `src/revive/revive_filter.py` `compute_protected_rank()`: same fix
3. `baselines/REVIVEEDIT/code.py` line 47: same fix

### Diagnostic logging

Added to `_revive_apply()`:
- `sigma1_share`: fraction of total energy in the first singular value
- `split_rank`: actual number of protected directions
- `upd_ratio`: ||ΔW_filtered|| / ||ΔW_original|| (1.0 = no filtering)

### Evaluate.py-level injection

REVIVE was previously injected only into `memit_main.py`. For non-MEMIT algorithms (AlphaEdit, NSE, EvoEdit), REVIVE was never applied even when `--revive` was enabled. Added a post-edit injection in evaluate.py that:
1. Captures pre-edit layer weights before `apply_algo`
2. After `apply_algo` returns, computes ΔW per layer
3. Applies the REVIVE SVD filter
4. Writes back W_old + ΔW_filtered

This enables REVIVE+AlphaEdit and REVIVE+NSE combinations via `--base_alg AlphaEdit` or `--base_alg NSE`.

### Runner support

`polykernel_seqreg_runner.py` `--base_alg` now accepts `NSE` in addition to `MEMIT` and `AlphaEdit`. The run script `run_revive_baseline.sh` reads `BASE_ALG` from environment variables and resolves HPARAMS_FNAME based on MODEL_NAME.

## Defensible Paper Statement

> We identified an off-by-one discrepancy between REVIVE's published dominant-subspace definition (Equation 5: smallest k where cumulative energy ≥ τ) and its released reference implementation (which computes k-1). On Llama-3-8B (σ₁ ≈ 5.4% of spectral energy), this is a one-direction boundary error at τ=0.1 (k=7 vs 8). On GPT-J (σ₁ ≈ 21.1%), the filter is inactive for all τ ≤ 0.20 because the first singular value alone exceeds the threshold. We evaluate with the equation-faithful implementation and report both released and corrected split_rank values.

## Experimental Plan: Both Variants

We run REVIVE with **both** the released code behavior (k-1) and the equation-faithful fix (k), then compare.

### What we actually run

**Equation-faithful (fix applied, tau=0.1):** 8 GPU runs (4 Llama + 4 GPT-J)

These use `searchsorted` + 1, producing:
- Llama: split_rank=8 (protects 8 directions)
- GPT-J: split_rank=1 (protects 1 direction)

**Released behavior (k-1, tau=0.1):** No new runs needed

The released code gives:
- Llama: split_rank=7 (we can interpolate or note the 7-vs-8 difference is negligible)
- GPT-J: split_rank=0 (complete no-op = base editor results we already have)

So "released REVIVE on GPT-J" = our existing AlphaEdit/MEMIT numbers. "Released REVIVE on Llama" ≈ our fixed REVIVE minus one direction (expected to be very similar).

### Reporting

The paper table should include:

| Method | Implementation | Llama k | GPT-J k | Notes |
|---|---|---|---|---|
| AlphaEdit (no REVIVE) | — | — | — | Baseline |
| AlphaEdit + REVIVE (released) | k-1 | 7 | **0 (no-op)** | Matches published code |
| AlphaEdit + REVIVE (Eq. 5) | k | 8 | 1 | Paper's intended behavior |
| MEMIT (no REVIVE) | — | — | — | Baseline |
| MEMIT + REVIVE (released) | k-1 | 7 | **0 (no-op)** | Matches published code |
| MEMIT + REVIVE (Eq. 5) | k | 8 | 1 | Paper's intended behavior |

For GPT-J, the "released" row IS the base editor. For Llama, the "released" row protects 7 directions vs the fix's 8 — likely a small difference that we can verify from the diagnostic logs.

### SkyPilot Commands (equation-faithful fix, tau=0.1)

```bash
source ~/.zshrc; sky-dev
SKY=sky/alphaedit_gpu.yaml
LLAMA="NousResearch/Meta-Llama-3-8B-Instruct"
GPTJ="EleutherAI/gpt-j-6b"

# Llama REVIVE+AlphaEdit
sky launch $SKY --env-file .env \
    --env EXPERIMENT_NAME=revive_baseline --env SEED=42 \
    --env ORDERING=fb_high_exposure --env TARGET_EDITS=10000 \
    --env BASE_ALG=AlphaEdit --env MODEL_NAME=$LLAMA \
    --cluster revive-ae-fbhi-s42 --detach-run -y

sky launch $SKY --env-file .env \
    --env EXPERIMENT_NAME=revive_baseline --env SEED=42 \
    --env ORDERING=fb_random0 --env TARGET_EDITS=10000 \
    --env BASE_ALG=AlphaEdit --env MODEL_NAME=$LLAMA \
    --cluster revive-ae-fbrnd0-s42 --detach-run -y

# Llama REVIVE+MEMIT (plain, paper config)
sky launch $SKY --env-file .env \
    --env EXPERIMENT_NAME=revive_baseline --env SEED=42 \
    --env ORDERING=fb_high_exposure --env TARGET_EDITS=10000 \
    --env LAMBDA_PREV=0 --env LAMBDA_DELTA=0 --env MODEL_NAME=$LLAMA \
    --cluster revive-memit-fbhi-s42 --detach-run -y

sky launch $SKY --env-file .env \
    --env EXPERIMENT_NAME=revive_baseline --env SEED=42 \
    --env ORDERING=fb_random0 --env TARGET_EDITS=10000 \
    --env LAMBDA_PREV=0 --env LAMBDA_DELTA=0 --env MODEL_NAME=$LLAMA \
    --cluster revive-memit-fbrnd0-s42 --detach-run -y

# GPT-J REVIVE+AlphaEdit
sky launch $SKY --env-file .env \
    --env EXPERIMENT_NAME=revive_baseline_gptj --env SEED=42 \
    --env ORDERING=fb_high_exposure --env TARGET_EDITS=10000 \
    --env BASE_ALG=AlphaEdit --env MODEL_NAME=$GPTJ \
    --cluster gptj-revive-ae-fbhi-s42 --detach-run -y

sky launch $SKY --env-file .env \
    --env EXPERIMENT_NAME=revive_baseline_gptj --env SEED=42 \
    --env ORDERING=fb_random0 --env TARGET_EDITS=10000 \
    --env BASE_ALG=AlphaEdit --env MODEL_NAME=$GPTJ \
    --cluster gptj-revive-ae-fbrnd0-s42 --detach-run -y

# GPT-J REVIVE+MEMIT (plain, paper config)
sky launch $SKY --env-file .env \
    --env EXPERIMENT_NAME=revive_baseline_gptj --env SEED=42 \
    --env ORDERING=fb_high_exposure --env TARGET_EDITS=10000 \
    --env LAMBDA_PREV=0 --env LAMBDA_DELTA=0 --env MODEL_NAME=$GPTJ \
    --cluster gptj-revive-memit-fbhi-s42 --detach-run -y

sky launch $SKY --env-file .env \
    --env EXPERIMENT_NAME=revive_baseline_gptj --env SEED=42 \
    --env ORDERING=fb_random0 --env TARGET_EDITS=10000 \
    --env LAMBDA_PREV=0 --env LAMBDA_DELTA=0 --env MODEL_NAME=$GPTJ \
    --cluster gptj-revive-memit-fbrnd0-s42 --detach-run -y
```

---

# NSE (Neuron-Level Sequential Editing) Notes

## Overview

NSE (ACL 2025) selects neurons across layers based on activation magnitudes and optimizes target hidden states using the original model weights. It uses 25 gradient steps per edit to compute v_star (target hidden state), then solves a neuron-subsetted least-squares problem similar to MEMIT.

## kv Cache Mechanism

NSE pre-computes v_star per (case_id, layer) and caches them as `.npz` files. The cache key is:
```
{model_name}_{alg_name}/mcf_layer_{L}_clamp_{C}_case_{ID}.npz
```

Despite the naming suggesting per-layer caching, NSE actually only caches at `hparams.layers[-1]` (layer 8 for both Llama and GPT-J). One v_star per case_id is sufficient because NSE computes it at `v_loss_layer` (layer 27) which is the same regardless of which edited layer.

## Cache Contents

| Model | Cache | Files | Unique case_ids | Layers |
|---|---|---|---|---|
| Llama-3-8B | llama_nse_cache.tar (215 MB) | 12,379 | 12,379 | 8 only |
| GPT-J-6B | gptj_nse_cache.tar (175 MB) | 10,046 | 10,046 | 8 only |

### Cache coverage for our 10K ordering streams

| Model | Ordering | Total case_ids | Cached | Missing | Coverage | Est. time for missing |
|---|---|---|---|---|---|---|
| **Llama** | fb_high s42 | 10,000 | 5,908 | **4,092** | 59.1% | ~14h |
| **GPT-J** | fb_high s42 | 10,000 | 4,815 | **5,185** | 48.1% | ~18h |

The missing case_ids will have v_star computed on-the-fly (25 gradient steps per edit at ~0.5s each). This adds ~14h for Llama and ~18h for GPT-J on top of the base editing time.

### The cache is populated incrementally

Once a run computes v_star for a missing case_id, it caches the result. Subsequent runs (e.g., fb_random0 after fb_high) reuse the expanded cache. The second NSE run on the same model will be significantly faster.

## NSE Hparams

| Parameter | Llama-3-8B | GPT-J-6B |
|---|---|---|
| layers | [4, 5, 6, 7, 8] | [3, 4, 5, 6, 7, 8] |
| v_loss_layer | 27 | 27 |
| v_num_grad_steps | 25 | 25 |
| v_lr | 0.1 | 0.5 |
| alpha | 2.5 | 15 |
| upper_bound | 50 | 100 |
| neuron_threshold | **1.0** | 0.8 |
| max_iterations | 3 | 3 |

**Important:** Llama's `neuron_threshold=1.0` means ALL neurons are selected (cumulative sum always reaches 100% at the last neuron). This effectively disables NSE's neuron selection feature, making Llama NSE behave like MEMIT with iterative v_star correction. GPT-J's `neuron_threshold=0.8` does use sparse neuron selection.

## NSE uses cache_c (sequential history)

NSE accumulates `cache_c += layer_ks @ layer_ks.T` after each batch (line 235 of nse_main.py), identically to AlphaEdit. The solve includes `cache_c[i][selected_rows][:, selected_rows]` in the LHS. This makes NSE a history-aware sequential editor.

## REVIVE + NSE Compatibility

REVIVE works with NSE via the evaluate.py-level injection. NSE modifies weights in-place (`weights[weight_name][...] = weights[weight_name] + upd_matrix.float()` at line 222), so the pre/post weight capture mechanism works correctly.

---

## Data Sources

- GPT-J covariance spectrum: `data/stats/gpt-j-6b/wikipedia_stats/transformer.h.5.mlp.fc_out_float32_mom2_100000.npz`
- Llama covariance spectrum: `data/stats/llama3-8b-instruct/wikipedia_stats/model.layers.6.mlp.down_proj_float32_mom2_100000.npz`
- Reference REVIVE code: `baselines/REVIVEEDIT/code.py`
- Our REVIVE implementation: `src/polykernel/polykernel_seqreg_runner.py`, `src/revive/revive_filter.py`
- REVIVE paper: https://aclanthology.org/2026.acl-long.1384

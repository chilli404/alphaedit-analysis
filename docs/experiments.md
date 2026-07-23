# Experiment Reference

This document describes every experiment in the AlphaEdit reproducibility study: what scientific question it answers, how it works mechanistically, what data it uses, how measurements are computed, and where results and checkpoints are stored.

---

## Background: How AlphaEdit Works

AlphaEdit edits LLM weights by solving a constrained least-squares problem. Given a batch of edit requests (prompt → new_target), it:

1. **Extracts keys**: Runs each prompt through the model and captures the input to `model.layers.{L}.mlp.down_proj` at the subject's last token position. These are the "key vectors" K ∈ ℝ^{d_in × n_batch}.

2. **Computes residuals**: For each layer, computes how much the output must change to produce the new target: R ∈ ℝ^{d_out × n_batch}.

3. **Solves for the weight update** (constrained to null-space of prior knowledge):
   ```
   ΔW = solve(P @ (K@K^T + cache_c) + λI, P @ K @ R^T)
   ```
   Where:
   - **P** is a static null-space projection matrix (computed once from Wikipedia covariance statistics)
   - **cache_c** is an accumulated covariance of all previously-used keys (grows with each batch)
   - **λ** is L2 regularization

4. **Updates the cache** after each batch: `cache_c += K @ K^T`

The null-space projection P is computed by SVD of the Wikipedia covariance matrix C:
```python
U, S, _ = torch.linalg.svd(C)
null_indices = (S < threshold).nonzero()
P = U[:, null_indices] @ U[:, null_indices].T
```

**Critical finding**: On Llama-3-8B with threshold=2e-2, P retains 99.7–99.9% of dimensions (only 10–43 directions excluded per layer). The null-space constraint is effectively vacuous — P ≈ I.

---

## Background: MEMIT (Baseline)

MEMIT (Mass-Editing Memory in a Transformer) is the unconstrained predecessor:
```
ΔW = solve(α·C₀ + K@K^T, K @ R^T)
```
No projection P, no cache_c. Each batch is solved independently with no memory of prior edits.

---

## Background: Evaluation Metrics

All experiments measuring editing quality use the same metrics from `eval_utils_counterfact.py`:

| Metric | What It Measures | How |
|--------|-----------------|-----|
| **Efficacy** | Was the edit installed? | P(target_new) > P(target_true) on the rewrite prompt |
| **Generalization** | Does it transfer to paraphrases? | Same test on paraphrase prompts |
| **Specificity** | Are unrelated facts preserved? | P(target_true) > P(target_new) on neighborhood prompts |
| **Fluency** | Is output coherent? | Weighted n-gram entropy on generated text |
| **GLUE** | General language understanding | SST-2, MRPC, RTE accuracy |

Probability comparison works by tokenizing prompt + target_new and prompt + target_true, computing average per-token NLL for each, and checking which target is more probable.

---

## Background: Datasets

### MultiCounterFact (MCF)

Source: `memit.baulab.info/data/dsets/multi_counterfact.json` (~21K records)

Each record contains:
```json
{
  "case_id": 12345,
  "requested_rewrite": {
    "prompt": "The mother tongue of {} is",
    "subject": "Danielle Darrieux",
    "target_new": {"str": "English"},
    "target_true": {"str": "French"},
    "relation_id": "P103"
  },
  "paraphrase_prompts": ["The native language of Danielle Darrieux is", ...],
  "neighborhood_prompts": [
    {"prompt": "The mother tongue of Catherine Deneuve is", "target": "French"},
    ...
  ],
  "generation_prompts": ["Danielle Darrieux's mother tongue is"]
}
```

### zsRE (Zero-Shot Relation Extraction)

Source: `memit.baulab.info/data/dsets/zsre_mend_eval.json`

QA-format factual knowledge. Similar structure to MCF but derived from a different knowledge base.

---

## Background: Source Injection Pattern

All Python runners avoid import path issues with the vendor code by:
1. Reading `vendor/AlphaEdit/experiments/evaluate.py` as text
2. Injecting measurement/tracking code at specific anchor points (pinned at vendor commit `b84624f`)
3. Using `compile()` + `exec()` to run the modified source

**Dual source injection** (used by some runners): patches both an algorithm file (`memit_main.py` or `AlphaEdit_main.py`) AND `evaluate.py`.

---

## Experiment 1: MVE1 — AlphaEdit on MultiCounterFact

### Scientific Question
Can we reproduce AlphaEdit's headline result: 2000 sequential edits with high efficacy and preserved specificity?

### Data
MultiCounterFact, first 2000 records in canonical ordering (order_id=0, no shuffle), processed in batches of 100.

### Measurement
After each batch of 100 edits is applied, ALL 2000 records are re-evaluated (efficacy, generalization, specificity). GLUE evaluation runs every 20 batches (with 20 total batches, this means GLUE runs at the start and end only).

### How It Works
- **Script**: `scripts/run_mve1_alphaedit_mcf.sh [SEED]`
- **Runner**: `src/runners/seeded_runner.py`
- Seeds all RNGs, patches CUDA device, injects deterministic shuffle into `evaluate.py`, overrides the results directory
- Source injection: single file (evaluate.py)

### Results Path
```
$RESULT_ROOT/mve1_alphaedit_mcf/seed{SEED}/2000edits/AlphaEdit/run_000/
    100_edits-case_0.json
    100_edits-case_1.json
    ...
    glue_eval/
```

### Checkpoints
None (single run, ~2–3 hours on L40s).

---

## Experiment 2: MVE2 — MEMIT on MultiCounterFact

### Scientific Question
How does MEMIT (no null-space projection, no cache) compare to AlphaEdit on identical data?

### Data
Same 2000 MCF records, same batches, same canonical ordering as MVE1.

### Measurement
Identical to MVE1. Direct comparison: does the null-space projection actually help?

### How It Works
- **Script**: `scripts/run_mve2_memit_mcf.sh [SEED]`
- **Runner**: `src/runners/seeded_runner.py` with `--alg_name MEMIT`

### Results Path
```
$RESULT_ROOT/mve2_memit_mcf/seed{SEED}/2000edits/MEMIT/run_000/
```

### Checkpoints
None.

---

## Experiment 3: MVE3 — AlphaEdit on zsRE

### Scientific Question
Does AlphaEdit's performance generalize beyond MultiCounterFact to a different factual dataset?

### Data
zsRE dataset, first 2000 records (same batch structure as MVE1).

### Measurement
Identical metric computation to MVE1, applied to zsRE-format records.

### How It Works
- **Script**: `scripts/run_mve3_alphaedit_zsre.sh [SEED]`
- **Runner**: `src/runners/seeded_runner.py` with `--ds_name zsre`

### Results Path
```
$RESULT_ROOT/mve3_alphaedit_zsre/seed{SEED}/2000edits/AlphaEdit/run_000/
```

### Checkpoints
None.

---

## Experiment 4: Failure Curve (Checkpointed)

### Scientific Question
At what edit count does AlphaEdit's null-space advantage disappear? Where does MEMIT catch up or surpass it?

### Data
MultiCounterFact, up to 10,000 records processed in batches of 100.

### Measurement
Three evaluation modes:
- **Normal**: Evaluate ALL records after every batch (complete, slow)
- **Fast** (`FAST_CHECKPOINT=true`): Evaluate only the current batch's records (partial preservation data)
- **Milestone** (`EVAL_AT_CHECKPOINTS_ONLY=true`): Full evaluation only at checkpoint boundaries (balanced, RECOMMENDED)

### How It Works
- **Script**: `scripts/run_failure_curve_checkpointed.sh [SEED] [ALG] [TARGET_EDITS]`
- **Runner**: `src/runners/checkpoint_runner.py`
- Injects checkpoint save/load/skip logic into `evaluate.py`
- Saves only edited layer weights (not full model) + cache_c (AlphaEdit only)
- Automatically resumes from the latest checkpoint on re-run
- Supports AlphaEdit, MEMIT, and MEMIT-Seq variants

### Checkpoint Implementation
Before the edit loop: loads weights + cache_c from the latest checkpoint, sets batch counter.
Before each batch: skips already-processed batches.
After each batch (at save_interval boundaries): saves model weights + cache_c + metadata.

### Results Path
```
$RESULT_ROOT/failure_curve_checkpointed/seed{SEED}/{TARGET}edits/{ALG}/run_000/
    100_edits-case_0.json
    ...
```

### Checkpoints
```
$CHECKPOINT_ROOT/failure_curve/{ALG}/seed{SEED}/batch_{N}/
    metadata.json         # {batch_idx, total_edits, alg_name, seed, timestamp}
    model_weights.pt      # Edited layer weights only (~560MB)
    cache_c.pt            # Accumulated covariance cache (AlphaEdit only, ~320MB)
```

---

## Experiment 5: MEMIT with Sequential Regularization (MEMIT-Seq)

### Scientific Question
Does MEMIT with AlphaEdit-like sequential regularization (but WITHOUT null-space projection P) close the performance gap? Is P actually necessary, or is the cache regularization sufficient?

### Mathematical Formulation

AlphaEdit (projected, Eq. 12 from the paper):
```
minimize ||ΔP·K - R||² + λ_prev·||ΔP·K_prev||² + λ_delta·||ΔP||²
```

MEMIT-Seq (non-projected analogue):
```
minimize ||Δ·K - R||² + λ_prev·||Δ·K_prev||² + λ_delta·||Δ||²
```

Implemented via LHS augmentation in the normal equation:
```
lhs = α·C₀ + K_new@K_new^T + λ_prev·K_prev@K_prev^T + λ_delta·I
```
Where α = `hparams.mom2_update_weight` (MEMIT's covariance weighting coefficient) and C₀ is the precomputed Wikipedia covariance matrix for that layer.

### Data
Same MultiCounterFact stream as the failure curve.

### Key Implementation Details
- `K_prev` is computed from the key cache BEFORE appending current batch keys
- Logging (‖ΔW‖, ‖ΔW@K_prev‖, etc.) happens BEFORE cache update
- Current keys are appended to cache AFTER logging
- Variant naming: `MEMIT-Seq-lp{λ_prev}-ld{λ_delta}-cache{max}` (cache0 = unlimited)
- **Cache strategies**: `recent` (default) keeps only the last N batches of keys; `all` keeps all keys (subject to cache_max if set). Keys are added to cache when `lambda_prev > 0` or `cache_strategy == "all"`

### How It Works
- **Script**: `scripts/run_failure_curve_checkpointed.sh [SEED] MEMIT-Seq-lp1.0-ld0.0-cache0 [TARGET]`
- **Runner**: `src/runners/memit_sequential_runner.py`
- Dual source injection: patches `memit_main.py` (LHS augmentation) AND `evaluate.py` (batch tracking)

### Calibration Settings

Note: Code defaults are λ_prev=0.0, λ_delta=0.0 (plain MEMIT). The table below shows recommended experimental settings:

| λ_prev | λ_delta | Interpretation |
|--------|---------|----------------|
| 1 | 0 | Full-history prior-key protection, no ridge |
| 1 | 1 | Direct Eq. 12 coefficient analogue |
| 1 | 1e-4 | Weak ridge regularization |
| 10 | 1 | Strong prev-key protection |
| 100 | 1 | Very strong prev-key protection |

### Results Path
```
$RESULT_ROOT/failure_curve_checkpointed/seed{SEED}/{TARGET}edits/MEMIT-Seq-lp{X}-ld{Y}-cache{Z}/run_000/
```

### Checkpoints
```
$CHECKPOINT_ROOT/failure_curve/MEMIT-Seq-lp{X}-ld{Y}-cache{Z}/seed{SEED}/batch_{N}/
```

---

## Experiment 6: Comparison Ordered

### Scientific Question
Are results sensitive to the random ordering of edit facts? If we run 10 different random shuffles, how much variance do we observe?

### Data
MultiCounterFact, 3000 edits (configurable), with 10 different random orderings (order0–order9) generated from the same seed.

### Measurement
Standard efficacy/generalization/specificity metrics at checkpoint boundaries. Compares AlphaEdit, MEMIT, and MEMIT_seq across all 10 orderings.

### How It Works
- **Script**: `scripts/run_comparison_ordered.sh SEED`
- **Env vars**: `ALG_NAME`, `ORDER_ID`, `TARGET_EDITS`
- For AlphaEdit/MEMIT: uses `src/runners/checkpoint_runner.py` with `--order_id`
- For MEMIT_seq: uses `src/runners/memit_sequential_runner.py` with `--order_id`

### Results Path
```
$RESULT_ROOT/comparison_ordered/seed{SEED}/{TARGET}edits/order{0-9}/{ALG}/run_000/
    100_edits-case_*.json
```

### Checkpoints
```
$CHECKPOINT_ROOT/comparison_ordered/{ALG}/seed{SEED}/order{ORDER_ID}/batch_{N}/
```

---

## Experiment 7: Matched Ordering

### Scientific Question
Does the KEY GEOMETRY of edit orderings matter? When keys from the same cluster are grouped together (clustered), does that cause faster degradation than when they are dispersed?

### Data Construction

The ordering streams are pre-generated by `src/datasets/generate_orderings.py`:

1. **Extract keys**: For each of the first 5000 MCF records, extract the key vector at layer 6 using `src/mechanism/compute_keys.py` (input to `model.layers.6.mlp.down_proj` at subject's last token)

2. **Cluster keys**: Apply spherical k-means (L2-normalize, then k-means) to group keys into clusters

3. **Generate orderings** (4 total):
   - `clustered`: Grouped by semantic relation (same `relation_id` facts together)
   - `dispersed`: Round-robin across relation groups (semantic diversity)
   - `key_clustered`: Same key-geometry cluster records grouped together (consecutive batches share similar key geometry)
   - `key_dispersed`: Round-robin across key clusters (consecutive batches have maximally diverse key geometry)

The same 5000 facts appear in all orderings — only the order differs. The primary experiment uses `key_clustered` vs `key_dispersed` (geometry-based); the semantic orderings (`clustered`/`dispersed`) serve as a control.

### Measurement
- Editing with checkpointing (fast mode by default)
- Post-hoc evaluation: loads each checkpoint, evaluates ALL 5000 records for efficacy/generalization/specificity
- Additionally (AlphaEdit): cache eigenspectrum tracking (effective rank, condition number, top SV share) at each batch
- Additionally (AlphaEdit): **functional projection loss** — measures how much edit signal is lost through projection:
  - `q_t`: ratio of edit signal preserved after P-projection
  - `fit_quality_proj`: how well the projected update achieves the edit target
  - `fit_quality_raw`: how well an unconstrained update would achieve the edit target
  - `removed_fraction`: fraction of RHS-level signal removed by projection

### How It Works
- **Script**: `scripts/run_matched_ordering.sh [SEED] [ALG] [ORDERING]`
- For AlphaEdit: uses `src/runners/alphaedit_stream_runner.py` (dual source injection into AlphaEdit_main.py + evaluate.py)
- For MEMIT-Seq: uses `src/runners/memit_sequential_runner.py` with `--dataset_override`
- **Auto-evaluation**: If all checkpoints already exist, runs `scripts/eval_matched_ordering.py` instead of re-editing

### Results Path
```
$RESULT_ROOT/matched_ordering/orderings/{ORDERING}_seed{SEED}.json     # The stream file
$RESULT_ROOT/matched_ordering/{ALG}/{ORDERING}/seed{SEED}/
    full_eval_seed{SEED}.json       # Aggregate evaluation
    stream_metrics.jsonl            # Per-batch mechanism metrics (AlphaEdit)
```

### Checkpoints
```
$CHECKPOINT_ROOT/matched_ordering/{ALG}/{ORDERING}/seed{SEED}/batch_{N}/
    model_weights.pt
    cache_c.pt (AlphaEdit only)
    metadata.json
```

---

## Experiment 8: Capability Probe

### Scientific Question
Do sequential edits destroy the model's GENERAL capabilities (language modeling, reasoning) even while CounterFact-specific metrics look fine?

### Data
- **Perplexity**: WikiText-103 test split (200 text passages, >100 chars each, loaded from pre-downloaded `wikitext_103_test.json`)
- **MMLU**: 5-shot MMLU on 4 diverse categories (50 questions each): abstract_algebra, world_religions, us_foreign_policy, college_biology
- **Edits**: MultiCounterFact (same as failure curve)

### Measurement

**Perplexity computation** (`src/util/capability_probe.py`):
1. Load 200 text passages from WikiText-103 test set
2. Tokenize each to max 512 tokens, batch size 4
3. For each sample: mask padding, shift logits/labels for next-token prediction, sum cross-entropy loss over valid tokens
4. **Corpus-level perplexity**: `exp(total_NLL / total_tokens)` — all tokens pooled across all samples equally weighted (standard definition)
5. Also computes per-sample perplexities for spread reporting: `median_perplexity`, `std_perplexity`

**MMLU computation**:
1. For each of 4 categories, use validation set examples as 5-shot prompt
2. Present test question in multiple-choice format (A/B/C/D)
3. Check logits at the final position for A/B/C/D tokens
4. Predicted answer = highest logit choice
5. Report per-category and overall accuracy

### How It Works
- **Script**: `scripts/run_capability_probe.sh [SEED] [ALG] [DATASET_SIZE_LIMIT]`
- **Shared utility**: `src/util/capability_probe.py` (constants: `WIKITEXT_N_SAMPLES=200`, `WIKITEXT_MAX_LENGTH=512`, `MMLU_N_SHOTS=5`, `MMLU_N_QUESTIONS=50`)
- **Two modes**:
  - **Offline** (preferred): If failure curve checkpoints exist, loads model once, applies checkpoint weights at each batch boundary, measures, then restores base weights. Uses `src/mechanism/capability_probe_offline.py`. The offline probe imports `compute_perplexity` and `load_wikitext_samples` from the shared utility for consistent measurement.
  - **Online**: Full editing with inline probing. Uses `src/runners/capability_probe_runner.py`. Monkey-patches `GLUEEval.evaluate()` to additionally run the probe at each downstream eval step. The online runner has its own inline `_compute_perplexity` that uses HuggingFace `load_dataset("wikitext", "wikitext-103-raw-v1")` directly instead of the pre-downloaded JSON.

### Results Path
```
$RESULT_ROOT/capability_probe/seed{SEED}/{ALG}/offline_probe_*.jsonl
```
or (flat layout):
```
$RESULT_ROOT/capability_probe/offline_seed{SEED}_{ALG}*.jsonl
```

Each JSONL line:
```json
{
  "edit_count": 1000,
  "mean_perplexity": 8.42,
  "median_perplexity": 7.95,
  "std_perplexity": 3.1,
  "n_samples": 200,
  "n_tokens": 51234,
  "mmlu_accuracy": 0.63,
  "mmlu_correct": 126,
  "mmlu_total": 200,
  "mmlu_per_category": {"abstract_algebra": {"accuracy": 0.56, "correct": 28, "total": 50}, ...},
  "timestamp_utc": 1721700000.0,
  "source": "offline_probe",
  "metadata": {"probe_version": "1.1.0", "model_dtype": "torch.float16", ...}
}
```

### Checkpoints
None (reuses failure curve checkpoints for offline mode).

---

## Experiment 9: Polykernel Editor

### Scientific Question
Can a non-linear kernel transformation of the key space mitigate failure modes? If the failure is a LINEAR key-space capacity bottleneck, then projecting keys into a higher-dimensional feature space (via polynomial or RBF kernel) should delay degradation.

### Mathematical Formulation

Standard solve (linear): `K @ K^T`

Kernel-weighted solve: `K @ (G_kernel * scale) @ K^T`

Where:
- **Polynomial kernel**: `G_poly = (1 + K^T @ K)^p` (degree-2 default)
- **RBF kernel**: `G_rbf = exp(-||k_i - k_j||² / 2σ²)`
- **Scale factor**: `scale = trace(G_lin) / (G_kernel * G_lin).sum()` — equalizes the Frobenius inner product so the kernel-weighted KKT has the same trace as the linear Gram matrix

### Data
MultiCounterFact, configurable size (default 2000 for diagnostic, up to 5000+ for full runs).

### Measurement
Standard editing metrics (efficacy, generalization, specificity). Additionally logs per-batch mechanism metrics:
- `trace_ratio`: tr(G_kernel) / tr(I)
- `G_lin_rank`: Numerical rank of the linear Gram matrix
- `kernel_type`: poly or rbf
- `phase`: early/mid/late editing

### How It Works
- **Script**: `scripts/run_polykernel_editor.sh [SEED] [ALG_NAME] [KERNEL_DEGREE] [DATASET_SIZE_LIMIT]`
- **Runner**: `src/polykernel/polykernel_editor_runner.py`
- Dual source injection into the algorithm file (replaces K@K^T computation) and evaluate.py
- Supports both AlphaEdit and MEMIT with kernels

### Results Path
```
$RESULT_ROOT/polykernel_editor/seed{SEED}/{E}edits/{ALG}-{kernel}/
    run_000/
        100_edits-case_*.json
    log_{ALG}_seed{SEED}_{kernel}_*.jsonl       # Per-batch mechanism metrics
    metadata_{ALG}_seed{SEED}_{kernel}.json     # Experiment metadata
```

### Checkpoints
Optional (via `--edit_only --save_interval`).

---

## Experiment 10: Polykernel Diagnostic (Gram Matrix Analysis)

### Scientific Question
Is the failure mode consistent with a LINEAR key-space capacity bottleneck? If we compare the Gram matrix `K^T @ K` (linear) vs `(1 + K^T @ K)²` (poly-2), does the polynomial kernel provide more "room" (higher effective rank)?

### Data
Keys extracted during actual editing of MultiCounterFact (from Stage 1).

### How It Works (Two Stages)

**Stage 1 (GPU)**: Extract raw edit keys during AlphaEdit/MEMIT editing
- Runner: `src/polykernel/polykernel_key_extractor.py`
- Captures keys at each batch during actual editing
- Saves: `results/polykernel_diagnostic/keys_{ALG}_seed{SEED}.pt`

**Stage 2 (CPU)**: Gram matrix geometry analysis
- Runner: `src/polykernel/polykernel_diagnostic.py`
- Constructs G_linear = K^T @ K and G_poly2 = (1 + K^T @ K)²
- Computes per-window (sliding window of `window_size` batches):
  - `eff_rank`: Effective rank (exponential entropy of normalized eigenvalues)
  - `stable_rank`: ‖G‖_F² / ‖G‖²
  - `num_rank`: Count of eigenvalues above threshold
  - `mean_offdiag`: Mean off-diagonal element (key similarity)
  - `mean_nn_sim`: Mean nearest-neighbor cosine similarity
  - `max_nn_sim`: Maximum nearest-neighbor cosine similarity
  - `condition_num`: Condition number κ(G)
- Computes **poly2/linear ratios** for: `eff_rank`, `stable_rank`, `num_rank`, `mean_nn_sim`, `mean_offdiag`

### Script
```bash
bash scripts/run_polykernel_diagnostic.sh [SEED] [ALG_NAME] [DATASET_SIZE_LIMIT]
```

### Results Path
```
$RESULT_ROOT/polykernel_diagnostic/
    keys_{ALG}_seed{SEED}.pt                    # Raw keys tensor
    analysis_{ALG}_seed{SEED}.json              # Full diagnostic results
```

---

## Experiment 11: Poly2 Diagnostic (Extended)

### Scientific Question
Same as polykernel editor, but with additional capability probing (perplexity measurement at intervals) and optional ordering control.

### How It Works
- **Script**: `scripts/run_poly2_diagnostic.sh SEED [TARGET_EDITS]`
- **Runner**: `src/polykernel/polykernel_editor_runner.py` with `--capability_probe_interval 10`
- Adds WikiText perplexity measurement every 10 batches alongside standard metrics

### Results Path
Same structure as polykernel_editor, with additional perplexity data in the JSONL log.

---

## Experiment 12: Update Interference

### Scientific Question
When we apply weight update ΔW at batch t, how much does it interfere with previously-installed edits? Does clustered ordering cause more destructive interference than dispersed ordering?

### Multi-Phase Pipeline

The interference experiment is a complex multi-phase pipeline:

| Phase | Name | Compute | What It Does |
|-------|------|---------|------------|
| 0 | Verification | CPU | Checks checkpoint integrity |
| 1 | Coarse interference | CPU | Computes ΔW = W_after - W_before between checkpoints, measures ‖ΔW @ K_prior‖ |
| eval | Behavioral eval | GPU | Loads 5K checkpoint, evaluates ALL records for per-case retained/forgotten |
| install_eval | Install eval | GPU | Evaluates first 1K records at the 1K checkpoint (with NLL margins) |
| extract_base | Base weight | GPU | Extracts pre-edit W₀ from the base model |
| directional | Alignment | CPU | Measures cosine alignment between ΔW directions and prior key directions |
| install_analyze | Installation strength | CPU | Logistic regression predicting forgetting from geometric features |
| phase2 | Fine-grained | GPU | Runs actual AlphaEdit editing with per-batch delta capture and inline interference measurement |

### Fine-Grained Interference Measurement (Phase 2)

For each batch t with weight update ΔW_t:
```python
effects = ΔW_t @ all_keys.T     # [d_out, n_total_keys]
norms = torch.norm(effects, dim=0)  # per-key interference magnitude
I_fro = norms / (‖ΔW_t‖_F * ‖key‖ + ε)  # normalized

# Only accumulate for keys installed BEFORE batch t
eligible = installation_batch < t
path_interference[eligible] += norms[eligible]
```

### Installation Strength Model

A logistic regression predicts whether an edit is forgotten, using features:
- `position`: Normalized batch position (0=first, 1=last)
- `margin_1k/10`: NLL margin at the 1K checkpoint (how strongly installed), scaled by 1/10
- `e_norm`: Installation effect norm ‖(W_1K - W₀) @ k_i‖ — measures how strongly the 1K-edit weight change affects this key direction
- `future_max_cos`: Maximum cosine similarity between this key and any future key (stored internally as `future_key_max_sim`, renamed for display)

Key finding: `future_max_cos` is highly predictive of forgetting under dispersed ordering (the coefficient is negative: high future similarity → forgetting), while `position` dominates under clustered ordering.

### How It Works
- **Script**: `scripts/run_interference_experiment.sh SEED PHASE [ORDERING]`
- **Runners**: `src/experiments/interference_from_checkpoints.py`, `src/experiments/directional_alignment.py`, `src/experiments/installation_strength.py`, `src/runners/update_interference_runner.py`
- Operates on matched ordering checkpoints (requires Experiment 7 to have completed)

### Results Path
```
$RESULT_ROOT/interference/
    installation_strength_seed{SEED}.json           # Logistic model: all layers joint
    installation_strength_seed{SEED}_layer{L}.json  # Per-layer models
    base_weight_layer{L}.npy                        # Extracted base model weights
    AlphaEdit/{ORDERING}/seed{SEED}/
        phase1_coarse.json                          # Checkpoint-difference interference
        percase_eval.json                           # Per-case behavioral eval at 5K
        percase_eval_1k.json                        # Per-case eval at 1K
        directional_alignment.json                  # ΔW–key cosine alignment (layer 6)
        directional_alignment_layer{L}.json         # Multi-layer directional
        directional_comparison.json                 # Cross-ordering comparison
        installation_features.json                  # Geometric features for logistic model
        installation_features_layer{L}.json         # Multi-layer features
        fine_grained.json                           # Phase 2: per-batch delta + inline interference
```

---

## Experiment 13: Mechanism Analysis

### Scientific Question
How do the internal properties of the model (weight spectrum, cache geometry) evolve as edits accumulate?

### Data
Operates post-hoc on failure curve checkpoints — no new editing, just diagnostic measurements.

### Measurement (at each checkpoint boundary, per layer)

**Weight-spectrum distortion** (`weight_spectrum` field):
- `relative_perturbation`: ‖ΔW‖_F / ‖W₀‖_F
- `spectral_norm_update`: Largest singular value of ΔW
- `W0_frobenius`, `delta_frobenius`: Frobenius norms
- `stable_rank_base`, `stable_rank_current`: ‖W‖²_F / σ₁² (base vs edited)
- `sv_entropy_base`, `sv_entropy_current`: Shannon entropy of normalized singular values
- `top5_sv_base`, `top5_sv_current`: Top 5 singular values
- `top10_sv_change`, `top5_sv_relative_change`: How top SVs shifted
- `update_energy_in_original_subspace`: Fraction of ΔW energy in the original top-k SV subspace
- `principal_angles_top32`: Angles between base and edited principal subspaces

**Cache geometry** (`cache` field):
- `cache_numerical_rank`: Count of eigenvalues above threshold (1e-5 × max)
- `cache_effective_rank`: Exponential entropy of normalized eigenvalues
- `cache_stable_rank`: ‖C‖²_F / σ₁²
- `cache_condition`: Condition number σ₁/σ_min
- `cache_trace`: Total trace of cache_c
- `cache_top5_svs`: Top 5 singular values
- `cache_1pct_sv`, `cache_5pct_sv`, `cache_10pct_sv`: Singular value at 1%/5%/10% position
- `cache_top_sv_share`: σ₁ / Σσᵢ (concentration)

### How It Works
- **Script**: `scripts/run_mechanism_analysis.sh SEED`
- **Runner**: `src/mechanism/mechanism_analyzer.py`
- Loads base model, then loads checkpoint weights at each batch boundary (batch_9, batch_19, ..., batch_99)
- Computes diagnostics comparing edited weights to base weights

### Results Path
```
$RESULT_ROOT/mechanism_analysis/seed{SEED}/{ALG}/
    mechanism_seed{SEED}_{timestamp}.jsonl
```

Each JSONL line is a flat per-layer record (one record per layer per checkpoint):
```json
{
  "batch_idx": 9,
  "total_edits": 1000,
  "layer_idx": 5,
  "layer_position": "model.layers.5.mlp.down_proj",
  "weight_spectrum": {"relative_perturbation": 0.0042, "spectral_norm_update": 1.23, ...},
  "cache": {"cache_numerical_rank": 3200, "cache_effective_rank": 1800.5, "cache_condition": 1e4, ...},
  "seed": 42,
  "algorithm": "AlphaEdit",
  "model": "meta-llama/Meta-Llama-3-8B-Instruct"
}
```

### Checkpoints
None (reads from failure curve checkpoints at `$CHECKPOINT_ROOT/failure_curve/AlphaEdit/seed{SEED}/`).

---

## Experiment 14: Cache Mitigation Sweep

### Scientific Question
Can we mitigate the cache_c growth problem (which causes AlphaEdit degradation) through cache management strategies?

### Strategies Tested (12 variants)

**SVD Truncation** (6 variants): Every K batches, decompose cache_c via SVD and keep only the top `retain_ratio` singular values.
- Intervals: 5, 10 batches
- Retain ratios: 0.5, 0.75, 0.9

**Exponential Decay** (3 variants): After each batch, decay the cache: `cache_c *= decay_factor`
- Decay factors: 0.90, 0.95, 0.99

**Periodic Reset** (3 variants): Every K batches, zero out cache_c entirely.
- Reset intervals: 5, 10, 20 batches

### Data
MultiCounterFact, 2000 edits (configurable), batches of 100.

### Measurement
Standard editing metrics at each GLUE eval step. Comparison against unmodified AlphaEdit baseline.

### How It Works
- **Script**: `scripts/run_mitigation_sweep.sh [SEED] [DATASET_SIZE_LIMIT]`
- **Runner**: `src/mechanism/cache_mitigation_runner.py`
- Injects cache modification logic after the standard `cache_c += K@K^T` update

### Results Path
```
$RESULT_ROOT/mitigation/
    (per-variant subdirectories with standard evaluation JSONs)
```

---

## Experiment 15: Predictive Divergence (Extended Cache Metrics)

### Scientific Question
Can we identify early-warning indicators from the cache geometry that predict when AlphaEdit will start failing?

### Measurement
Computes four extended metrics from raw `cache_c.pt` checkpoints:
- **Rayleigh quotient**: How much the top eigenvector of cache_c aligns with the key directions
- **Key crowding proxy**: Measures how concentrated keys are in few dimensions
- **Linear effective rank**: Shannon entropy-based dimensionality of cache_c
- **Top eigenvalue concentration**: Fraction of trace captured by the top eigenvalue

### How It Works
- **Script**: `scripts/run_predictive_divergence_gpu.sh [SEED]`
- **Runner**: `analysis/predictive_divergence_gpu.py`
- Requires failure curve checkpoints with `cache_c.pt` files

### Results Path
```
$RESULT_ROOT/predictive_divergence/
    extended_cache_metrics_seed{SEED}.json
```

---

## Experiment 16: Projector Diagnostics (P ≈ I Verification)

### Scientific Question
Is AlphaEdit's null-space projection P actually constraining anything on Llama-3-8B? Or is P ≈ I, making the constraint vacuous?

### Measurement
Three-part analysis:

1. **Covariance eigenspectrum**: Load Wikipedia statistics (`mom2_100000.npz`), normalize by sample count, compute eigenvalues, count how many are below threshold (2e-2)

2. **Projector verification**: Load saved `null_space_project.pt`, verify:
   - `tr(P)` = number of retained dimensions
   - Idempotence: `‖P² - P‖_F / ‖P‖_F` should be ≈ 0
   - Symmetry: `‖P - P^T‖_F / ‖P‖_F` should be ≈ 0
   - Eigenvalue analysis: eigenvalues should be clustered at 0 or 1

3. **Cross-validation**: Reconstruct P from the covariance statistics using the same algorithm as `evaluate.py`, compare with the saved P file

### Key Findings
- P retains 99.7–99.9% of dimensions per layer (only 10–43 excluded out of 14336)
- Idempotence error: ~4e-6 (valid orthogonal projector)
- **The null-space constraint is effectively vacuous on Llama-3-8B with threshold=2e-2**

### How It Works
- **Script**: `scripts/run_projector_diagnostics.sh [STATS_DIR] [PROJECTOR_PATH]`
- Inline Python (no separate runner file)
- Requires precomputed covariance statistics and saved projector file

### Results Path
```
$RESULT_ROOT/projector_diagnostics/
    projector_diagnostics.json
```

---

## Experiment 17: Key Vector Extraction

### Scientific Question
Support experiment: pre-extracts key vectors used by matched ordering and interference analysis.

### Data
Runs the base model on MCF prompts and captures the MLP down_proj input at subject's last token position for specified layers.

### How It Works
- **Runner**: `src/mechanism/compute_keys.py`
- Registers a forward hook on `model.layers.{L}.mlp.down_proj`
- Runs each prompt, captures the hook input at the subject's last token position
- Used by: `src/datasets/generate_orderings.py`, interference analysis

### Results Path
```
$RESULT_ROOT/key_vectors/seed{SEED}/
    keys_seed{SEED}.npz         # Key vectors for all records, primary layer
    keys_seed{SEED}_meta.json   # Metadata (case_ids, subjects, layer)
$RESULT_ROOT/key_vectors/full_mcf/
    keys_seed{SEED}_layer{L}.npz   # Full MCF keys for specific layers
```

Also used for matched ordering:
```
$RESULT_ROOT/matched_ordering/key_geometry/
    keys_seed{SEED}_layer{L}.npz
```

---

## Path Resolution Summary

### Environment Variables

| Variable | Local Default | SkyPilot |
|----------|--------------|----------|
| `RESULT_ROOT` | `./results` | `/s3-data/continual-learning/alphaedit/results` |
| `CHECKPOINT_ROOT` | `~/.cache/alphaedit_checkpoints` | `/s3-data/continual-learning/alphaedit/checkpoints` |

### Complete Results Directory
```
$RESULT_ROOT/
├── mve1_alphaedit_mcf/seed{N}/2000edits/AlphaEdit/run_000/
├── mve2_memit_mcf/seed{N}/2000edits/MEMIT/run_000/
├── mve3_alphaedit_zsre/seed{N}/2000edits/AlphaEdit/run_000/
├── failure_curve_checkpointed/seed{N}/{E}edits/{ALG}/run_000/
├── comparison_ordered/seed{N}/{E}edits/order{0-9}/{ALG}/run_000/
├── matched_ordering/
│   ├── orderings/{ORDERING}_seed{N}.json
│   ├── key_geometry/keys_seed{N}_layer{L}.npz
│   └── {ALG}/{ORDERING}/seed{N}/
├── polykernel_editor/seed{N}/{E}edits/{ALG}-{kernel}/
├── polykernel_diagnostic/analysis_{ALG}_seed{N}.json
├── capability_probe/seed{N}/{ALG}/offline_probe_*.jsonl
├── mechanism_analysis/seed{N}/{ALG}/mechanism_*.jsonl
├── interference/AlphaEdit/{ORDERING}/seed{N}/
├── key_vectors/seed{N}/keys_seed{N}.npz
├── projector_diagnostics/projector_diagnostics.json
└── figures/
```

### Complete Checkpoint Directory
```
$CHECKPOINT_ROOT/
├── failure_curve/{ALG}/seed{N}/batch_{B}/
├── comparison_ordered/{ALG}/seed{N}/order{O}/batch_{B}/
└── matched_ordering/{ALG}/{ORDERING}/seed{N}/batch_{B}/
```

---

## Reproducibility

- **Seeds**: 42, 137, 2024, 7, 99 (5 seeds per experiment)
- **Deterministic seeding**: Python `random`, NumPy, PyTorch CPU+CUDA, cuDNN deterministic mode
- **Frozen dependencies**: `uv.lock`
- **Pinned vendor code**: Git submodule at commit `b84624f`
- **Source injection at anchor points**: Pinned to exact line patterns in vendor code

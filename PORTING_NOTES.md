# Cross-Architecture Porting Notes

Documents all model-specific assumptions found during Phase 0 codebase discovery
and how they were resolved for Qwen2.5-7B-Instruct and GPT-J-6B support.

## Integration Points

| # | File | Lines | Assumption | Resolution |
|---|------|-------|------------|------------|
| 1 | `vendor/AlphaEdit/experiments/evaluate.py` | 197-204 | Hardcoded model-name list for `cache_c`/`P` tensor shape (`W_out.shape[1]` for GPT-J/Llama/Phi, `W_out.shape[0]` for GPT-2) | Runtime patch adds `"Qwen2.5-7B"` to list. GPT-J already present. See `src/util/source_patches.py:apply_model_list_patch()` |
| 2 | `vendor/AlphaEdit/experiments/py/eval_utils_counterfact.py` | 149-159 | BOS token stripping: `'llama' in model.config._name_or_path.lower()` | **No change needed.** Qwen2.5 (`_name_or_path = "Qwen/Qwen2.5-7B-Instruct"`) and GPT-J (`"EleutherAI/gpt-j-6b"`) do not match this check, and their tokenizers do not add BOS. |
| 3 | `vendor/AlphaEdit/glue_eval/useful_functions.py` | 36-43 | Context length map keyed by lowercase model name | Runtime patch adds `"qwen2.5-7b-instruct": 4096` and `"gpt-j-6b": 2048`. The GPT-J entry also fixes a pre-existing bug where only `"eleutherai_gpt-j-6b"` was mapped (HF repo basename is `"gpt-j-6b"`). |
| 4 | `vendor/AlphaEdit/AlphaEdit/compute_z.py` | 83-88 | Hidden size detection: `config.n_embd` (GPT-2/GPT-J) vs `config.hidden_size` (Llama/Qwen) | **No change needed.** Code already handles both attribute names via fallback. |
| 5 | `vendor/AlphaEdit/rome/layer_stats.py` | 121-122 | Qwen2 max sequence length: `model.config.model_type == 'qwen2'` → maxlen=4096 | **No change needed.** Already handles Qwen2 model type. |
| 6 | `src/util/source_patches.py` | 35 | P-cache S3 path hardcoded `"llama3-8b-instruct"` | Generalized to use `hparams.model_name.lower()`. |
| 7 | `scripts/link_stats.sh` | 20-21 | Stats path hardcoded to llama3 | Added `stats_subdir_for_model()` routing function. |
| 8 | `vendor/AlphaEdit/glue_eval/*.py` (8 files) | 24 occurrences | BOS stripping via `'llama' in _name_or_path` | **No change needed.** Same reasoning as #2. |

## BOS Token Behavior

| Model | `add_bos_token` | `bos_token_id` | Eval code fires BOS strip? |
|-------|-----------------|----------------|---------------------------|
| Llama-3-8B-Instruct | True | 128000 (`<\|begin_of_text\|>`) | Yes (`'llama' in path`) |
| Qwen2.5-7B-Instruct | False | 151643 (exists but not prepended) | No |
| GPT-J-6B | False | None (uses `eos_token` as fallback) | No |

The vendor code's `'llama' in _name_or_path.lower()` check works correctly for all three models because Qwen and GPT-J tokenizers do not prepend BOS tokens by default.

## Leading-Space Convention

All models use the same leading-space target convention (AlphaEdit_main.py line 39-41):
```python
if request["target_new"]["str"][0] != " ":
    requests[i]["target_new"]["str"] = " " + request["target_new"]["str"]
```

This ensures the first token of the target is tokenized as a continuation token (with leading whitespace) rather than a word-initial token. This convention is universal across BPE tokenizers.

## Weight Matrix Dimension Convention

The `evaluate.py` tensor initialization uses `W_out.shape[1]` for GPT-J/Llama/Qwen/Phi and `W_out.shape[0]` for GPT-2. The distinction:
- **GPT-2**: `c_proj` weight is `[hidden_size, hidden_size]` → use `shape[0]`
- **GPT-J/Llama/Qwen**: `fc_out`/`down_proj` weight is `[intermediate_size, hidden_size]` → use `shape[1]`

All three of our target models use `shape[1]`.

## Hyperparameter Provenance

| Model | Source | Commit | Key Differences from Llama-3 |
|-------|--------|--------|------------------------------|
| GPT-J-6B | Official AlphaEdit repo (`vendor/AlphaEdit/hparams/AlphaEdit/EleutherAI_gpt-j-6B.json`) | b84624f (pinned) | layers=[3-8] (6 layers vs 5), v_lr=5e-1 (vs 1e-1) |
| Qwen2.5-7B | EasyEdit (`github.com/zjunlp/EasyEdit`) | 95a0db4 (2025-02-18) | v_lr=5e-1, v_weight_decay=1e-3, clamp_norm_factor=4, L2=1 |

## Statistics Generation

Per-model covariance stats must be generated before runs. Each model requires ~2-4 GPU-hours:

```bash
# Generate stats (requires GPU)
uv run python scripts/build_stats.py --model Qwen/Qwen2.5-7B-Instruct
uv run python scripts/build_stats.py --model EleutherAI/gpt-j-6b

# Verify + report retained dimensions
uv run python scripts/build_stats.py --model Qwen/Qwen2.5-7B-Instruct --verify_only
```

Expected stats files:
- **Qwen2.5-7B**: `data/stats/qwen2.5-7b-instruct/wikipedia_stats/model.layers.{4..8}.mlp.down_proj_float32_mom2_100000.npz`
- **GPT-J-6B**: `data/stats/gpt-j-6b/wikipedia_stats/transformer.h.{3..8}.mlp.fc_out_float32_mom2_100000.npz`

## Retained Null-Space Dimensions

For Llama-3-8B, the null-space projector retains 99.7-99.9% of dimensions (near-identity P matrix). The `build_stats.py` script reports this fraction for each model in `retained_dims_report.json`. Whether Qwen/GPT-J projectors are similarly near-identity is itself a reportable observation.

## Clustering Layer Selection

The key extraction layer for stream ordering experiments uses the middle of the edit-layer range:

| Model | Edit Layers | Clustering Layer |
|-------|-------------|-----------------|
| Llama-3-8B | [4, 5, 6, 7, 8] | 6 |
| GPT-J-6B | [3, 4, 5, 6, 7, 8] | 5 |
| Qwen2.5-7B | [4, 5, 6, 7, 8] | 6 |

## Byte-Identical Llama-3 Verification

The refactored code must produce identical results for Llama-3:
1. All source patches are additive (extend lists, don't modify existing entries)
2. The P-cache local path (`null_space_project.pt`) is unchanged
3. BOS checks use the same `'llama' in ...` condition
4. Context length map retains all original entries
5. `seeded_runner.py` produces identical `sys.argv` for Llama-3

Verification command:
```bash
bash scripts/smoke_test.sh  # Must produce identical output to pre-refactor
```

## Known Considerations

1. **Qwen2.5-7B context window**: Native 131K tokens, capped to 4096 for covariance computation (matching the existing Llama-3 convention and the EasyEdit config).

2. **GPT-J GLUE context length bug**: The vendor code maps `"eleutherai_gpt-j-6b": 2048` but when loaded from the HF repo `EleutherAI/gpt-j-6b`, the lookup key is `"gpt-j-6b"` (basename). Our patch adds both entries.

3. **Qwen2.5-7B hidden size**: 3584 (vs Llama's 4096). This means smaller P matrices (3584×3584 instead of 4096×4096 per layer), resulting in faster SVD computation.

4. **GPT-J has no instruct tuning**: Evaluation prompts are raw MCF prompts with no chat template, identical to the Llama-3-8B-Instruct setup (which also uses raw prompts for comparability).

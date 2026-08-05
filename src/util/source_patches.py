"""
Shared source patches for evaluate.py injection.

All runners that exec evaluate.py can use these patches to apply
common modifications without editing the submodule directly.
These are runtime patches (applied on disk before subprocess reads the file),
following the same pattern as the YAML sed patches.

Also provides shared source injection builders (order shuffle, fingerprint)
that multiple runners can use without duplicating injection logic.
"""

from pathlib import Path


# The original P computation block in evaluate.py (commit b84624f)
P_COMPUTE_ANCHOR = """\
    if alg_name == "AlphaEdit":
        for i, layer in enumerate(hparams.layers):
            P[i,:,:] = get_project(model,tok,layer,hparams)
        torch.save(P, "null_space_project.pt")"""

P_COMPUTE_CACHED = """\
    if alg_name == "AlphaEdit":
        if Path("null_space_project.pt").exists():
            P = torch.load("null_space_project.pt", map_location="cpu")
            print(f"Loaded cached null-space projection from null_space_project.pt")
        else:
            for i, layer in enumerate(hparams.layers):
                P[i,:,:] = get_project(model,tok,layer,hparams)
            torch.save(P, "null_space_project.pt")
            import os as _os
            _stats_root = Path(_os.environ.get("STATS_ROOT", ""))
            if not _stats_root.is_dir():
                _model_short = hparams.model_name.lower().replace("/", "-").replace("_", "-")
                _stats_root = Path(_os.environ.get("CHECKPOINT_ROOT", "")).parent / "stats" / _model_short
            if _stats_root.is_dir():
                torch.save(P, str(_stats_root / "null_space_project.pt"))
                print(f"Computed and cached null-space projection (persisted to {_stats_root})")
            else:
                print(f"Computed and cached null-space projection to null_space_project.pt")"""


# --- Model-name list patch for evaluate.py cache_c/P shape initialization ---
# The vendor evaluate.py (line 201) hardcodes which models use W_out.shape[1].
# Qwen2.5-7B needs to be added; GPT-J is already present.
SHAPE_MODEL_LIST_ANCHOR = '["EleutherAI_gpt-j-6B","Llama3-8B","phi-1.5"]'
SHAPE_MODEL_LIST_EXTENDED = '["EleutherAI_gpt-j-6B","Llama3-8B","phi-1.5","Qwen2.5-7B"]'


def apply_model_list_patch(source: str) -> str:
    """
    Add Qwen2.5-7B to the model-name list for cache_c/P tensor shape in evaluate.py.

    The vendor code uses W_out.shape[1] for GPT-J/Llama/Phi and W_out.shape[0] for
    GPT-2. Qwen2.5-7B follows the same convention as Llama (shape[1]).

    Idempotent — returns source unchanged if already patched.
    """
    if SHAPE_MODEL_LIST_ANCHOR not in source:
        return source
    return source.replace(SHAPE_MODEL_LIST_ANCHOR, SHAPE_MODEL_LIST_EXTENDED, 1)


# --- GLUE context length map patch ---
GLUE_MAP_ANCHOR = '"gpt2-medium": 1024'
GLUE_MAP_EXTENDED = '"gpt2-medium": 1024, "qwen2.5-7b-instruct": 4096, "gpt-j-6b": 2048'


def apply_glue_context_patch(source: str) -> str:
    """
    Add Qwen2.5-7B and GPT-J context length entries to glue_eval/useful_functions.py.

    The vendor code maps lowercase model names to max context length. Qwen2.5-7B
    uses 4096 (capped from 131K); GPT-J uses 2048. The GPT-J entry also fixes a
    pre-existing bug where "gpt-j-6b" (from HF repo basename) was not mapped.

    Idempotent — returns source unchanged if already patched.
    """
    if GLUE_MAP_ANCHOR not in source:
        return source
    if "qwen2.5-7b-instruct" in source:
        return source  # Already patched
    return source.replace(GLUE_MAP_ANCHOR, GLUE_MAP_EXTENDED, 1)


def apply_p_cache_patch(source: str) -> str:
    """
    Patch evaluate.py source to cache the null-space projection P.

    P depends only on model architecture + covariance stats + threshold (not on
    edits or seed). Computing it requires SVD on 5 x (14336x14336) matrices which
    takes ~45 minutes. This patch loads from a cached file if available.

    Args:
        source: The evaluate.py source text.

    Returns:
        Patched source with P-caching logic.
    """
    if P_COMPUTE_ANCHOR not in source:
        # Already patched or upstream changed — skip silently
        return source
    return source.replace(P_COMPUTE_ANCHOR, P_COMPUTE_CACHED, 1)


MODEL_LOAD_ANCHOR = '    model = AutoModelForCausalLM.from_pretrained(model_name).cuda()'
MODEL_LOAD_FP32 = (
    '    _load_dtype = torch.float32 if "qwen" in model_name.lower() else None\n'
    '    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=_load_dtype).cuda()\n'
    '    if _load_dtype: print(f"  [PATCH] Loaded {model_name} in {_load_dtype} (compute_z stability)")'
)


def apply_model_dtype_patch(source: str) -> str:
    """
    Patch evaluate.py to load Qwen models in float32 instead of default bf16.

    Qwen2.5-7B's config specifies torch_dtype=bfloat16, but compute_z's
    v-optimization (Adam lr=0.5, 25 steps) produces NaN when the forward
    pass runs in bf16 due to imprecise gradients through log_softmax.
    Loading in float32 fixes this; L40s (48GB) has headroom.

    Only affects Qwen models — Llama-3 works fine in bf16.

    Idempotent — returns source unchanged if already patched.
    """
    if MODEL_LOAD_ANCHOR not in source:
        return source
    return source.replace(MODEL_LOAD_ANCHOR, MODEL_LOAD_FP32, 1)


def patch_evaluate_file(alphaedit_root: Path) -> None:
    """
    Apply runtime patches to vendor/AlphaEdit/experiments/evaluate.py on disk.

    Idempotent — safe to call multiple times. Applies:
      - P-cache: loads null_space_project.pt if present instead of recomputing SVD
      - Model-list: adds Qwen2.5-7B to the cache_c/P shape initialization list
      - Model-dtype: loads Qwen in float32 for compute_z stability
      - Layer stats: fixes deprecated Wikipedia dataset config (20200501.en → 20220301.en)

    Args:
        alphaedit_root: Path to vendor/AlphaEdit/ directory.
    """
    eval_path = alphaedit_root / "experiments" / "evaluate.py"
    source = eval_path.read_text()
    patched = apply_p_cache_patch(source)
    patched = apply_model_list_patch(patched)
    patched = apply_model_dtype_patch(patched)
    if patched != source:
        eval_path.write_text(patched)
        print("  Applied P-cache + model-list + dtype patches to evaluate.py")
    # Also patch layer_stats.py (called by AlphaEdit_main.py for on-the-fly stats)
    patch_layer_stats_file(alphaedit_root)


def patch_layer_stats_file(alphaedit_root: Path) -> None:
    """
    Fix deprecated Wikipedia dataset config in vendor/AlphaEdit/rome/layer_stats.py.

    HuggingFace removed the '20200501.en' config; the equivalent is '20220301.en'.
    This affects stats computation (both build_stats.py and on-the-fly in AlphaEdit_main.py).

    Idempotent — safe to call multiple times.

    Args:
        alphaedit_root: Path to vendor/AlphaEdit/ directory.
    """
    stats_path = alphaedit_root / "rome" / "layer_stats.py"
    source = stats_path.read_text()
    if "20200501.en" not in source:
        return  # Already patched or upstream changed
    patched = source.replace("20200501.en", "20220301.en")
    stats_path.write_text(patched)
    print("  Applied Wikipedia dataset config patch to rome/layer_stats.py")


def patch_glue_eval_file(alphaedit_root: Path) -> None:
    """
    Apply runtime patches to vendor/AlphaEdit/glue_eval/useful_functions.py on disk.

    Idempotent — safe to call multiple times. Applies:
      - Context length map: adds Qwen2.5-7B and GPT-J entries

    Args:
        alphaedit_root: Path to vendor/AlphaEdit/ directory.
    """
    glue_path = alphaedit_root / "glue_eval" / "useful_functions.py"
    source = glue_path.read_text()
    patched = apply_glue_context_patch(source)
    if patched != source:
        glue_path.write_text(patched)
        print("  Applied GLUE context-length patch to useful_functions.py")


# --- Source anchor used by order shuffle injection ---
SHUFFLE_ANCHOR = '    for record_chunks in chunks(ds, num_edits):'


def build_order_shuffle_injection(order_id: int) -> str:
    """
    Build source injection code for dataset shuffling by order_id.

    Injected into evaluate.py BEFORE the main edit loop. If order_id == 0,
    no shuffle is performed (canonical ordering). If order_id > 0, the
    dataset is shuffled using Random(order_id).

    Args:
        order_id: Shuffle seed. 0 = canonical (no shuffle), >0 = shuffle.

    Returns:
        Python source code string to inject before the loop anchor.
        Empty string if order_id == 0.
    """
    if order_id == 0:
        return ""

    return (
        f'    # === ORDER SHUFFLE: shuffle dataset with order_id={order_id} (injected) ===\n'
        f'    import random as _order_rng_module\n'
        f'    _order_rng = _order_rng_module.Random({order_id})\n'
        f'    _shuffled_indices = list(range(len(ds)))\n'
        f'    _order_rng.shuffle(_shuffled_indices)\n'
        f'    ds.data = [ds.data[i] for i in _shuffled_indices]\n'
        f'    print("ORDER SHUFFLE: shuffled " + str(len(ds)) + " records with order_id={order_id}")\n'
        f'    # === END order shuffle ===\n'
    )

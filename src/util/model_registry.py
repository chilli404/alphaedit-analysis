"""
Central model registry for cross-architecture support.

Maps model identifiers (HuggingFace repo IDs, short names, _name_or_path values)
to their architecture-specific configuration. Eliminates scattered model-name
conditionals throughout the codebase.

Usage:
    from model_registry import get_model_spec, MODEL_REGISTRY
    spec = get_model_spec("Qwen/Qwen2.5-7B-Instruct")
    print(spec.hidden_size)  # 3584
"""

from dataclasses import dataclass, field


@dataclass(frozen=True)
class ModelSpec:
    """Architecture-specific configuration for a model."""

    # Identity
    short_name: str  # hparams model_name field: "Qwen2.5-7B", "EleutherAI_gpt-j-6B"
    hf_repo: str  # Primary HuggingFace repo ID

    # Architecture
    hidden_size: int
    n_layers: int
    edit_layers: tuple[int, ...]
    weight_dim_index: int  # Which dim of W_out is hidden_size (0 for gpt2, 1 for others)

    # Tokenizer / evaluation
    context_length: int
    has_bos_token: bool  # Whether tokenizer prepends BOS by default

    # Experiment configuration
    clustering_layer: int  # Middle of edit layers, used for key extraction in ordering
    stats_dir_name: str  # Subdirectory name under data/stats/

    # Lookup keys
    glue_map_keys: tuple[str, ...]  # Lowercase keys for glue_eval context length map
    hparams_fname: str  # JSON filename in hparams/{ALG}/

    # All identifiers that should resolve to this model
    aliases: tuple[str, ...] = field(default_factory=tuple)


# ─── Registry ────────────────────────────────────────────────────────────────

LLAMA3_8B = ModelSpec(
    short_name="Llama3-8B",
    hf_repo="meta-llama/Meta-Llama-3-8B-Instruct",
    hidden_size=4096,
    n_layers=32,
    edit_layers=(4, 5, 6, 7, 8),
    weight_dim_index=1,
    context_length=4096,
    has_bos_token=True,
    clustering_layer=6,
    stats_dir_name="llama3-8b-instruct",
    glue_map_keys=("llama3-8b-instruct", "meta-llama-3-8b-instruct"),
    hparams_fname="Llama3-8B.json",
    aliases=(
        "Llama3-8B",
        "meta-llama/Meta-Llama-3-8B-Instruct",
        "Meta-Llama-3-8B-Instruct",
        "nousresearch/Meta-Llama-3-8B-Instruct",
        "nousresearch--meta-llama-3-8b-instruct",
    ),
)

GPTJ_6B = ModelSpec(
    short_name="EleutherAI_gpt-j-6B",
    hf_repo="EleutherAI/gpt-j-6b",
    hidden_size=4096,
    n_layers=28,
    edit_layers=(3, 4, 5, 6, 7, 8),
    weight_dim_index=1,
    context_length=2048,
    has_bos_token=False,
    clustering_layer=5,  # Middle of [3,4,5,6,7,8]
    stats_dir_name="gpt-j-6b",
    glue_map_keys=("gpt-j-6b", "eleutherai_gpt-j-6b", "eleutherai/gpt-j-6b"),
    hparams_fname="EleutherAI_gpt-j-6B.json",
    aliases=(
        "EleutherAI_gpt-j-6B",
        "EleutherAI/gpt-j-6b",
        "gpt-j-6b",
        "gpt-j-6B",
    ),
)

QWEN25_7B = ModelSpec(
    short_name="Qwen2.5-7B",
    hf_repo="Qwen/Qwen2.5-7B-Instruct",
    hidden_size=3584,
    n_layers=28,
    edit_layers=(4, 5, 6, 7, 8),
    weight_dim_index=1,
    context_length=4096,  # Capped from 131K for covariance computation
    has_bos_token=False,
    clustering_layer=6,  # Middle of [4,5,6,7,8]
    stats_dir_name="qwen2.5-7b-instruct",
    glue_map_keys=("qwen2.5-7b-instruct",),
    hparams_fname="Qwen2.5-7B.json",
    aliases=(
        "Qwen2.5-7B",
        "Qwen/Qwen2.5-7B-Instruct",
        "Qwen2.5-7B-Instruct",
    ),
)

# All registered models
MODEL_REGISTRY: dict[str, ModelSpec] = {
    spec.short_name: spec
    for spec in [LLAMA3_8B, GPTJ_6B, QWEN25_7B]
}

# Alias lookup (case-insensitive)
_ALIAS_MAP: dict[str, ModelSpec] = {}
for _spec in MODEL_REGISTRY.values():
    for _alias in _spec.aliases:
        _ALIAS_MAP[_alias.lower()] = _spec
    _ALIAS_MAP[_spec.short_name.lower()] = _spec
    _ALIAS_MAP[_spec.hf_repo.lower()] = _spec


def get_model_spec(name_or_path: str) -> ModelSpec:
    """Resolve any model identifier to its ModelSpec.

    Accepts: HF repo ID, short_name, _name_or_path, or any registered alias.
    Case-insensitive matching.

    Raises:
        KeyError: If the model is not registered.
    """
    key = name_or_path.lower().strip()

    # Direct lookup
    if key in _ALIAS_MAP:
        return _ALIAS_MAP[key]

    # Try the last path component (handles "path/to/Model-Name")
    basename = key.rsplit("/", 1)[-1]
    if basename in _ALIAS_MAP:
        return _ALIAS_MAP[basename]

    # Try replacing / with _ (handles "EleutherAI/gpt-j-6b" -> "eleutherai_gpt-j-6b")
    underscore_form = key.replace("/", "_")
    if underscore_form in _ALIAS_MAP:
        return _ALIAS_MAP[underscore_form]

    raise KeyError(
        f"Model '{name_or_path}' not found in registry. "
        f"Known models: {list(MODEL_REGISTRY.keys())}"
    )


def get_evaluate_model_names() -> list[str]:
    """Return the list of model short_names that use W_out.shape[1] for hidden dim.

    This corresponds to the model-name list at evaluate.py line 201.
    All registered models except gpt2-xl use dim index 1.
    """
    return [spec.short_name for spec in MODEL_REGISTRY.values() if spec.weight_dim_index == 1]

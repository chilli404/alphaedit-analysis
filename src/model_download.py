"""
Model download utility with Artifactory fallback.

When standard HuggingFace Hub access fails (gated repos, auth issues),
falls back to an internal Artifactory-hosted mirror.

Usage:
    from model_download import resolve_model_path
    model_path = resolve_model_path("meta-llama/Meta-Llama-3-8B-Instruct")
    # Returns either the original model_id if HF works, or a local path
"""

import os
from pathlib import Path

from huggingface_hub import snapshot_download


# Internal HuggingFace mirror endpoint
HF_ENDPOINT = "https://graingerreadonly.jfrog.io/artifactory/api/huggingfaceml/huggingfaceml-remote"


def get_default_hf_cache_dir() -> Path:
    """
    Return the default Hugging Face Hub cache directory.

    Default behavior:
        HF_HUB_CACHE, if set
        otherwise $HF_HOME/hub
        otherwise ~/.cache/huggingface/hub
    """
    hf_hub_cache = os.getenv("HF_HUB_CACHE")
    if hf_hub_cache:
        return Path(hf_hub_cache).expanduser()

    hf_home = os.getenv("HF_HOME")
    if hf_home:
        return Path(hf_home).expanduser() / "hub"

    xdg_cache_home = os.getenv("XDG_CACHE_HOME")
    if xdg_cache_home:
        return Path(xdg_cache_home).expanduser() / "huggingface" / "hub"

    return Path.home() / ".cache" / "huggingface" / "hub"


def download_model(
    model_id: str,
    cache_dir: Path | str | None = None,
    revision: str | None = None,
    local: bool = False,
) -> str:
    """
    Download a model from HuggingFace with custom endpoint configuration.

    Args:
        model_id: HuggingFace model identifier
        cache_dir: Local cache directory. Defaults to Hugging Face's default hub cache.
        revision: Specific revision/commit to download
        local: If True, download to a flat local directory

    Returns:
        str: Local path to the downloaded model
    """
    cache_dir = Path(cache_dir).expanduser() if cache_dir else get_default_hf_cache_dir()
    cache_dir.mkdir(parents=True, exist_ok=True)

    print(f"Downloading model: {model_id}")
    print(f"Cache directory: {cache_dir}")

    try:
        args = {
            "repo_id": model_id,
            "etag_timeout": 86400,
            "force_download": False,
            "endpoint": HF_ENDPOINT,
        }

        if local:
            # Use a stable model-specific directory inside the HF cache.
            # This avoids dumping files directly into ~/.cache/huggingface/hub.
            safe_model_name = model_id.replace("/", "--")
            local_dir = cache_dir / safe_model_name
            local_dir.mkdir(parents=True, exist_ok=True)
            args["local_dir"] = str(local_dir)
        else:
            args["cache_dir"] = str(cache_dir)

        if revision:
            args["revision"] = revision

        model_path = snapshot_download(**args)

        if not os.path.exists(model_path):
            raise FileNotFoundError(f"No files found at downloaded model path: {model_path}")

        if local:
            Path(model_path, "__init__.py").touch()

        print(f"Model downloaded successfully to: {model_path}")
        return model_path

    except Exception as e:
        print(f"Error downloading model {model_id}: {e}")
        raise


def resolve_model_path(
    model_id: str,
    cache_dir: Path | str | None = None,
) -> str:
    """
    Resolve a model identifier to a usable path.

    Tries standard HuggingFace access first. If that fails, falls back to
    the internal Artifactory mirror with local=True.

    Args:
        model_id: HuggingFace model identifier, e.g.
            "meta-llama/Meta-Llama-3-8B-Instruct"
        cache_dir: Where to download on fallback.
            Defaults to Hugging Face's default hub cache.

    Returns:
        str: Either the original model_id if HF works, or a local path
    """
    from huggingface_hub import model_info

    try:
        model_info(model_id)
        print(f"Model accessible on HuggingFace: {model_id}")
        return model_id
    except Exception as e:
        print(f"HuggingFace access failed for {model_id}: {e}")
        print("Falling back to Artifactory mirror...")

    local_path = download_model(model_id, cache_dir=cache_dir, local=True)
    return local_path
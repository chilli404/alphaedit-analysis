"""
Model download utility with Artifactory fallback.

When standard HuggingFace Hub access fails (gated repos, auth issues),
falls back to an internal Artifactory-hosted mirror.

Usage:
    from model_download import resolve_model_path
    model_path = resolve_model_path("meta-llama/Meta-Llama-3-8B-Instruct")
    # Returns either the original model_id (if HF works) or a local path
"""

import os
from pathlib import Path

from huggingface_hub import snapshot_download


# Internal HuggingFace mirror endpoint
HF_ENDPOINT = "https://graingerreadonly.jfrog.io/artifactory/api/huggingfaceml/huggingfaceml-remote"


def download_model(
    model_id: str,
    cache_dir: Path = None,
    revision: str = None,
    local: bool = False,
) -> str:
    """
    Download a model from HuggingFace with custom endpoint configuration.

    Args:
        model_id: HuggingFace model identifier
        cache_dir: Local cache directory (defaults to /local_disk0)
        revision: Specific revision/commit to download
        local: If True, download to a flat local directory

    Returns:
        str: Local path to the downloaded model
    """
    if cache_dir is None:
        cache_dir = Path("/local_disk0")
    else:
        cache_dir = Path(cache_dir)

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
            args["local_dir"] = str(cache_dir)
        else:
            args["cache_dir"] = str(cache_dir)

        if revision:
            args["revision"] = revision

        model_path = snapshot_download(**args)

        if not os.path.exists(model_path):
            raise FileNotFoundError(f"No files found at downloaded model path: {model_path}")
        if local:
            open(f"{model_path}/__init__.py", "a").close()
        print(f"Model downloaded successfully to: {model_path}")
        return model_path

    except Exception as e:
        print(f"Error downloading model {model_id}: {e}")
        raise


def resolve_model_path(
    model_id: str,
    cache_dir: Path = None,
) -> str:
    """
    Resolve a model identifier to a usable path.

    Tries standard HuggingFace access first. If that fails (gated repo,
    auth issues, network problems), falls back to the internal Artifactory
    mirror with local=True.

    Args:
        model_id: HuggingFace model identifier (e.g. "meta-llama/Meta-Llama-3-8B-Instruct")
        cache_dir: Where to download on fallback (defaults to /local_disk0)

    Returns:
        str: Either the original model_id (if HF works) or a local path
    """
    from huggingface_hub import model_info

    # Quick check: can we access this model on HuggingFace?
    try:
        model_info(model_id)
        print(f"  Model accessible on HuggingFace: {model_id}")
        return model_id
    except Exception as e:
        print(f"  HuggingFace access failed for {model_id}: {e}")
        print(f"  Falling back to Artifactory mirror...")

    # Fallback: download via internal mirror
    local_path = download_model(model_id, cache_dir=cache_dir, local=True)
    return local_path

"""Resolve model name to a local path or HuggingFace repo ID.

Checks MODEL_PATH env var first, then common local cache locations,
then returns the original name for HuggingFace download.
"""

import os
from pathlib import Path


_S3_MODEL_PATHS = [
    "/s3-data/continual-learning/models",
]


def resolve_model_path(model_name: str) -> str:
    """Resolve model name to a loadable path.

    Priority:
      1. MODEL_PATH environment variable (explicit override)
      2. Local path if model_name is already a directory
      3. S3 FUSE mount (/s3-data/continual-learning/models/)
      4. HuggingFace cache (~/.cache/huggingface/hub/)
      5. Return model_name as-is (transformers will download from HF Hub)
    """
    env_path = os.environ.get("MODEL_PATH")
    if env_path and os.path.isdir(env_path):
        return env_path

    if os.path.isdir(model_name):
        return model_name

    # Check S3 FUSE mount for common model name variants
    short_name = model_name.split("/")[-1]
    dash_name = model_name.replace("/", "--")
    candidates = [short_name, dash_name, short_name.replace("EleutherAI-", "")]
    # Also check shortened names and NousResearch mirror
    if "Meta-Llama" in short_name:
        candidates.append(f"NousResearch--{short_name}")
        # S3 stores as "Meta-Llama-3-8B" (no -Instruct suffix)
        candidates.append(short_name.replace("-Instruct", ""))
    for base in _S3_MODEL_PATHS:
        for candidate in candidates:
            full = os.path.join(base, candidate)
            if os.path.isdir(full):
                return full

    hf_cache = Path.home() / ".cache" / "huggingface" / "hub"
    cache_name = "models--" + model_name.replace("/", "--")
    cached = hf_cache / cache_name
    if cached.exists():
        snapshots = cached / "snapshots"
        if snapshots.exists():
            revisions = sorted(snapshots.iterdir(), key=lambda p: p.stat().st_mtime, reverse=True)
            for rev in revisions:
                has_model = any(rev.glob("*.safetensors")) or any(rev.glob("*.bin"))
                if has_model:
                    return str(rev)

    return model_name

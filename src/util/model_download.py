"""
Model download utility with Artifactory static token authentication.

On corporate infrastructure (K8s clusters), uses a JFrog Identity Token
(passed via HF_TOKEN env var) to download from the Artifactory HuggingFace
shared virtual repo. Locally, falls back to standard HuggingFace access.

Usage:
    from model_download import resolve_model_path
    model_path = resolve_model_path("meta-llama/Meta-Llama-3-8B-Instruct")
    # Returns either the original model_id if HF works, or a local path
"""

import os
from pathlib import Path

# Disable filelock before importing huggingface_hub/transformers — prevents
# hangs on SkyPilot cluster filesystems where flock() doesn't work reliably.
import filelock
import filelock._api

class _NoOpFileLock(filelock.BaseFileLock):
    """A file lock that always succeeds immediately without actually locking."""
    def _acquire(self):
        self._context.lock_file_fd = True

    def _release(self):
        self._context.lock_file_fd = None

# Patch everywhere filelock exposes lock classes
filelock.FileLock = _NoOpFileLock
filelock.SoftFileLock = _NoOpFileLock
filelock.UnixFileLock = _NoOpFileLock
filelock._api.FileLock = _NoOpFileLock

import sys
# Also patch any already-imported modules that grabbed a reference
for mod_name, mod in list(sys.modules.items()):
    if mod is None:
        continue
    for attr in ("FileLock", "SoftFileLock"):
        if hasattr(mod, attr) and isinstance(getattr(mod, attr), type) and \
           issubclass(getattr(mod, attr), filelock.BaseFileLock) and \
           getattr(mod, attr) is not _NoOpFileLock:
            setattr(mod, attr, _NoOpFileLock)

from huggingface_hub import snapshot_download


# Artifactory HuggingFace shared virtual repo
HF_ENDPOINT = "https://graingerinc.jfrog.io/artifactory/api/huggingfaceml/huggingfaceml-remote"


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
    Download a model from Artifactory using a static JFrog Identity Token.

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
    print(f"Endpoint: {HF_ENDPOINT}")

    try:
        token = os.getenv("HF_TOKEN")
        if not token:
            raise RuntimeError("HF_TOKEN not set — cannot authenticate to Artifactory")

        args = {
            "repo_id": model_id,
            "etag_timeout": 86400,
            "force_download": False,
            "endpoint": HF_ENDPOINT,
            "token": token,
        }

        if local:
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


def _artifactory_reachable() -> bool:
    """Check if Artifactory is reachable (i.e., on corporate infra)."""
    import urllib.request
    import urllib.error

    try:
        req = urllib.request.Request(
            "https://graingerinc.jfrog.io", method="HEAD"
        )
        urllib.request.urlopen(req, timeout=5)
        return True
    except (urllib.error.URLError, OSError):
        return False


def resolve_model_path(
    model_id: str,
    cache_dir: Path | str | None = None,
) -> str:
    """
    Resolve a model identifier to a usable path.

    On corporate infrastructure (Artifactory reachable), sets HF_ENDPOINT
    so that transformers downloads through Artifactory natively. This keeps
    model.config._name_or_path as the original model_id, which is required
    for stats lookup and GLUE eval context length mapping.

    Locally (Artifactory unreachable), returns the model_id unchanged for
    standard HuggingFace access.

    Args:
        model_id: HuggingFace model identifier, e.g.
            "meta-llama/Meta-Llama-3-8B-Instruct"
        cache_dir: Unused, kept for API compatibility.

    Returns:
        str: model_id (always). On corporate infra, HF_ENDPOINT is set
             so transformers routes through Artifactory automatically.
    """
    if _artifactory_reachable():
        os.environ["HF_ENDPOINT"] = HF_ENDPOINT
        print(f"Artifactory reachable — set HF_ENDPOINT={HF_ENDPOINT}")
    else:
        print(f"Artifactory not reachable — using standard HuggingFace")

    print(f"Model: {model_id}")
    return model_id

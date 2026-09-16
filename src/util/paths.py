"""Centralized path resolution for results and checkpoints.

Environment variables:
    RESULT_ROOT     — Base directory for all experiment results.
                      Default: {project_root}/results
    CHECKPOINT_ROOT — Base directory for all checkpoints.
                      Default: ~/.cache/alphaedit_checkpoints
"""

import os
from pathlib import Path

def _on_skypilot() -> bool:
    return bool(os.environ.get("SKYPILOT_TASK_ID")) and not os.environ.get("_PYTEST_RUNNING")


def _require_s3(path: Path, label: str) -> Path:
    if _on_skypilot() and "/s3-data/" not in str(path):
        raise RuntimeError(
            f"{label} must be on S3 (via /s3-data/) on SkyPilot clusters, "
            f"but resolved to: {path}\n"
            f"Set {label} env var to an /s3-data/... path."
        )
    return path


def get_project_root() -> Path:
    """Return the alphaedit-analysis/ project root directory."""
    return Path(__file__).resolve().parent.parent.parent


def get_alphaedit_root() -> Path:
    """Return the vendor/AlphaEdit/ directory."""
    return get_project_root() / "vendor" / "AlphaEdit"


def get_result_root() -> Path:
    """Return the base directory for experiment results.

    Priority: RESULT_ROOT env var > {project_root}/results
    On SkyPilot clusters, RESULT_ROOT must point to /s3-data/.
    """
    env = os.environ.get("RESULT_ROOT", "")
    if env:
        return _require_s3(Path(env), "RESULT_ROOT")
    p = get_project_root() / "results"
    return _require_s3(p, "RESULT_ROOT")


def get_checkpoint_root() -> Path:
    """Return the base directory for checkpoints.

    Priority: CHECKPOINT_ROOT env var > ~/.cache/alphaedit_checkpoints
    On SkyPilot clusters, CHECKPOINT_ROOT must point to /s3-data/.
    """
    env = os.environ.get("CHECKPOINT_ROOT", "")
    if env:
        return _require_s3(Path(env), "CHECKPOINT_ROOT")
    p = Path.home() / ".cache" / "alphaedit_checkpoints"
    return _require_s3(p, "CHECKPOINT_ROOT")

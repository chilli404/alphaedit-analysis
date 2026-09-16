"""Centralized path resolution for results and checkpoints.

Environment variables:
    RESULT_ROOT     — Base directory for all experiment results.
                      Default: {project_root}/results
    CHECKPOINT_ROOT — Base directory for all checkpoints.
                      Default: ~/.cache/alphaedit_checkpoints
"""

import os
from pathlib import Path


def get_project_root() -> Path:
    """Return the alphaedit-analysis/ project root directory."""
    return Path(__file__).resolve().parent.parent.parent


def get_alphaedit_root() -> Path:
    """Return the vendor/AlphaEdit/ directory."""
    return get_project_root() / "vendor" / "AlphaEdit"


def get_result_root() -> Path:
    """Return the base directory for experiment results.

    Priority: RESULT_ROOT env var > {project_root}/results
    """
    env = os.environ.get("RESULT_ROOT", "")
    if env:
        return Path(env)
    return get_project_root() / "results"


def get_checkpoint_root() -> Path:
    """Return the base directory for checkpoints.

    Priority: CHECKPOINT_ROOT env var > ~/.cache/alphaedit_checkpoints
    """
    env = os.environ.get("CHECKPOINT_ROOT", "")
    if env:
        return Path(env)
    return Path.home() / ".cache" / "alphaedit_checkpoints"

"""
Frozen evaluation configuration loader and validator.

Ensures all experiment runs use identical evaluation parameters
(context templates, decoding settings, prompt configuration) by:
  1. Loading the frozen config from configs/eval_config.yaml
  2. Computing a SHA-256 hash for provenance tracking
  3. Providing runtime assertions for source-injected code
"""

import hashlib
from pathlib import Path

import yaml


def _get_config_path() -> Path:
    """Resolve path to configs/eval_config.yaml."""
    project_root = Path(__file__).resolve().parent.parent.parent
    return project_root / "configs" / "eval_config.yaml"


def load_eval_config() -> dict:
    """
    Load the frozen evaluation configuration.

    Returns:
        Dict with keys: version, frozen_at, context_templates, decoding, evaluation
    """
    config_path = _get_config_path()
    if not config_path.exists():
        raise FileNotFoundError(
            f"Frozen eval config not found at {config_path}. "
            "Run scripts/freeze_eval_commit.sh to generate it."
        )
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def hash_eval_config() -> str:
    """
    Compute SHA-256 hash of the eval config file contents.

    Returns:
        Hex digest string (64 chars).
    """
    config_path = _get_config_path()
    if not config_path.exists():
        return "config_not_found"
    content = config_path.read_bytes()
    return hashlib.sha256(content).hexdigest()


def build_eval_config_assertion() -> str:
    """
    Build source injection code that verifies generation params at runtime.

    Injected into patched evaluate.py to assert that the decoding settings
    match the frozen config. Raises AssertionError if mismatch detected.

    Returns:
        Python source code string to inject (indented at function level).
    """
    config = load_eval_config()
    decoding = config.get("decoding", {})

    checks = []
    if "max_new_tokens" in decoding:
        checks.append(
            f"    assert gen_kwargs.get('max_new_tokens', 100) == {decoding['max_new_tokens']}, "
            f"'max_new_tokens mismatch vs frozen config'"
        )
    if "do_sample" in decoding:
        checks.append(
            f"    assert gen_kwargs.get('do_sample', False) == {decoding['do_sample']}, "
            f"'do_sample mismatch vs frozen config'"
        )

    if not checks:
        return ""

    return (
        "    # === EVAL CONFIG: verify decoding params match frozen config (injected) ===\n"
        + "\n".join(checks)
        + "\n    # === END eval config assertion ===\n"
    )

#!/usr/bin/env python3
"""Tests for centralized model name usage.

All Python runners must use DEFAULT_MODEL from model_registry
instead of hardcoding "meta-llama/Meta-Llama-3-8B-Instruct".

Run with: uv run pytest tests/test_model_names.py -v
"""
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src" / "util"))

RUNNER_DIR = PROJECT_ROOT / "src" / "runners"
POLYKERNEL_DIR = PROJECT_ROOT / "src" / "polykernel"


def _get_python_runner_files():
    files = []
    for d in [RUNNER_DIR, POLYKERNEL_DIR]:
        if d.exists():
            files.extend(f for f in d.glob("*.py") if not f.name.startswith("__"))
    return files


class TestNoHardcodedModelNames:
    """No Python runner should hardcode model name strings."""

    @pytest.mark.parametrize("filepath", _get_python_runner_files(),
                             ids=lambda f: f.relative_to(PROJECT_ROOT).as_posix())
    def test_no_hardcoded_llama_string(self, filepath):
        source = filepath.read_text()
        assert '"meta-llama/Meta-Llama-3-8B-Instruct"' not in source, (
            f"{filepath.name} has hardcoded model name — use DEFAULT_MODEL from model_registry"
        )

    @pytest.mark.parametrize("filepath", _get_python_runner_files(),
                             ids=lambda f: f.relative_to(PROJECT_ROOT).as_posix())
    def test_no_nousresearch_reference(self, filepath):
        source = filepath.read_text()
        assert "NousResearch" not in source, (
            f"{filepath.name} references NousResearch — canonical name doesn't use it"
        )


class TestDefaultModelConsistency:
    """DEFAULT_MODEL must be consistent across the codebase."""

    def test_default_model_exists(self):
        from model_registry import DEFAULT_MODEL
        assert DEFAULT_MODEL == "meta-llama/Meta-Llama-3-8B-Instruct"

    def test_default_model_matches_llama_spec(self):
        from model_registry import DEFAULT_MODEL, LLAMA3_8B
        assert DEFAULT_MODEL == LLAMA3_8B.hf_repo

    @pytest.mark.parametrize("filepath", _get_python_runner_files(),
                             ids=lambda f: f.relative_to(PROJECT_ROOT).as_posix())
    def test_runners_import_default_model(self, filepath):
        """Runners that use a model default should import DEFAULT_MODEL."""
        source = filepath.read_text()
        if "MODEL_NAME" in source and "default=" in source:
            assert "DEFAULT_MODEL" in source or "model_registry" in source, (
                f"{filepath.name} has a model default but doesn't import DEFAULT_MODEL"
            )

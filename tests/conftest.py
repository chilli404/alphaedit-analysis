"""Shared test fixtures for alphaedit-analysis test suite."""
import os
import sys
import types
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
VENDOR_ROOT = PROJECT_ROOT / "vendor" / "AlphaEdit"
BASELINES_ROOT = PROJECT_ROOT / "baselines" / "EvoEdit"

# pythonpath = ["src", "src/util"] is set in pyproject.toml [tool.pytest.ini_options]

# Disable S3 path guard during tests (pytest may run on SkyPilot clusters)
os.environ["_PYTEST_RUNNING"] = "1"


@pytest.fixture
def mock_gpu_imports(monkeypatch):
    """Mock GPU-only imports so runners can be imported without CUDA."""
    for mod_name in ("model_resolve", "setup_hparams", "eval_config"):
        if mod_name not in sys.modules:
            m = types.ModuleType(mod_name)
            if mod_name == "model_resolve":
                m.resolve_model_path = lambda x: x
            elif mod_name == "setup_hparams":
                m.link_hparams = lambda: None
            elif mod_name == "eval_config":
                m.hash_eval_config = lambda: "test_hash"
            monkeypatch.setitem(sys.modules, mod_name, m)


@pytest.fixture
def tmp_checkpoint_root(tmp_path, monkeypatch):
    """Set CHECKPOINT_ROOT to a temp directory."""
    ckpt = tmp_path / "checkpoints"
    ckpt.mkdir()
    monkeypatch.setenv("CHECKPOINT_ROOT", str(ckpt))
    return ckpt


@pytest.fixture
def tmp_result_root(tmp_path, monkeypatch):
    """Set RESULT_ROOT to a temp directory."""
    res = tmp_path / "results"
    res.mkdir()
    monkeypatch.setenv("RESULT_ROOT", str(res))
    return res

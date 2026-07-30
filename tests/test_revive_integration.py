"""
Integration tests for REVIVE in the polykernel_seqreg_runner.

Tests script generation and compilation (no GPU required).
Run with: uv run python -m pytest tests/test_revive_integration.py -v
"""

import ast
import sys
import types
from pathlib import Path

import pytest

# Add src paths
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src" / "util"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src" / "polykernel"))

# Mock GPU-only imports
mock_model_download = types.ModuleType("model_download")
mock_model_download.resolve_model_path = lambda x: x
sys.modules["model_download"] = mock_model_download

mock_setup_hparams = types.ModuleType("setup_hparams")
mock_setup_hparams.link_hparams = lambda: None
sys.modules["setup_hparams"] = mock_setup_hparams

mock_source_patches = types.ModuleType("source_patches")
mock_source_patches.patch_evaluate_file = lambda x: None
sys.modules["source_patches"] = mock_source_patches

mock_eval_config = types.ModuleType("eval_config")
mock_eval_config.hash_eval_config = lambda: "mock_hash"
sys.modules["eval_config"] = mock_eval_config

mock_paths = types.ModuleType("paths")
mock_paths.get_project_root = lambda: Path("/tmp/test_project")
mock_paths.get_alphaedit_root = lambda: Path("/tmp/test_project/vendor/AlphaEdit")
mock_paths.get_result_root = lambda: Path("/tmp/test_results")
mock_paths.get_checkpoint_root = lambda: Path("/tmp/test_checkpoints")
sys.modules["paths"] = mock_paths

from polykernel_seqreg_runner import build_polykernel_seqreg_script, resolve_checkpoint_dir


class TestScriptGeneration:
    """Test that generated scripts are valid Python."""

    def _build_default(self, **kwargs):
        defaults = {
            "seed": 42,
            "cuda_device": "0",
            "alg_name": "MEMIT",
            "model_name": "test-model",
            "hparams_fname": "Llama3-8B.json",
            "ds_name": "mcf",
            "dataset_size_limit": 100,
            "num_edits": 100,
            "downstream_eval_steps": 0,
            "conserve_memory": True,
            "lambda_prev": 1.0,
            "lambda_delta": 0.0,
            "cache_strategy": "all",
            "cache_max": None,
            "kernel_type": "poly",
            "kernel_degree": 2,
            "kernel_sigma": "median",
            "output_jsonl": "/tmp/test.jsonl",
            "fast_checkpoint": False,
            "eval_at_checkpoints_only": False,
            "order_id": 0,
            "save_interval": 10,
            "checkpoint_dir": "/tmp/test_ckpt",
            "start_from_batch": 0,
            "dataset_override": None,
            "eval_results_dir": "/tmp/test_results",
            "variant_name": "test-variant",
            "kernel_prev": False,
            "revive": False,
            "revive_tau": 0.2,
            "revive_svd_device": "cpu",
            "revive_svd_dtype": "float32",
            "revive_cache_dir": "",
            "revive_log_interval": 1,
            "revive_mode": "hard",
        }
        defaults.update(kwargs)
        return build_polykernel_seqreg_script(**defaults)

    def test_script_compiles_without_revive(self):
        """Generated script without REVIVE is valid Python."""
        script = self._build_default(revive=False)
        ast.parse(script)

    def test_script_compiles_with_revive(self):
        """Generated script with REVIVE enabled is valid Python."""
        script = self._build_default(revive=True, revive_tau=0.2)
        ast.parse(script)

    def test_revive_functions_present(self):
        """When REVIVE enabled, script contains init and apply functions."""
        script = self._build_default(revive=True)
        assert "_revive_init" in script
        assert "_revive_apply" in script
        assert "_revive_enabled = True" in script

    def test_revive_functions_absent_when_disabled(self):
        """When REVIVE disabled, _revive_enabled is False."""
        script = self._build_default(revive=False)
        assert "_revive_enabled = False" in script

    def test_revive_injection_present(self):
        """When REVIVE enabled, WEIGHT_UPDATE_ANCHOR injection exists."""
        script = self._build_default(revive=True)
        assert "REVIVE: filter update through pretrained spectral subspace" in script

    def test_revive_tau_interpolated(self):
        """tau value is correctly interpolated into script."""
        script = self._build_default(revive=True, revive_tau=0.35)
        assert "_revive_tau = 0.35" in script

    def test_revive_svd_device_interpolated(self):
        """SVD device is correctly interpolated."""
        script = self._build_default(revive=True, revive_svd_device="cuda")
        assert '_revive_svd_device = "cuda"' in script

    def test_different_tau_values_compile(self):
        """Sweep of tau values all produce valid scripts."""
        for tau in [0.05, 0.10, 0.20, 0.30, 0.40]:
            script = self._build_default(revive=True, revive_tau=tau)
            ast.parse(script)


class TestVariantNaming:
    """Test that REVIVE produces correct variant names."""

    def test_hybrid_without_revive(self):
        """Without REVIVE: MEMIT-Seq-poly2-hybrid-lp1.0-ld0.0-cache0"""
        path = resolve_checkpoint_dir(
            None, 42, 1.0, 0.0,
            cache_max=None, kernel_type="poly", kernel_degree=2,
            kernel_prev=False, revive=False,
        )
        assert "poly2-hybrid" in str(path)
        assert "REVIVE" not in str(path)

    def test_hybrid_with_revive(self):
        """With REVIVE: MEMIT-Seq-poly2-hybrid-REVIVE-tau0.2-lp1.0-ld0.0-cache0"""
        path = resolve_checkpoint_dir(
            None, 42, 1.0, 0.0,
            cache_max=None, kernel_type="poly", kernel_degree=2,
            kernel_prev=False, revive=True, revive_tau=0.2,
        )
        assert "poly2-hybrid-REVIVE-tau0.2" in str(path)

    def test_different_tau_different_path(self):
        """Different tau values produce different checkpoint paths."""
        path1 = resolve_checkpoint_dir(
            None, 42, 1.0, 0.0,
            cache_max=None, kernel_type="poly", kernel_degree=2,
            kernel_prev=False, revive=True, revive_tau=0.2,
        )
        path2 = resolve_checkpoint_dir(
            None, 42, 1.0, 0.0,
            cache_max=None, kernel_type="poly", kernel_degree=2,
            kernel_prev=False, revive=True, revive_tau=0.3,
        )
        assert path1 != path2
        assert "tau0.2" in str(path1)
        assert "tau0.3" in str(path2)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

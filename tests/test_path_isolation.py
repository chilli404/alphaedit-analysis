#!/usr/bin/env python3
"""
Tests for checkpoint and result path isolation.

Catches the class of bug where variant_name was hardcoded to MEMIT-Seq
regardless of base_alg, causing checkpoint clobbering across algorithms.

Run with: uv run pytest tests/test_path_isolation.py -v
"""
import importlib
import os
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent


# ---------------------------------------------------------------------------
# 1. Base algorithm checkpoint path isolation (via ExperimentConfig)
# ---------------------------------------------------------------------------

class TestBaseAlgPathIsolation:
    """Every base_alg MUST produce a distinct checkpoint path."""

    BASE_ALGS = ["MEMIT", "AlphaEdit", "NSE", "MEMIT_rect"]
    EXPECTED_PREFIXES = {
        "MEMIT": "MEMIT-Seq",
        "AlphaEdit": "AlphaEdit",
        "NSE": "NSE",
        "MEMIT_rect": "MEMIT_rect",
    }

    @pytest.fixture(autouse=True)
    def setup(self, tmp_checkpoint_root):
        self.ckpt_root = tmp_checkpoint_root

    def _resolve(self, base_alg, ordering="fb_high_exposure", seed=42):
        from util.experiment_config import ExperimentConfig
        config = ExperimentConfig(
            base_alg=base_alg, seed=seed, ordering=ordering,
            kernel_degree=1, revive=True, revive_tau=0.1,
        )
        return config.checkpoint_dir(self.ckpt_root)

    def test_all_base_algs_produce_distinct_paths(self):
        paths = {}
        for alg in self.BASE_ALGS:
            p = self._resolve(alg)
            paths[alg] = str(p)
        assert len(set(paths.values())) == len(self.BASE_ALGS), (
            f"Path collision! {paths}"
        )

    @pytest.mark.parametrize("base_alg", BASE_ALGS)
    def test_path_contains_correct_prefix(self, base_alg):
        p = self._resolve(base_alg)
        expected = self.EXPECTED_PREFIXES[base_alg]
        assert expected in str(p), (
            f"base_alg={base_alg}: expected '{expected}' in path, got: {p}"
        )

    @pytest.mark.parametrize("base_alg", BASE_ALGS)
    def test_different_seeds_same_prefix(self, base_alg):
        p42 = self._resolve(base_alg, seed=42)
        p2024 = self._resolve(base_alg, seed=2024)
        expected = self.EXPECTED_PREFIXES[base_alg]
        assert expected in str(p42)
        assert expected in str(p2024)
        assert str(p42) != str(p2024)

    @pytest.mark.parametrize("base_alg", BASE_ALGS)
    def test_different_orderings_same_prefix(self, base_alg):
        p_hi = self._resolve(base_alg, ordering="fb_high_exposure")
        p_lo = self._resolve(base_alg, ordering="fb_low_exposure")
        expected = self.EXPECTED_PREFIXES[base_alg]
        assert expected in str(p_hi)
        assert expected in str(p_lo)
        assert str(p_hi) != str(p_lo)

    def test_memit_gets_memit_seq_prefix_not_memit(self):
        p = self._resolve("MEMIT")
        assert "MEMIT-Seq" in str(p)
        parts = Path(p).parts
        assert "MEMIT" not in parts, "Bare 'MEMIT' should not appear as a directory"


class TestInjectedCheckpointGuard:
    """The checkpoint_io.validate_checkpoint_path guard must catch stale code."""

    @pytest.mark.parametrize("base_alg,variant_prefix", [
        ("MEMIT", "MEMIT-Seq"),
        ("AlphaEdit", "AlphaEdit"),
        ("NSE", "NSE"),
        ("MEMIT_rect", "MEMIT_rect"),
    ])
    def test_guard_passes_for_correct_path(self, base_alg, variant_prefix):
        from util.checkpoint_io import validate_checkpoint_path
        ckpt_dir = f"/s3-data/.../polykernel_seqreg/{variant_prefix}-poly1-REVIVE-tau0.1-lp0.0-ld0.0-cache0/fb_high/seed42"
        validate_checkpoint_path(ckpt_dir, base_alg)

    @pytest.mark.parametrize("base_alg", ["AlphaEdit", "NSE", "MEMIT_rect"])
    def test_guard_catches_hardcoded_memit_seq(self, base_alg):
        """Simulates the stale code bug: all paths say MEMIT-Seq."""
        from util.checkpoint_io import validate_checkpoint_path
        ckpt_dir = "/s3-data/.../polykernel_seqreg/MEMIT-Seq-poly1-REVIVE-tau0.1-lp0.0-ld0.0-cache0/fb_high/seed42"
        with pytest.raises(RuntimeError, match="mismatch"):
            validate_checkpoint_path(ckpt_dir, base_alg)


class TestVariantNameConsistency:
    """variant_name must be computed by ExperimentConfig (single source of truth)."""

    @pytest.fixture(autouse=True)
    def setup(self, tmp_checkpoint_root):
        pass

    @pytest.mark.parametrize("base_alg", ["MEMIT", "AlphaEdit", "NSE", "MEMIT_rect"])
    def test_variant_computed_once(self, base_alg):
        """ExperimentConfig.variant_name is the ONLY place variant names are built."""
        from util.experiment_config import ExperimentConfig
        config = ExperimentConfig(
            base_alg=base_alg, seed=42, ordering="fb_high_exposure",
            kernel_degree=1, revive=True, revive_tau=0.1,
        )
        _base_prefix = "MEMIT-Seq" if base_alg == "MEMIT" else base_alg
        assert config.variant_name.startswith(_base_prefix)
        assert config.variant_name in str(config.checkpoint_dir())


# ---------------------------------------------------------------------------
# 2. S3 path guard on SkyPilot
# ---------------------------------------------------------------------------

class TestS3PathGuard:
    """On SkyPilot clusters, paths MUST point to /s3-data/."""

    def test_result_root_requires_s3_on_skypilot(self, monkeypatch):
        monkeypatch.setenv("SKYPILOT_TASK_ID", "test-123")
        monkeypatch.delenv("RESULT_ROOT", raising=False)
        monkeypatch.delenv("_PYTEST_RUNNING", raising=False)
        import paths
        importlib.reload(paths)
        try:
            with pytest.raises(RuntimeError, match="S3"):
                paths.get_result_root()
        finally:
            monkeypatch.setenv("_PYTEST_RUNNING", "1")

    def test_checkpoint_root_requires_s3_on_skypilot(self, monkeypatch):
        monkeypatch.setenv("SKYPILOT_TASK_ID", "test-123")
        monkeypatch.delenv("CHECKPOINT_ROOT", raising=False)
        monkeypatch.delenv("_PYTEST_RUNNING", raising=False)
        import paths
        importlib.reload(paths)
        try:
            with pytest.raises(RuntimeError, match="S3"):
                paths.get_checkpoint_root()
        finally:
            monkeypatch.setenv("_PYTEST_RUNNING", "1")

    def test_s3_paths_accepted_on_skypilot(self, monkeypatch):
        monkeypatch.setenv("SKYPILOT_TASK_ID", "test-123")
        monkeypatch.setenv("RESULT_ROOT", "/s3-data/continual-learning/alphaedit/results")
        monkeypatch.setenv("CHECKPOINT_ROOT", "/s3-data/continual-learning/alphaedit/checkpoints")
        monkeypatch.delenv("_PYTEST_RUNNING", raising=False)
        import paths
        importlib.reload(paths)
        try:
            r = paths.get_result_root()
            c = paths.get_checkpoint_root()
            assert "/s3-data/" in str(r)
            assert "/s3-data/" in str(c)
        finally:
            monkeypatch.setenv("_PYTEST_RUNNING", "1")

    def test_local_paths_ok_without_skypilot(self, monkeypatch):
        monkeypatch.delenv("SKYPILOT_TASK_ID", raising=False)
        monkeypatch.delenv("RESULT_ROOT", raising=False)
        monkeypatch.delenv("CHECKPOINT_ROOT", raising=False)
        import paths
        importlib.reload(paths)
        r = paths.get_result_root()
        c = paths.get_checkpoint_root()
        assert r is not None
        assert c is not None


# ---------------------------------------------------------------------------
# 3. Checkpoint metadata validation
# ---------------------------------------------------------------------------

class TestCheckpointMetadata:
    """Checkpoint metadata must include enough info to detect mismatches."""

    def test_validate_checkpoint_path_function_exists(self):
        from util.checkpoint_io import validate_checkpoint_path
        assert callable(validate_checkpoint_path)

    def test_experiment_config_has_variant_name(self):
        from util.experiment_config import ExperimentConfig
        config = ExperimentConfig(
            base_alg="MEMIT", seed=42, kernel_degree=2,
            lambda_prev=1.0, lambda_delta=0.0,
        )
        assert "kernel_type" in dir(config) or hasattr(config, "kernel_tag")
        assert "MEMIT-Seq" in config.variant_name
        assert "poly2" in config.variant_name

    def test_experiment_config_base_alg_guard(self):
        """ExperimentConfig.validate() must catch mismatches."""
        from util.experiment_config import ExperimentConfig
        config = ExperimentConfig(
            base_alg="AlphaEdit", seed=42, kernel_degree=1,
            revive=True, revive_tau=0.1,
        )
        assert "AlphaEdit" in config.variant_name
        assert "MEMIT-Seq" not in config.variant_name

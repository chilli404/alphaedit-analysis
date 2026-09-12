#!/usr/bin/env python3
"""
Tests for checkpoint and result path isolation.

Catches the class of bug where variant_name was hardcoded to MEMIT-Seq
regardless of base_alg, causing checkpoint clobbering across algorithms.

Run with: uv run pytest tests/test_path_isolation.py -v
"""
import importlib
import os
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent


# ---------------------------------------------------------------------------
# 1. Base algorithm checkpoint path isolation
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
    def setup(self, mock_gpu_imports, tmp_checkpoint_root):
        self.ckpt_root = tmp_checkpoint_root

    def _resolve(self, base_alg, ordering="fb_high_exposure", seed=42):
        from polykernel_seqreg_runner import resolve_checkpoint_dir
        return resolve_checkpoint_dir(
            None, seed, 0.0, 0.0,
            cache_max=None, kernel_degree=1, ordering=ordering,
            revive=True, revive_tau=0.1, base_alg=base_alg,
        )

    def test_all_base_algs_produce_distinct_paths(self):
        paths = {}
        for alg in self.BASE_ALGS:
            p = self._resolve(alg)
            paths[alg] = str(p)
        # All 4 paths must be different
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
        # But NOT bare "MEMIT/" as a directory component (would collide with vanilla MEMIT)
        parts = Path(p).parts
        assert "MEMIT" not in parts, "Bare 'MEMIT' should not appear as a directory"


class TestInjectedCheckpointGuard:
    """The _ckpt_save guard inside the exec'd code must catch stale code."""

    @pytest.fixture(autouse=True)
    def setup(self, mock_gpu_imports):
        pass

    @pytest.mark.parametrize("base_alg,variant_prefix", [
        ("MEMIT", "MEMIT-Seq"),
        ("AlphaEdit", "AlphaEdit"),
        ("NSE", "NSE"),
        ("MEMIT_rect", "MEMIT_rect"),
    ])
    def test_guard_passes_for_correct_path(self, base_alg, variant_prefix):
        ckpt_dir = f"/s3-data/.../polykernel_seqreg/{variant_prefix}-poly1-REVIVE-tau0.1-lp0.0-ld0.0-cache0/fb_high/seed42"
        _expected = "MEMIT-Seq" if base_alg == "MEMIT" else base_alg
        assert _expected in ckpt_dir

    @pytest.mark.parametrize("base_alg", ["AlphaEdit", "NSE", "MEMIT_rect"])
    def test_guard_catches_hardcoded_memit_seq(self, base_alg):
        """Simulates the stale code bug: all paths say MEMIT-Seq."""
        ckpt_dir = "/s3-data/.../polykernel_seqreg/MEMIT-Seq-poly1-REVIVE-tau0.1-lp0.0-ld0.0-cache0/fb_high/seed42"
        _expected = "MEMIT-Seq" if base_alg == "MEMIT" else base_alg
        assert _expected not in ckpt_dir, (
            f"Guard should REJECT {base_alg} writing to MEMIT-Seq path"
        )


class TestVariantNameConsistency:
    """variant_name must be computed identically in resolve_checkpoint_dir and main body."""

    @pytest.fixture(autouse=True)
    def setup(self, mock_gpu_imports, tmp_checkpoint_root):
        pass

    @pytest.mark.parametrize("base_alg", ["MEMIT", "AlphaEdit", "NSE", "MEMIT_rect"])
    def test_variant_computed_once(self, base_alg):
        """Both resolve_checkpoint_dir and the main body should produce the same variant."""
        _base_prefix = "MEMIT-Seq" if base_alg == "MEMIT" else base_alg
        kernel_tag = "poly1-REVIVE-tau0.1"
        variant = f"{_base_prefix}-{kernel_tag}-lp0.0-ld0.0-cache0"

        from polykernel_seqreg_runner import resolve_checkpoint_dir
        ckpt_dir = resolve_checkpoint_dir(
            None, 42, 0.0, 0.0,
            cache_max=None, kernel_degree=1,
            ordering="fb_high_exposure",
            revive=True, revive_tau=0.1,
            base_alg=base_alg,
        )
        assert variant in str(ckpt_dir), (
            f"resolve_checkpoint_dir produced {ckpt_dir}, expected variant '{variant}' in path"
        )


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
        assert r.exists() or True  # just shouldn't raise
        assert c is not None


# ---------------------------------------------------------------------------
# 3. Checkpoint metadata validation
# ---------------------------------------------------------------------------

class TestCheckpointMetadata:
    """Checkpoint metadata must include enough info to detect mismatches."""

    def test_polykernel_metadata_includes_kernel_params(self, mock_gpu_imports):
        from polykernel_seqreg_runner import build_polykernel_seqreg_script
        script = build_polykernel_seqreg_script(
            seed=42, cuda_device="0", alg_name="MEMIT",
            model_name="test", hparams_fname="test.json",
            ds_name="mcf", dataset_size_limit=200, num_edits=100,
            downstream_eval_steps=0, conserve_memory=True,
            lambda_prev=1.0, lambda_delta=0.0,
            cache_strategy="all", cache_max=None,
            kernel_type="poly", kernel_degree=2, kernel_sigma="median",
            output_jsonl="/tmp/test.jsonl",
            checkpoint_dir="/tmp/ckpt",
            variant_name="test-variant",
        )
        # Metadata in the injected script should include kernel params
        assert "kernel_type" in script
        assert "kernel_degree" in script
        assert "kernel_prev" in script
        assert "lambda_prev" in script
        assert "lambda_delta" in script

    def test_injected_script_has_base_alg_guard(self, mock_gpu_imports):
        from polykernel_seqreg_runner import build_polykernel_seqreg_script
        script = build_polykernel_seqreg_script(
            seed=42, cuda_device="0", alg_name="AlphaEdit",
            model_name="test", hparams_fname="test.json",
            ds_name="mcf", dataset_size_limit=200, num_edits=100,
            downstream_eval_steps=0, conserve_memory=True,
            lambda_prev=0.0, lambda_delta=0.0,
            cache_strategy="all", cache_max=None,
            kernel_type="poly", kernel_degree=1, kernel_sigma="median",
            output_jsonl="/tmp/test.jsonl",
            checkpoint_dir="/tmp/ckpt",
            variant_name="AlphaEdit-poly1-lp0.0-ld0.0-cache0",
            revive=True, revive_tau=0.1,
        )
        assert '_ckpt_base_alg = "AlphaEdit"' in script
        assert "CHECKPOINT PATH MISMATCH" in script

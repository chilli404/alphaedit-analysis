#!/usr/bin/env python3
"""Tests that ExperimentConfig.checkpoint_dir() produces identical paths
to the 4 old resolve_checkpoint_dir implementations.

Run with: uv run pytest tests/test_checkpoint_dir_dedup.py -v
"""
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent


class TestPolyKernelSeqregPaths:
    """ExperimentConfig must produce the same paths as polykernel_seqreg_runner."""

    @pytest.fixture(autouse=True)
    def setup(self, mock_gpu_imports, tmp_checkpoint_root):
        self.root = tmp_checkpoint_root

    @pytest.mark.parametrize("base_alg", ["MEMIT", "AlphaEdit", "NSE", "MEMIT_rect"])
    def test_basic_path(self, base_alg):
        from experiment_config import ExperimentConfig
        cfg = ExperimentConfig(
            base_alg=base_alg, seed=42,
            lambda_prev=1.0, lambda_delta=0.0,
            kernel_degree=2, cache_max=None,
            experiment_type="polykernel_seqreg",
        )
        path = cfg.checkpoint_dir(self.root)
        prefix = "MEMIT-Seq" if base_alg == "MEMIT" else base_alg
        assert prefix in str(path)
        assert "polykernel_seqreg" in str(path)
        assert "seed42" in str(path)

    def test_with_ordering(self):
        from experiment_config import ExperimentConfig
        cfg = ExperimentConfig(
            base_alg="MEMIT", seed=2024,
            ordering="fb_high_exposure",
            kernel_degree=1, lambda_prev=0.0, lambda_delta=0.0,
            experiment_type="polykernel_seqreg",
        )
        path = cfg.checkpoint_dir(self.root)
        assert "fb_high_exposure" in str(path)
        assert "seed2024" in str(path)

    def test_with_revive(self):
        from experiment_config import ExperimentConfig
        cfg = ExperimentConfig(
            base_alg="AlphaEdit", seed=42,
            kernel_degree=1, revive=True, revive_tau=0.1,
            lambda_prev=0.0, lambda_delta=0.0,
            experiment_type="polykernel_seqreg",
        )
        path = cfg.checkpoint_dir(self.root)
        assert "AlphaEdit-poly1-REVIVE-tau0.1" in str(path)

    def test_gptj_model_tag(self):
        from experiment_config import ExperimentConfig
        cfg = ExperimentConfig(
            base_alg="MEMIT", seed=42,
            model_name="EleutherAI/gpt-j-6b",
            kernel_degree=2,
            experiment_type="polykernel_seqreg",
        )
        path = cfg.checkpoint_dir(self.root)
        assert "gpt-j-6b" in str(path)

    def test_hybrid_kernel(self):
        from experiment_config import ExperimentConfig
        cfg = ExperimentConfig(
            base_alg="MEMIT", seed=42,
            kernel_degree=2, kernel_prev=False,
            lambda_prev=1.0, lambda_delta=0.0,
            experiment_type="polykernel_seqreg",
        )
        path = cfg.checkpoint_dir(self.root)
        assert "poly2-hybrid" in str(path)


class TestCheckpointRunnerPaths:
    """ExperimentConfig must produce the same paths as checkpoint_runner."""

    @pytest.fixture(autouse=True)
    def setup(self, mock_gpu_imports, tmp_checkpoint_root):
        self.root = tmp_checkpoint_root

    def test_alphaedit_standard(self):
        from experiment_config import ExperimentConfig
        cfg = ExperimentConfig(
            base_alg="AlphaEdit", seed=42,
            experiment_type="failure_curve",
        )
        path = cfg.checkpoint_dir(self.root)
        assert str(path) == str(self.root / "failure_curve" / "AlphaEdit" / "seed42")

    def test_memit_standard(self):
        from experiment_config import ExperimentConfig
        cfg = ExperimentConfig(
            base_alg="MEMIT", seed=137,
            experiment_type="failure_curve",
        )
        path = cfg.checkpoint_dir(self.root)
        assert str(path) == str(self.root / "failure_curve" / "MEMIT" / "seed137")

    def test_comparison_ordered(self):
        from experiment_config import ExperimentConfig
        cfg = ExperimentConfig(
            base_alg="AlphaEdit", seed=42, order_id=3,
            experiment_type="comparison_ordered",
        )
        path = cfg.checkpoint_dir(self.root)
        assert str(path) == str(self.root / "comparison_ordered" / "AlphaEdit" / "seed42" / "order3")

    def test_gptj_failure_curve(self):
        from experiment_config import ExperimentConfig
        cfg = ExperimentConfig(
            base_alg="AlphaEdit", seed=42,
            model_name="EleutherAI/gpt-j-6b",
            experiment_type="failure_curve",
        )
        path = cfg.checkpoint_dir(self.root)
        assert "gpt-j-6b" in str(path)
        assert "failure_curve" in str(path)


class TestMemitSeqPaths:
    """ExperimentConfig must produce the same paths as memit_sequential_runner."""

    @pytest.fixture(autouse=True)
    def setup(self, mock_gpu_imports, tmp_checkpoint_root):
        self.root = tmp_checkpoint_root

    def test_standard_memit_seq(self):
        from experiment_config import ExperimentConfig
        cfg = ExperimentConfig(
            base_alg="MEMIT", seed=42,
            lambda_prev=1.0, lambda_delta=0.0, cache_max=None,
            experiment_type="failure_curve",
        )
        path = cfg.checkpoint_dir(self.root)
        expected = self.root / "failure_curve" / "MEMIT-Seq-lp1.0-ld0.0-cache0" / "seed42"
        assert str(path) == str(expected)

    def test_with_ordering(self):
        from experiment_config import ExperimentConfig
        cfg = ExperimentConfig(
            base_alg="MEMIT", seed=42,
            lambda_prev=1.0, lambda_delta=0.0, cache_max=None,
            ordering="key_clustered",
            experiment_type="failure_curve",
        )
        path = cfg.checkpoint_dir(self.root)
        assert "matched_ordering" in str(path)
        assert "key_clustered" in str(path)
        assert "MEMIT-Seq-lp1.0-ld0.0-cache0" in str(path)

    def test_with_mom2_override(self):
        from experiment_config import ExperimentConfig
        cfg = ExperimentConfig(
            base_alg="MEMIT", seed=42,
            lambda_prev=1.0, lambda_delta=0.0, cache_max=None,
            mom2_override=15000.0,
            experiment_type="failure_curve",
        )
        path = cfg.checkpoint_dir(self.root)
        assert "-a15000.0" in str(path)


class TestNoDuplicateImplementations:
    """After dedup, only experiment_config.py should define core path logic."""

    def test_experiment_config_is_single_source(self):
        """ExperimentConfig must exist and have checkpoint_dir method."""
        from experiment_config import ExperimentConfig
        cfg = ExperimentConfig(base_alg="MEMIT", seed=42)
        assert hasattr(cfg, "checkpoint_dir")
        assert hasattr(cfg, "variant_name")
        assert hasattr(cfg, "model_tag")

    def test_all_experiment_types_supported(self):
        from experiment_config import ExperimentConfig
        for exp_type in ["polykernel_seqreg", "failure_curve", "comparison_ordered"]:
            cfg = ExperimentConfig(
                base_alg="MEMIT", seed=42,
                experiment_type=exp_type,
                order_id=1 if exp_type == "comparison_ordered" else 0,
            )
            path = cfg.checkpoint_dir()
            assert path is not None

    def test_invalid_experiment_type_raises(self):
        from experiment_config import ExperimentConfig
        cfg = ExperimentConfig(
            base_alg="MEMIT", seed=42,
            experiment_type="invalid_type",
        )
        with pytest.raises(ValueError):
            cfg.checkpoint_dir()


class TestPathIsolation:
    """Different experiment types must produce distinct path structures."""

    @pytest.fixture(autouse=True)
    def setup(self, mock_gpu_imports, tmp_checkpoint_root):
        self.root = tmp_checkpoint_root

    def test_polykernel_and_failure_curve_are_distinct(self):
        from experiment_config import ExperimentConfig
        pk = ExperimentConfig(
            base_alg="MEMIT", seed=42,
            experiment_type="polykernel_seqreg",
        ).checkpoint_dir(self.root)
        fc = ExperimentConfig(
            base_alg="MEMIT", seed=42,
            experiment_type="failure_curve",
        ).checkpoint_dir(self.root)
        assert str(pk) != str(fc)
        assert "polykernel_seqreg" in str(pk)
        assert "failure_curve" in str(fc)

    def test_failure_curve_alphaedit_vs_memit_seq(self):
        """AlphaEdit uses alg name directly; MEMIT-Seq uses variant name with params."""
        from experiment_config import ExperimentConfig
        ae = ExperimentConfig(
            base_alg="AlphaEdit", seed=42,
            experiment_type="failure_curve",
        ).checkpoint_dir(self.root)
        ms = ExperimentConfig(
            base_alg="MEMIT", seed=42,
            lambda_prev=1.0, lambda_delta=0.0,
            experiment_type="failure_curve",
        ).checkpoint_dir(self.root)
        assert "AlphaEdit" in str(ae)
        assert "MEMIT-Seq-lp1.0" in str(ms)
        assert str(ae) != str(ms)

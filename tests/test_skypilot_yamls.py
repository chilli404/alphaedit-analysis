#!/usr/bin/env python3
"""Tests for the consolidated SkyPilot infrastructure.

3 files replace the previous 24:
  sky/run.yaml    — generic experiment runner (any method via --env METHOD=X)
  sky/test.yaml   — CPU + GPU test runner (filter via --env SMOKE_FILTER=X)
  sky/launch.sh   — smart launcher script

Run with: uv run pytest tests/test_skypilot_yamls.py -v
"""
import os
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SKY_DIR = PROJECT_ROOT / "sky"


class TestSkyFilesExist:
    def test_run_yaml(self):
        assert (SKY_DIR / "run.yaml").exists()

    def test_test_yaml(self):
        assert (SKY_DIR / "test.yaml").exists()

    def test_launch_sh(self):
        assert (SKY_DIR / "launch.sh").exists()

    def test_launch_sh_executable(self):
        assert os.access(SKY_DIR / "launch.sh", os.X_OK)

    def test_core_files_present(self):
        """Core files: run.yaml (experiments), test.yaml (tests), launch.sh (orchestrator)."""
        for f in ["run.yaml", "test.yaml", "launch.sh"]:
            assert (SKY_DIR / f).exists(), f"sky/{f} missing"


class TestRunYaml:
    @pytest.fixture
    def source(self):
        return (SKY_DIR / "run.yaml").read_text()

    def test_uses_method_env(self, source):
        assert "METHOD" in source
        assert "python -m src run" in source

    def test_uses_seed_env(self, source):
        assert "SEED" in source

    def test_supports_ordering(self, source):
        assert "ORDERING" in source

    def test_uses_apply_all(self, source):
        assert "apply_all.py" in source

    def test_links_data(self, source):
        assert "link_stats.sh" in source
        assert "link_dsets.sh" in source

    def test_sets_s3_paths(self, source):
        assert "RESULT_ROOT" in source
        assert "CHECKPOINT_ROOT" in source
        assert "/s3-data/" in source

    def test_has_setup_block(self, source):
        assert "remote_setup.sh" in source

    def test_no_inline_patches(self, source):
        assert "sed -i" not in source

    def test_no_experiment_dispatch(self, source):
        """run.yaml must NOT have the old if/elif experiment dispatch — CLI handles it."""
        assert "EXPERIMENT_NAME" not in source


class TestTestYaml:
    @pytest.fixture
    def source(self):
        return (SKY_DIR / "test.yaml").read_text()

    def test_runs_cpu_tests(self, source):
        assert "pytest" in source

    def test_runs_gpu_smoke_test(self, source):
        assert "test_smoke_all_algorithms.sh" in source

    def test_supports_smoke_filter(self, source):
        assert "SMOKE_FILTER" in source

    def test_supports_skip_gpu(self, source):
        assert "SKIP_GPU" in source

    def test_uses_apply_all(self, source):
        assert "apply_all.py" in source


class TestLaunchSh:
    @pytest.fixture
    def source(self):
        return (SKY_DIR / "launch.sh").read_text()

    def test_supports_test_flag(self, source):
        assert "--test" in source

    def test_supports_list_flag(self, source):
        assert "--list" in source

    def test_supports_multi_seed(self, source):
        assert "SEED_ARRAY" in source or "IFS" in source

    def test_uses_run_yaml(self, source):
        assert "sky/run.yaml" in source

    def test_uses_test_yaml(self, source):
        assert "sky/test.yaml" in source

    def test_supports_ordering(self, source):
        assert "ORDERING" in source


class TestNoOldFiles:
    """Old YAMLs must not exist — replaced by run.yaml + test.yaml."""

    OLD_FILES = [
        "eval_evoedit_anchor.yaml",
        "eval_evoedit.yaml",
        "eval_generic.yaml",
        "eval_multi_ckpt.yaml",
        "eval_probpref.yaml",
        "eval_single_ckpt.yaml",
        "eval_task_zsre.yaml",
        "eval_task.yaml",
        "eval_temporal.yaml",
        "eval_wait_then_run.yaml",
        "gpu_cmd.yaml",
        "logit_damage_memit.yaml",
        "reedit_from_ckpt.yaml",
        "same_fact_stage2.yaml",
        "suffix_eval.yaml",
        "suffix_switch.yaml",
        "sky_launch.sh",
    ]

    @pytest.mark.parametrize("filename", OLD_FILES)
    def test_old_file_removed(self, filename):
        assert not (SKY_DIR / filename).exists(), (
            f"sky/{filename} should be deleted — use sky/run.yaml or sky/test.yaml instead"
        )

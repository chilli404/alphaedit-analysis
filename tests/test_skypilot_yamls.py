#!/usr/bin/env python3
"""Tests for SkyPilot YAML infrastructure.

Verifies the 4 parallel test clusters, the launcher script,
and that all YAMLs follow the clean pattern.

Run with: uv run pytest tests/test_skypilot_yamls.py -v
"""
import os
import stat
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SKY_DIR = PROJECT_ROOT / "sky"


class TestSplitTestYamls:
    """The 4 parallel test cluster YAMLs must exist and be correct."""

    TEST_YAMLS = [
        "test_vendor_runners.yaml",
        "test_revive_runners.yaml",
        "test_baseline_runners.yaml",
        "test_eval_and_measure.yaml",
    ]

    @pytest.mark.parametrize("filename", TEST_YAMLS)
    def test_yaml_exists(self, filename):
        assert (SKY_DIR / filename).exists(), f"Missing: sky/{filename}"

    @pytest.mark.parametrize("filename", TEST_YAMLS)
    def test_yaml_uses_apply_all(self, filename):
        source = (SKY_DIR / filename).read_text()
        assert "apply_all.py" in source, f"{filename} must call apply_all.py"

    @pytest.mark.parametrize("filename", TEST_YAMLS)
    def test_yaml_links_data(self, filename):
        source = (SKY_DIR / filename).read_text()
        assert "link_stats.sh" in source, f"{filename} must call link_stats.sh"
        assert "link_dsets.sh" in source, f"{filename} must call link_dsets.sh"

    @pytest.mark.parametrize("filename", TEST_YAMLS)
    def test_yaml_has_correct_resources(self, filename):
        source = (SKY_DIR / filename).read_text()
        assert "L40s:1" in source, f"{filename} must use L40s:1"
        assert "cpus: 4+" in source, f"{filename} must have 4+ cpus"
        assert "memory: 16+" in source, f"{filename} must have 16+ memory"

    @pytest.mark.parametrize("filename", TEST_YAMLS)
    def test_yaml_runs_cpu_tests(self, filename):
        source = (SKY_DIR / filename).read_text()
        assert "pytest" in source, f"{filename} must run CPU tests"

    @pytest.mark.parametrize("filename", TEST_YAMLS)
    def test_yaml_uses_smoke_filter(self, filename):
        source = (SKY_DIR / filename).read_text()
        assert "test_smoke_all_algorithms.sh" in source, (
            f"{filename} must use the smoke test script"
        )

    def test_vendor_tests_correct_algorithms(self):
        source = (SKY_DIR / "test_vendor_runners.yaml").read_text()
        assert "AlphaEdit" in source
        assert "MEMIT" in source
        assert "PathGuard" in source

    def test_revive_tests_correct_algorithms(self):
        source = (SKY_DIR / "test_revive_runners.yaml").read_text()
        assert "REVIVE" in source

    def test_baseline_tests_correct_algorithms(self):
        source = (SKY_DIR / "test_baseline_runners.yaml").read_text()
        assert "EvoEdit" in source
        assert "NSE" in source
        assert "RECT" in source


class TestLauncher:
    """The test_all.sh launcher must exist and be correct."""

    def test_launcher_exists(self):
        assert (SKY_DIR / "test_all.sh").exists()

    def test_launcher_is_executable(self):
        path = SKY_DIR / "test_all.sh"
        assert os.access(path, os.X_OK), "test_all.sh must be executable"

    def test_launcher_starts_all_4_clusters(self):
        source = (SKY_DIR / "test_all.sh").read_text()
        assert "test-vendor" in source
        assert "test-revive" in source
        assert "test-baseline" in source
        assert "test-eval" in source

    def test_launcher_has_status_command(self):
        source = (SKY_DIR / "test_all.sh").read_text()
        assert "--status" in source

    def test_launcher_has_down_command(self):
        source = (SKY_DIR / "test_all.sh").read_text()
        assert "--down" in source


class TestExperimentYamls:
    """Experiment YAMLs must use apply_all.py and not have inline patches."""

    EXPERIMENT_YAMLS = [
        "alphaedit_gpu.yaml",
        "suffix_switch.yaml",
        "reedit_from_ckpt.yaml",
        "eval_evoedit_anchor.yaml",
        "eval_evoedit.yaml",
        "eval_generic.yaml",
        "eval_probpref.yaml",
        "logit_damage_memit.yaml",
        "same_fact_stage2.yaml",
        "suffix_eval.yaml",
    ]

    @pytest.mark.parametrize("filename", EXPERIMENT_YAMLS)
    def test_no_sed_patches(self, filename):
        path = SKY_DIR / filename
        if not path.exists():
            pytest.skip(f"{filename} not present")
        source = path.read_text()
        assert source.count("sed -i") == 0, (
            f"{filename} has inline sed patches — use scripts/patches/apply_all.py"
        )

    @pytest.mark.parametrize("filename", EXPERIMENT_YAMLS)
    def test_uses_apply_all_if_links_data(self, filename):
        """Any YAML that calls link_stats/link_dsets must also call apply_all."""
        path = SKY_DIR / filename
        if not path.exists():
            pytest.skip(f"{filename} not present")
        source = path.read_text()
        has_link = "link_stats" in source or "link_dsets" in source
        has_apply = "apply_all" in source
        if has_link:
            assert has_apply, (
                f"{filename} calls link_stats/link_dsets but not apply_all.py — "
                f"vendor code won't be patched"
            )

    def test_no_nousresearch_in_yamls(self):
        """No YAML should reference NousResearch — use meta-llama."""
        for f in SKY_DIR.glob("*.yaml"):
            source = f.read_text()
            assert "NousResearch" not in source, (
                f"{f.name} references NousResearch — use meta-llama/Meta-Llama-3-8B-Instruct"
            )


class TestNoInlinePatches:
    """No YAML should have inline sed patches — all patching via apply_all.py."""

    @pytest.mark.parametrize("filename", [
        "test_vendor_runners.yaml",
        "test_revive_runners.yaml",
        "test_baseline_runners.yaml",
        "test_eval_and_measure.yaml",
        "alphaedit_gpu.yaml",
        "smoke_test.yaml",
    ])
    def test_no_sed_patches(self, filename):
        path = SKY_DIR / filename
        if not path.exists():
            pytest.skip(f"{filename} not present")
        source = path.read_text()
        assert source.count("sed -i") == 0, (
            f"{filename} has inline sed patches — use apply_all.py"
        )

    def test_alphaedit_gpu_uses_apply_all(self):
        path = SKY_DIR / "alphaedit_gpu.yaml"
        if not path.exists():
            pytest.skip("alphaedit_gpu.yaml not present")
        source = path.read_text()
        assert "apply_all.py" in source

    def test_alphaedit_gpu_no_individual_file_mounts(self):
        """alphaedit_gpu.yaml should only mount .env, not individual scripts."""
        path = SKY_DIR / "alphaedit_gpu.yaml"
        if not path.exists():
            pytest.skip("alphaedit_gpu.yaml not present")
        source = path.read_text()
        assert "remote_setup.sh:" not in source, (
            "alphaedit_gpu.yaml should not mount individual scripts — workdir syncs everything"
        )

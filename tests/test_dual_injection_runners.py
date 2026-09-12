#!/usr/bin/env python3
"""Tests for runners that still use exec(compile()) for algorithm injection.

After migration, only polykernel_editor_runner still uses the full dual-injection
pattern. memit_sequential, polykernel_seqreg, and pathguard now use evaluate_harness
+ algorithm hooks (tested in test_algorithm_hooks.py and test_evaluate_harness.py).

Run with: uv run pytest tests/test_dual_injection_runners.py -v
"""
import ast
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Only polykernel_editor still uses dual exec(compile())
DUAL_INJECTION_RUNNERS = [
    ("src/polykernel/polykernel_editor_runner.py", "memit_main.py OR AlphaEdit_main.py", "evaluate.py"),
]


class TestDualInjectionStructure:
    """Verify remaining dual-injection runners have correct structure."""

    @pytest.mark.parametrize("runner,algo_file,eval_file", DUAL_INJECTION_RUNNERS)
    def test_parses(self, runner, algo_file, eval_file):
        source = (PROJECT_ROOT / runner).read_text()
        ast.parse(source)

    @pytest.mark.parametrize("runner,algo_file,eval_file", DUAL_INJECTION_RUNNERS)
    def test_has_exec_compile(self, runner, algo_file, eval_file):
        source = (PROJECT_ROOT / runner).read_text()
        assert source.count("exec(compile(") >= 2


class TestMigratedRunners:
    """Runners that moved from exec(compile()) to evaluate_harness + hooks."""

    MIGRATED = [
        "src/runners/checkpoint_runner.py",
        "src/runners/memit_sequential_runner.py",
        "src/runners/pathguard_runner.py",
        "src/polykernel/polykernel_seqreg_runner.py",
    ]

    @pytest.mark.parametrize("runner", MIGRATED)
    def test_parses(self, runner):
        source = (PROJECT_ROOT / runner).read_text()
        ast.parse(source)

    @pytest.mark.parametrize("runner", MIGRATED)
    def test_uses_harness_or_hooks(self, runner):
        source = (PROJECT_ROOT / runner).read_text()
        assert ("run_experiment" in source or
                "evaluate_harness" in source or
                "apply_memit_with_hooks" in source or
                "apply_alphaedit_with_hooks" in source or
                "AlgorithmHooks" in source), (
            f"{runner} should use harness or hooks"
        )

    @pytest.mark.parametrize("runner", MIGRATED)
    def test_uses_shared_checkpoint_io(self, runner):
        source = (PROJECT_ROOT / runner).read_text()
        assert ("checkpoint_io" in source or
                "save_checkpoint" in source or
                "load_checkpoint" in source), (
            f"{runner} should use shared checkpoint_io"
        )


class TestAlgorithmSpecificBehavior:
    """Verify algorithm-specific behavior exists (in runner or hooks)."""

    def test_seqreg_prev_cache_in_hooks(self):
        source = (PROJECT_ROOT / "src/algorithms/hook_presets.py").read_text()
        assert "prev_cache" in source

    def test_polykernel_kernel_in_hooks(self):
        source = (PROJECT_ROOT / "src/algorithms/hook_presets.py").read_text()
        assert "kernel" in source.lower()

    def test_revive_in_hooks(self):
        source = (PROJECT_ROOT / "src/algorithms/hook_presets.py").read_text()
        assert "revive" in source.lower()
        assert "searchsorted" in source

    def test_pathguard_in_runner_or_hooks(self):
        runner = (PROJECT_ROOT / "src/runners/pathguard_runner.py").read_text()
        hooks = (PROJECT_ROOT / "src/algorithms/hook_presets.py").read_text()
        assert "pathguard" in runner.lower() or "pathguard" in hooks.lower()

    def test_polykernel_editor_supports_alphaedit_and_memit(self):
        source = (PROJECT_ROOT / "src/polykernel/polykernel_editor_runner.py").read_text()
        assert "AlphaEdit" in source
        assert "MEMIT" in source

#!/usr/bin/env python3
"""Tests for dual-injection runners.

Originally tested the exec(compile()) pattern for 4 runners. Several of these
runners have been partially or fully migrated to evaluate_harness + algorithm hooks.
Tests that check for the OLD exec(compile()) internals are now skipped.

Run with: uv run pytest tests/test_dual_injection_runners.py -v
"""
import ast
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
VENDOR_ROOT = PROJECT_ROOT / "vendor" / "AlphaEdit"


# Only runners that STILL use dual exec(compile()) — migrated ones are excluded
DUAL_INJECTION_RUNNERS = [
    ("src/polykernel/polykernel_seqreg_runner.py", "memit_main.py", "evaluate.py"),
    ("src/polykernel/polykernel_editor_runner.py", "memit_main.py OR AlphaEdit_main.py", "evaluate.py"),
]

# Migrated runners: memit_sequential_runner now uses hooks + evaluate_harness,
# pathguard_runner uses hooks + modified injection. Tested in test_algorithm_hooks.py.


class TestDualInjectionStructure:
    """Verify dual-injection runners have correct structure."""

    @pytest.mark.parametrize("runner,algo_file,eval_file", DUAL_INJECTION_RUNNERS)
    def test_parses(self, runner, algo_file, eval_file):
        source = (PROJECT_ROOT / runner).read_text()
        ast.parse(source)

    @pytest.mark.parametrize("runner,algo_file,eval_file", DUAL_INJECTION_RUNNERS)
    def test_has_two_exec_compile(self, runner, algo_file, eval_file):
        """Dual injection = 2 exec(compile()) calls."""
        source = (PROJECT_ROOT / runner).read_text()
        exec_count = source.count("exec(compile(")
        assert exec_count == 2, f"{runner} should have exactly 2 exec(compile()), has {exec_count}"

    @pytest.mark.parametrize("runner,algo_file,eval_file", DUAL_INJECTION_RUNNERS)
    def test_first_exec_is_algorithm(self, runner, algo_file, eval_file):
        """First exec must be the algorithm file (memit_main.py, AlphaEdit_main.py, or template var)."""
        source = (PROJECT_ROOT / runner).read_text()
        first_exec_pos = source.find("exec(compile(")
        exec_context = source[first_exec_pos:first_exec_pos + 200]
        assert any(x in exec_context for x in ["memit_main.py", "AlphaEdit_main.py", "algo_file", "_algo_source"]), (
            f"First exec in {runner} should compile algorithm file, got: {exec_context[:100]}"
        )

    @pytest.mark.parametrize("runner,algo_file,eval_file", DUAL_INJECTION_RUNNERS)
    def test_second_exec_is_evaluate(self, runner, algo_file, eval_file):
        """Second exec must be evaluate.py."""
        source = (PROJECT_ROOT / runner).read_text()
        # Find the SECOND exec(compile()
        first = source.find("exec(compile(")
        second = source.find("exec(compile(", first + 1)
        exec_context = source[second:second + 200]
        assert "evaluate.py" in exec_context, (
            f"Second exec in {runner} should compile evaluate.py, got: {exec_context[:100]}"
        )

    @pytest.mark.parametrize("runner,algo_file,eval_file", DUAL_INJECTION_RUNNERS)
    def test_extracts_apply_fn_from_first_exec(self, runner, algo_file, eval_file):
        """After first exec, must extract the patched apply function."""
        source = (PROJECT_ROOT / runner).read_text()
        assert "apply_memit_to_model" in source or "apply_AlphaEdit_to_model" in source

    @pytest.mark.parametrize("runner,algo_file,eval_file", DUAL_INJECTION_RUNNERS)
    def test_passes_apply_fn_to_second_exec(self, runner, algo_file, eval_file):
        """Second exec namespace must include the patched apply function."""
        source = (PROJECT_ROOT / runner).read_text()
        second_exec = source.find("exec(compile(", source.find("exec(compile(") + 1)
        namespace_section = source[second_exec:second_exec + 500]
        assert any(x in namespace_section for x in [
            "apply_memit_to_model", "apply_AlphaEdit_to_model",
            "_patched_apply", "apply_fn_name",
        ])


class TestDualInjectionAlgorithmPatches:
    """Verify the algorithm-level patches are correct."""

    @pytest.mark.parametrize("runner,algo_file,eval_file", DUAL_INJECTION_RUNNERS)
    def test_uses_shared_mega_batch_eval(self, runner, algo_file, eval_file):
        source = (PROJECT_ROOT / runner).read_text()
        assert "get_mega_batch_eval_source" in source, (
            f"{runner} must use shared mega_batch_eval module"
        )

    @pytest.mark.parametrize("runner,algo_file,eval_file", DUAL_INJECTION_RUNNERS)
    def test_has_checkpoint_save_load(self, runner, algo_file, eval_file):
        source = (PROJECT_ROOT / runner).read_text()
        assert "_ckpt_save" in source or "save_checkpoint" in source
        assert "_ckpt_load" in source or "load_checkpoint" in source

    @pytest.mark.parametrize("runner,algo_file,eval_file", DUAL_INJECTION_RUNNERS)
    def test_has_batch_counter(self, runner, algo_file, eval_file):
        source = (PROJECT_ROOT / runner).read_text()
        assert "_batch_idx" in source or "batch_idx" in source or "cnt" in source


class TestDualInjectionSpecific:
    """Runner-specific tests."""

    def test_memit_seq_has_prev_cache(self):
        source = (PROJECT_ROOT / "src/runners/memit_sequential_runner.py").read_text()
        assert "_memit_prev_cache" in source

    def test_polykernel_seqreg_has_kernel_solve(self):
        source = (PROJECT_ROOT / "src/polykernel/polykernel_seqreg_runner.py").read_text()
        assert "kernel" in source.lower()
        assert "linalg.solve" in source

    def test_polykernel_seqreg_has_revive(self):
        source = (PROJECT_ROOT / "src/polykernel/polykernel_seqreg_runner.py").read_text()
        assert "_revive_apply" in source

    def test_polykernel_seqreg_has_base_alg_guard(self):
        source = (PROJECT_ROOT / "src/polykernel/polykernel_seqreg_runner.py").read_text()
        assert "_ckpt_base_alg" in source

    def test_pathguard_has_displacement(self):
        source = (PROJECT_ROOT / "src/runners/pathguard_runner.py").read_text()
        assert "displacement" in source.lower() or "pathguard" in source.lower()

    def test_polykernel_editor_supports_alphaedit_and_memit(self):
        source = (PROJECT_ROOT / "src/polykernel/polykernel_editor_runner.py").read_text()
        assert "AlphaEdit" in source
        assert "MEMIT" in source


class TestMigrationReadiness:
    """Track what's needed before these runners can migrate to the harness."""

    @pytest.mark.parametrize("runner,algo_file,eval_file", DUAL_INJECTION_RUNNERS)
    def test_evaluate_exec_is_removable(self, runner, algo_file, eval_file):
        """The evaluate.py exec CAN be replaced with run_experiment() + hooks.
        This test documents the migration path, not the current state."""
        source = (PROJECT_ROOT / runner).read_text()
        # The evaluate.py exec passes shared state via namespace dict
        # Count how many shared variables are in the namespace
        second_exec = source.find("exec(compile(", source.find("exec(compile(") + 1)
        if second_exec == -1:
            pytest.skip("Can't find second exec")
        namespace = source[second_exec:source.find("})", second_exec) + 2]
        shared_vars = namespace.count('":')
        # Document: this many shared variables need to become hook closures
        assert shared_vars > 0, "No shared variables found — should be easy to migrate"

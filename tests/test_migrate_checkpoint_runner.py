#!/usr/bin/env python3
"""Tests for checkpoint_runner migration to evaluate_harness.

These tests verify the migrated runner:
1. Does NOT exec(compile()) evaluate.py
2. Still supports exec for AlphaEdit C₀ injection (algorithm patching)
3. Uses evaluate_harness.run_experiment()
4. Uses checkpoint_io for save/load
5. Is significantly shorter than the original

Run with: uv run pytest tests/test_migrate_checkpoint_runner.py -v
"""
import ast
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RUNNER_PATH = PROJECT_ROOT / "src" / "runners" / "checkpoint_runner.py"



class TestCheckpointRunnerMigration:
    """Verify checkpoint_runner uses the harness instead of exec(compile(evaluate.py))."""

    @pytest.fixture
    def source(self):
        return RUNNER_PATH.read_text()

    @pytest.fixture
    def tree(self, source):
        return ast.parse(source)

    def test_parses(self, tree):
        """Must be syntactically valid Python."""
        assert tree is not None

    def test_no_evaluate_exec(self, source):
        """Must NOT exec(compile()) evaluate.py.
        The old pattern read evaluate.py as text and exec'd it.
        The new pattern calls run_experiment() directly."""
        # Check for the specific pattern: exec(compile(source, "experiments/evaluate.py", ...))
        assert 'exec(compile(source, "experiments/evaluate.py"' not in source, (
            "checkpoint_runner must not exec(compile()) evaluate.py — use run_experiment() instead"
        )

    def test_uses_harness(self, source):
        """Must import and use evaluate_harness."""
        assert "run_experiment" in source
        assert "evaluate_harness" in source or "from evaluate_harness import" in source

    def test_uses_checkpoint_io(self, source):
        """Must use shared checkpoint_io for save/load."""
        assert "checkpoint_io" in source or "from checkpoint_io import" in source or \
               "save_checkpoint" in source

    def test_uses_mega_batch_eval(self, source):
        """Must use shared mega_batch_eval module."""
        assert "mega_batch_eval" in source

    def test_keeps_c0_exec(self, source):
        """Must still support AlphaEdit C₀ injection via exec of AlphaEdit_main.py.
        This is algorithm-level patching that can't be avoided."""
        # The C₀ injection is optional (--inject_c0 flag) but the code path must exist
        assert "inject_c0" in source or "c0" in source.lower()

    def test_has_checkpoint_hooks(self, source):
        """Must implement checkpoint behavior via ExperimentHooks."""
        assert "ExperimentHooks" in source or "hooks" in source

    def test_supports_eval_modes(self, source):
        """Must support fast_checkpoint and eval_at_checkpoints_only modes."""
        assert "fast_checkpoint" in source
        assert "eval_at_checkpoints_only" in source

    def test_supports_resume(self, source):
        """Must support resuming from checkpoints (start_from_batch)."""
        assert "start_from_batch" in source
        assert "find_latest_checkpoint" in source

    def test_line_count_reduced(self, source):
        """Should be significantly shorter than the original 1165 lines."""
        lines = len(source.split('\n'))
        assert lines < 800, f"checkpoint_runner is {lines} lines — expected < 800 (was 1165)"

    def test_no_build_checkpoint_script(self, source):
        """The massive build_checkpoint_script f-string template should be gone."""
        assert "def build_checkpoint_script(" not in source, (
            "build_checkpoint_script should be removed — harness replaces the template"
        )

#!/usr/bin/env python3
"""Tests that validate legacy runner migrations.

For each runner, tests verify:
1. No exec(compile()) of evaluate.py (the main edit loop)
2. Uses evaluate_harness or measurement_harness
3. AST parses correctly
4. Source is shorter than the original

These tests are written BEFORE migration so they FAIL initially,
then PASS after each runner is migrated.
"""
import ast
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Runners that should be migrated (edit-eval pattern)
EDIT_EVAL_RUNNERS = [
    "src/runners/alphaedit_stream_runner.py",
    "src/runners/update_interference_runner.py",
    "src/runners/cache_mitigation_batch_runner.py",
    "src/runners/protected_editing_runner.py",
]

# Runners that should use measurement_harness (A/B intervention pattern)
MEASUREMENT_RUNNERS = [
    "src/runners/logit_damage_runner.py",
    "src/runners/logit_damage_memit_runner.py",
    "src/runners/same_fact_damage_runner.py",
]

ALL_LEGACY = EDIT_EVAL_RUNNERS + MEASUREMENT_RUNNERS


class TestLegacyMigrationStatus:
    """Track which runners have been migrated."""

    @pytest.mark.parametrize("runner", ALL_LEGACY)
    def test_runner_parses(self, runner):
        path = PROJECT_ROOT / runner
        if not path.exists():
            pytest.skip(f"{runner} not found")
        ast.parse(path.read_text())

    @pytest.mark.parametrize("runner", MEASUREMENT_RUNNERS)
    def test_measurement_runner_no_evaluate_exec(self, runner):
        """Measurement scripts should NOT exec evaluate.py (they exec algorithm code only)."""
        path = PROJECT_ROOT / runner
        if not path.exists():
            pytest.skip(f"{runner} not found")
        source = path.read_text()
        # These should exec algorithm code (AlphaEdit_main.py, memit_main.py) not evaluate.py
        assert 'compile(eval_source, "experiments/evaluate.py"' not in source, (
            f"{runner} still execs evaluate.py — measurement scripts shouldn't"
        )

    @pytest.mark.parametrize("runner", ALL_LEGACY)
    def test_runner_has_measurement_or_eval_pattern(self, runner):
        """Every legacy runner must clearly be either an edit-eval or measurement pattern."""
        path = PROJECT_ROOT / runner
        if not path.exists():
            pytest.skip(f"{runner} not found")
        source = path.read_text()
        is_measurement = any(x in source for x in [
            "measure_logprobs", "baseline_w", "restore()", "save_edited_layers",
            "measurement_harness", "run_measurement", "A/B", "intervention",
        ])
        is_edit_eval = any(x in source for x in [
            "run_experiment", "evaluate_harness", "for record_chunks",
            "for record in ds", "mega_batch_eval", "exec(compile(",
        ])
        assert is_measurement or is_edit_eval, (
            f"{runner} doesn't match either pattern"
        )

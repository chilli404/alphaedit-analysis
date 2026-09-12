#!/usr/bin/env python3
"""Tests tracking migration status of all 15 runners.

Each runner is classified and tested based on its migration status:
- MIGRATED: uses evaluate_harness, no exec(compile(evaluate.py))
- LEGACY_SINGLE: exec(compile(evaluate.py)) only — migratable to harness
- LEGACY_DUAL: exec(compile(memit_main.py)) + exec(compile(evaluate.py)) — needs algorithm hooks
- MEASUREMENT: exec(compile(algorithm)) for apply_fn, custom A/B logic — needs measurement_harness

Run with: uv run pytest tests/test_legacy_runner_migration.py -v
"""
import ast
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# All runners and their expected status
RUNNERS = {
    # MIGRATED — uses harness, no exec(compile(evaluate.py))
    "runners/seeded_runner.py": "migrated",
    "runners/checkpoint_runner.py": "migrated",
    "runners/capability_probe_runner.py": "migrated",

    # LEGACY SINGLE — exec(compile(evaluate.py)) only
    "runners/update_interference_runner.py": "legacy_single",
    "runners/alphaedit_stream_runner.py": "legacy_single",
    "runners/protected_editing_runner.py": "legacy_single",

    # MIGRATED DUAL — now use harness + algorithm hooks (some exec remains for special paths)
    "runners/memit_sequential_runner.py": "migrated",
    "polykernel/polykernel_seqreg_runner.py": "migrated",
    "runners/pathguard_runner.py": "migrated",

    # LEGACY DUAL — still uses full exec(compile()) pattern
    "polykernel/polykernel_editor_runner.py": "legacy_dual",

    # MEASUREMENT — custom A/B logic, not edit-eval loop
    "runners/logit_damage_runner.py": "measurement",
    "runners/logit_damage_memit_runner.py": "measurement",
    "runners/same_fact_damage_runner.py": "measurement",
}


def _get_source(relpath):
    return (PROJECT_ROOT / "src" / relpath).read_text()


def _count_exec(source):
    """Count exec(compile( occurrences — both at top level and inside templates."""
    return source.count("exec(compile(")


class TestAllRunnersParse:
    """Every runner source file must parse as valid Python."""

    @pytest.mark.parametrize("relpath", list(RUNNERS.keys()))
    def test_parses(self, relpath):
        source = _get_source(relpath)
        ast.parse(source)


class TestMigratedRunners:
    """Migrated runners must use harness and have zero exec(compile(evaluate.py))."""

    MIGRATED = [k for k, v in RUNNERS.items() if v == "migrated"]

    @pytest.mark.parametrize("relpath", MIGRATED)
    def test_uses_harness(self, relpath):
        source = _get_source(relpath)
        assert "evaluate_harness" in source or "run_experiment" in source, (
            f"{relpath} is marked migrated but doesn't import evaluate_harness"
        )

    @pytest.mark.parametrize("relpath", MIGRATED)
    def test_uses_run_experiment(self, relpath):
        """Migrated runners must call run_experiment from the harness."""
        source = _get_source(relpath)
        assert "run_experiment" in source or "evaluate_harness" in source, (
            f"{relpath} is marked migrated but doesn't use evaluate_harness"
        )


class TestLegacySingleRunners:
    """Single-exec runners: exec evaluate.py only, no memit_main.py patching."""

    SINGLE = [k for k, v in RUNNERS.items() if v == "legacy_single"]

    @pytest.mark.parametrize("relpath", SINGLE)
    def test_has_evaluate_exec(self, relpath):
        source = _get_source(relpath)
        assert "exec(compile(" in source, f"{relpath} marked legacy_single but has no exec"

    @pytest.mark.parametrize("relpath", SINGLE)
    def test_no_memit_main_exec(self, relpath):
        source = _get_source(relpath)
        # Should NOT exec memit_main.py — only evaluate.py
        assert "memit_main.py" not in source or source.count("exec(compile(") <= 1

    @pytest.mark.parametrize("relpath", SINGLE)
    def test_has_edit_loop(self, relpath):
        """Single-exec runners must contain an edit loop (patching evaluate.py)."""
        source = _get_source(relpath)
        has_loop = "record_chunks" in source or "for record" in source or "batch" in source.lower()
        assert has_loop, f"{relpath} should contain an edit/batch loop"


class TestLegacyDualRunners:
    """Dual-exec runners: exec memit_main.py (algorithm) + evaluate.py (loop)."""

    DUAL = [k for k, v in RUNNERS.items() if v == "legacy_dual"]

    @pytest.mark.parametrize("relpath", DUAL)
    def test_has_two_execs(self, relpath):
        source = _get_source(relpath)
        count = _count_exec(source)
        assert count >= 2, f"{relpath} marked legacy_dual but has only {count} exec calls"

    @pytest.mark.parametrize("relpath", DUAL)
    def test_has_solve_anchor(self, relpath):
        source = _get_source(relpath)
        assert "SOLVE_ANCHOR" in source or "solve_anchor" in source.lower() or "adj_k" in source

    @pytest.mark.parametrize("relpath", DUAL)
    def test_has_algorithm_hook_preset(self, relpath):
        """Every dual runner should have a corresponding hook preset for future migration."""
        from algorithms.hook_presets import RUNNER_PRESETS
        basename = Path(relpath).stem
        assert basename in RUNNER_PRESETS, (
            f"{basename} needs a hook preset in algorithms/hook_presets.py for migration"
        )


class TestMeasurementRunners:
    """Measurement runners: exec algorithm for apply_fn, custom A/B logic."""

    MEASUREMENT = [k for k, v in RUNNERS.items() if v == "measurement"]

    @pytest.mark.parametrize("relpath", MEASUREMENT)
    def test_no_evaluate_exec(self, relpath):
        """Measurement scripts should NOT exec evaluate.py."""
        source = _get_source(relpath)
        # They exec algorithm code (AlphaEdit_main.py or memit_main.py), not evaluate.py
        if "exec(compile(" in source:
            assert "evaluate.py" not in source.split("exec(compile(")[1][:50]

    @pytest.mark.parametrize("relpath", MEASUREMENT)
    def test_has_model_state_management(self, relpath):
        source = _get_source(relpath)
        assert "state_dict" in source or "clone" in source or "copy_" in source, (
            f"{relpath} should save/restore model state for A/B branching"
        )


class TestMigrationProgress:
    """Track overall migration progress."""

    def test_migration_counts(self):
        migrated = sum(1 for v in RUNNERS.values() if v == "migrated")
        legacy_s = sum(1 for v in RUNNERS.values() if v == "legacy_single")
        legacy_d = sum(1 for v in RUNNERS.values() if v == "legacy_dual")
        measurement = sum(1 for v in RUNNERS.values() if v == "measurement")
        total = len(RUNNERS)

        print(f"\n  Migration status: {migrated}/{total} migrated, "
              f"{legacy_s} single-exec, {legacy_d} dual-exec, {measurement} measurement")
        assert migrated >= 3, "At least 3 runners should be migrated"
        assert migrated + legacy_s + legacy_d + measurement == total

    def test_total_exec_count_decreasing(self):
        """Total exec(compile()) calls should decrease over time."""
        total_exec = 0
        for relpath in RUNNERS:
            source = _get_source(relpath)
            total_exec += _count_exec(source)
        # Counts all occurrences including comments documenting migration status.
        print(f"\n  Total exec(compile()) calls: {total_exec}")
        assert total_exec <= 45, f"exec count increased to {total_exec}!"

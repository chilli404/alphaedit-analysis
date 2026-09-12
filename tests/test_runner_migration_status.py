#!/usr/bin/env python3
"""Tests documenting runner migration status.

Tracks which runners have been migrated from exec(compile(evaluate.py))
to evaluate_harness.run_experiment(), and verifies the migration is correct.

Run with: uv run pytest tests/test_runner_migration_status.py -v
"""
import sys
from pathlib import Path
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "src" / "util"))

# Runners that have been fully migrated to the harness
MIGRATED = [
    "src/runners/seeded_runner.py",
]

# Runners that still use exec(compile(evaluate.py)) but have the harness available
# These can be migrated with hooks but haven't been yet
MIGRATABLE = [
    "src/runners/checkpoint_runner.py",  # checkpoint hooks needed
    "src/runners/capability_probe_runner.py",  # GLUEEval monkey-patch
    "src/runners/cache_mitigation_batch_runner.py",  # simple
]

# Runners that use dual injection (exec both memit_main.py and evaluate.py)
# Only the evaluate.py exec can be replaced; memit_main.py exec stays
DUAL_INJECTION = [
    "src/polykernel/polykernel_seqreg_runner.py",
    "src/runners/memit_sequential_runner.py",
    "src/runners/pathguard_runner.py",
    "src/polykernel/polykernel_editor_runner.py",
    "src/runners/alphaedit_stream_runner.py",
]

# Runners that aren't edit-eval loops (measurement/intervention scripts)
NOT_EDIT_LOOPS = [
    "src/runners/logit_damage_runner.py",
    "src/runners/logit_damage_memit_runner.py",
    "src/runners/same_fact_damage_runner.py",
    "src/runners/protected_editing_runner.py",
    "src/runners/update_interference_runner.py",
]


class TestMigratedRunners:
    """Runners that have been migrated must NOT use exec(compile(evaluate.py))."""

    @pytest.mark.parametrize("runner_path", MIGRATED)
    def test_no_exec_compile_evaluate(self, runner_path):
        source = (PROJECT_ROOT / runner_path).read_text()
        # Count actual exec(compile() calls in code (not comments/docstrings)
        import re
        exec_calls = re.findall(r'^\s*exec\(compile\(', source, re.MULTILINE)
        assert len(exec_calls) == 0, (
            f"{runner_path} has {len(exec_calls)} exec(compile()) calls after migration"
        )

    @pytest.mark.parametrize("runner_path", MIGRATED)
    def test_uses_harness(self, runner_path):
        source = (PROJECT_ROOT / runner_path).read_text()
        assert "run_experiment" in source or "evaluate_harness" in source, (
            f"{runner_path} should use evaluate_harness.run_experiment()"
        )

    @pytest.mark.parametrize("runner_path", MIGRATED)
    def test_no_source_replace(self, runner_path):
        source = (PROJECT_ROOT / runner_path).read_text()
        assert "source.replace(" not in source, (
            f"{runner_path} should not use source.replace() after migration"
        )


class TestDualInjectionRunners:
    """Dual-injection runners must still exec memit_main.py but could drop evaluate.py exec."""

    @pytest.mark.parametrize("runner_path", DUAL_INJECTION)
    def test_has_memit_exec(self, runner_path):
        source = (PROJECT_ROOT / runner_path).read_text()
        assert "exec(compile(" in source, (
            f"{runner_path} should still exec(compile()) memit_main.py for kernel patches"
        )

    @pytest.mark.parametrize("runner_path", DUAL_INJECTION)
    def test_parses(self, runner_path):
        import ast
        source = (PROJECT_ROOT / runner_path).read_text()
        ast.parse(source)


class TestNotEditLoopRunners:
    """Measurement scripts don't follow the edit-eval pattern and can't use the harness."""

    @pytest.mark.parametrize("runner_path", NOT_EDIT_LOOPS)
    def test_parses(self, runner_path):
        import ast
        path = PROJECT_ROOT / runner_path
        if not path.exists():
            pytest.skip(f"{runner_path} not present")
        source = path.read_text()
        ast.parse(source)


class TestMigrationProgress:
    """Track overall migration progress."""

    def test_total_exec_compile_count(self):
        """Count remaining exec(compile()) calls across all runners."""
        import glob
        total = 0
        for pattern in ["src/runners/*.py", "src/polykernel/*.py"]:
            for f in glob.glob(str(PROJECT_ROOT / pattern)):
                if "__pycache__" in f or "test" in f:
                    continue
                source = Path(f).read_text()
                total += source.count("exec(compile(")
        # This number should decrease as we migrate
        # Current: 27 (1 migrated, 26 remaining)
        assert total <= 30, f"exec(compile() count is {total}, expected <= 30"
        print(f"\n  Migration progress: {total} exec(compile()) remaining")

    def test_migrated_count(self):
        assert len(MIGRATED) >= 1, "At least seeded_runner should be migrated"

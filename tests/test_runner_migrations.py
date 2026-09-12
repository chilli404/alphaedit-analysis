#!/usr/bin/env python3
"""
Tests for runner migration from exec(compile()) to evaluate_harness.

Each migrated runner must:
  1. Not use exec(compile(evaluate.py))
  2. Import from evaluate_harness
  3. Parse as valid Python
  4. Not use source.replace() for evaluate.py patches

Runners that exec ONLY algorithm files (not evaluate.py) are handled
differently — they can import the apply function normally instead.

Run with: uv run pytest tests/test_runner_migrations.py -v
"""
import ast
import re
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _has_exec_compile(filepath: Path) -> bool:
    """Check if a Python file has exec(compile(...)) calls using AST (ignores docstrings)."""
    tree = ast.parse(filepath.read_text())
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            func = node.func
            if isinstance(func, ast.Name) and func.id == "exec":
                for arg in node.args:
                    if isinstance(arg, ast.Call):
                        inner = arg.func
                        if isinstance(inner, ast.Name) and inner.id == "compile":
                            return True
    return False


# === Already migrated ===

class TestSeededRunnerMigrated:
    """seeded_runner is the reference migration — already done."""

    def test_no_exec_compile(self):
        assert not _has_exec_compile(PROJECT_ROOT / "src" / "runners" / "seeded_runner.py")

    def test_uses_harness(self):
        source = (PROJECT_ROOT / "src" / "runners" / "seeded_runner.py").read_text()
        assert "from evaluate_harness import" in source or "import evaluate_harness" in source

    def test_parses(self):
        source = (PROJECT_ROOT / "src" / "runners" / "seeded_runner.py").read_text()
        ast.parse(source)


# === Group A: Runners that exec evaluate.py (can be migrated to harness) ===

class TestCapabilityProbeRunnerMigration:
    """capability_probe_runner execs evaluate.py with probe hooks."""

    def test_parses(self):
        source = (PROJECT_ROOT / "src" / "runners" / "capability_probe_runner.py").read_text()
        ast.parse(source)

    def test_no_exec_compile(self):
        assert not _has_exec_compile(PROJECT_ROOT / "src" / "runners" / "capability_probe_runner.py"), (
            "capability_probe_runner should use evaluate_harness instead of exec(compile())"
        )

    def test_uses_harness(self):
        source = (PROJECT_ROOT / "src" / "runners" / "capability_probe_runner.py").read_text()
        assert "evaluate_harness" in source or "run_experiment" in source


class TestCacheMitigationRunnerMigration:
    """cache_mitigation_batch_runner — DEFERRED (multi-variant per model load, cache_c manipulation)."""

    def test_parses(self):
        source = (PROJECT_ROOT / "src" / "runners" / "cache_mitigation_batch_runner.py").read_text()
        ast.parse(source)

    @pytest.mark.skip(reason="Complex: multi-variant per model load with cache_c manipulation. Needs GPU verification.")
    def test_no_exec_compile(self):
        source = (PROJECT_ROOT / "src" / "runners" / "cache_mitigation_batch_runner.py").read_text()
        assert "exec(compile(" not in source


class TestProtectedEditingRunnerMigration:
    """protected_editing_runner — DEFERRED (dual source injection for protection hooks)."""

    def test_parses(self):
        source = (PROJECT_ROOT / "src" / "runners" / "protected_editing_runner.py").read_text()
        ast.parse(source)

    @pytest.mark.skip(reason="Complex: dual source injection for protection reapplication. Needs GPU verification.")
    def test_no_exec_compile_evaluate(self):
        source = (PROJECT_ROOT / "src" / "runners" / "protected_editing_runner.py").read_text()
        assert "exec(compile(" not in source


# === Group B: Runners that exec ONLY algorithm files (not evaluate.py) ===
# These can import the apply function normally instead of exec'ing the algorithm file.

class TestLogitDamageRunnerMigration:
    """logit_damage_runner execs AlphaEdit_main.py to get apply function."""

    def test_parses(self):
        source = (PROJECT_ROOT / "src" / "runners" / "logit_damage_runner.py").read_text()
        ast.parse(source)

    def test_no_exec_evaluate(self):
        """Should not exec evaluate.py (it's not an edit-eval loop)."""
        source = (PROJECT_ROOT / "src" / "runners" / "logit_damage_runner.py").read_text()
        lines = source.split("\n")
        for line in lines:
            if "exec(compile(" in line and "evaluate.py" in line:
                pytest.fail("logit_damage_runner should not exec evaluate.py")

    def test_imports_apply_fn_normally(self):
        """Should import apply_AlphaEdit_to_model normally, not via exec."""
        source = (PROJECT_ROOT / "src" / "runners" / "logit_damage_runner.py").read_text()
        has_normal_import = "from AlphaEdit" in source or "import AlphaEdit" in source
        has_exec_import = 'exec(compile(ae_src, "AlphaEdit' in source
        assert has_normal_import or not has_exec_import, (
            "logit_damage_runner should import apply function normally"
        )


class TestLogitDamageMemitRunnerMigration:
    """logit_damage_memit_runner execs memit_main.py to get apply function."""

    def test_parses(self):
        source = (PROJECT_ROOT / "src" / "runners" / "logit_damage_memit_runner.py").read_text()
        ast.parse(source)

    def test_no_exec_evaluate(self):
        source = (PROJECT_ROOT / "src" / "runners" / "logit_damage_memit_runner.py").read_text()
        lines = source.split("\n")
        for line in lines:
            if "exec(compile(" in line and "evaluate.py" in line:
                pytest.fail("logit_damage_memit_runner should not exec evaluate.py")


class TestSameFactDamageRunnerMigration:
    """same_fact_damage_runner execs AlphaEdit_main.py to get apply function."""

    def test_parses(self):
        source = (PROJECT_ROOT / "src" / "runners" / "same_fact_damage_runner.py").read_text()
        ast.parse(source)

    def test_no_exec_evaluate(self):
        source = (PROJECT_ROOT / "src" / "runners" / "same_fact_damage_runner.py").read_text()
        lines = source.split("\n")
        for line in lines:
            if "exec(compile(" in line and "evaluate.py" in line:
                pytest.fail("same_fact_damage_runner should not exec evaluate.py")


# === Summary test: count remaining exec(compile()) calls ===

class TestOverallMigrationProgress:
    """Track overall migration progress."""

    FULLY_MIGRATED = [
        "src/runners/seeded_runner.py",
        "src/runners/capability_probe_runner.py",
    ]

    # These should not exec evaluate.py (they do custom measurement, not edit-eval loops)
    NO_EVALUATE_EXEC = [
        "src/runners/logit_damage_runner.py",
        "src/runners/logit_damage_memit_runner.py",
        "src/runners/same_fact_damage_runner.py",
    ]

    def test_migrated_runners_use_harness(self):
        for runner in self.FULLY_MIGRATED:
            path = PROJECT_ROOT / runner
            source = path.read_text()
            assert "evaluate_harness" in source, f"{runner} doesn't use evaluate_harness"
            assert not _has_exec_compile(path), f"{runner} still has exec(compile()) calls"

    def test_measurement_runners_dont_exec_evaluate(self):
        for runner in self.NO_EVALUATE_EXEC:
            source = (PROJECT_ROOT / runner).read_text()
            for line in source.split("\n"):
                if "exec(compile(" in line and "evaluate.py" in line:
                    pytest.fail(f"{runner} should not exec evaluate.py")

    def test_count_remaining_exec_compile(self):
        """Track how many exec(compile()) calls remain across all runners."""
        import glob
        total = 0
        for pattern in ["src/runners/*.py", "src/polykernel/*.py"]:
            for filepath in glob.glob(str(PROJECT_ROOT / pattern)):
                source = Path(filepath).read_text()
                count = source.count("exec(compile(")
                total += count
        # Counts ALL occurrences including comments/docstrings documenting migration status.
        print(f"\n  Remaining exec(compile()) calls: {total}")
        assert total < 45, f"Too many exec(compile()) calls remaining: {total}"

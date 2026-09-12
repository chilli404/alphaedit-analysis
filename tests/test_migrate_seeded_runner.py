#!/usr/bin/env python3
"""Tests for the migrated seeded_runner (harness-based, no exec/compile).

These tests define what the migrated runner MUST satisfy.
Run BEFORE and AFTER migration to verify the transition.

Run with: uv run pytest tests/test_migrate_seeded_runner.py -v
"""
import ast
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RUNNER_PATH = PROJECT_ROOT / "src" / "runners" / "seeded_runner.py"

sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "src" / "util"))


class TestSeededRunnerMigration:
    """Post-migration: seeded_runner must use the harness, not exec/compile."""

    def test_source_parses(self):
        source = RUNNER_PATH.read_text()
        ast.parse(source)

    def test_uses_evaluate_harness(self):
        source = RUNNER_PATH.read_text()
        assert "evaluate_harness" in source or "run_experiment" in source, (
            "seeded_runner must import from evaluate_harness"
        )

    def test_no_exec_compile_evaluate_py(self):
        """Must not exec(compile()) vendor evaluate.py. exec of our own shared modules is OK."""
        source = RUNNER_PATH.read_text()
        assert 'exec(compile(source' not in source, (
            "seeded_runner must not exec(compile(source...)) — that's the vendor evaluate.py pattern"
        )
        assert '"experiments/evaluate.py"' not in source, (
            "seeded_runner must not reference vendor evaluate.py path"
        )

    def test_no_source_replace(self):
        source = RUNNER_PATH.read_text()
        assert "source.replace(" not in source, (
            "seeded_runner must not do string patching — patches applied by apply_all.py"
        )

    def test_no_read_evaluate_py(self):
        source = RUNNER_PATH.read_text()
        assert 'open("experiments/evaluate.py"' not in source, (
            "seeded_runner must not read evaluate.py as text"
        )

    def test_has_main(self):
        source = RUNNER_PATH.read_text()
        assert "def main(" in source or 'if __name__ == "__main__"' in source

    def test_seeds_all_rng(self):
        """Must seed Python random, numpy, torch CPU, torch CUDA."""
        source = RUNNER_PATH.read_text()
        assert "random.seed" in source
        assert "np.random.seed" in source or "numpy" in source
        assert "torch.manual_seed" in source
        assert "torch.cuda.manual_seed" in source or "manual_seed_all" in source

    def test_sets_cuda_device(self):
        source = RUNNER_PATH.read_text()
        assert "CUDA_VISIBLE_DEVICES" in source

    def test_sets_deterministic(self):
        source = RUNNER_PATH.read_text()
        assert "deterministic" in source

    def test_imports_apply_functions(self):
        """Must import algorithm apply functions via normal Python imports."""
        source = RUNNER_PATH.read_text()
        assert "apply_AlphaEdit_to_model" in source or "apply_memit_to_model" in source or "ALG_DICT" in source

    def test_loads_hparams(self):
        source = RUNNER_PATH.read_text()
        assert "hparams" in source.lower() and "from_json" in source

    def test_records_metadata(self):
        source = RUNNER_PATH.read_text()
        assert "record_metadata" in source or "metadata" in source.lower()

    def test_uses_mega_batch_eval(self):
        """Must use mega_batch_eval for evaluation (not vendor per-record loop)."""
        source = RUNNER_PATH.read_text()
        assert "mega_batch_eval" in source

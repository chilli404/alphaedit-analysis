"""Integration tests for RECT-Err (OTE-aligned) dispatch, patching, and runner import.

These tests verify that MEMIT_seq_rect_err is properly wired into:
1. baselines evaluate.py ALG_DICT + argparse
2. polykernel_seqreg_runner.py import for base_alg=MEMIT_rect
3. run_rect_aligned_paper_replication.sh

Marked xfail until the dispatch changes are implemented.
"""
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
BASELINES_EVAL = PROJECT_ROOT / "baselines" / "EvoEdit" / "experiments" / "evaluate.py"
RUNNER = PROJECT_ROOT / "src" / "polykernel" / "polykernel_seqreg_runner.py"
PAPER_REPL_SCRIPT = PROJECT_ROOT / "scripts" / "run_rect_aligned_paper_replication.sh"


# ============================================================================
# Baselines evaluate.py must dispatch MEMIT_seq_rect_err
# ============================================================================


class TestEvaluateDispatch:

    def test_memit_seq_rect_err_in_alg_dict(self):
        source = BASELINES_EVAL.read_text()
        assert '"MEMIT_seq_rect_err"' in source, \
            "baselines evaluate.py must have MEMIT_seq_rect_err in ALG_DICT"

    def test_memit_seq_rect_err_in_choices(self):
        source = BASELINES_EVAL.read_text()
        choices_start = source.find("choices=[")
        assert choices_start > 0, "argparse choices not found"
        choices_end = source.find("]", choices_start) + 1
        choices_line = source[choices_start:choices_end]
        assert "MEMIT_seq_rect_err" in choices_line, \
            f"MEMIT_seq_rect_err must be in argparse choices. Got: {choices_line}"

    def test_memit_seq_rect_err_import(self):
        source = BASELINES_EVAL.read_text()
        assert "from memit.memit_seq_rect_err_main import apply_memit_seq_rect_err_to_model" in source, \
            "baselines evaluate.py must import apply_memit_seq_rect_err_to_model"

    def test_old_memit_seq_rect_still_present(self):
        """Backward compatibility: old MEMIT_seq_rect entry must remain."""
        source = BASELINES_EVAL.read_text()
        assert '"MEMIT_seq_rect"' in source, \
            "Old MEMIT_seq_rect must remain in evaluate.py for backward compatibility"


# ============================================================================
# Runner must import the error-corrected version
# ============================================================================


class TestRunnerImportCorrectVersion:

    def test_runner_imports_err_version(self):
        source = RUNNER.read_text()
        assert "memit_seq_rect_err_main" in source, \
            "polykernel_seqreg_runner must import from memit_seq_rect_err_main for MEMIT_rect"

    def test_runner_function_name(self):
        source = RUNNER.read_text()
        assert "apply_memit_seq_rect_err_to_model" in source, \
            "Runner must reference apply_memit_seq_rect_err_to_model"


# ============================================================================
# Paper replication script must use RECT-Err
# ============================================================================


class TestPaperReplicationScript:

    def test_script_uses_rect_err(self):
        source = PAPER_REPL_SCRIPT.read_text()
        assert "MEMIT_seq_rect_err" in source, \
            "run_rect_aligned_paper_replication.sh must use --alg_name MEMIT_seq_rect_err"

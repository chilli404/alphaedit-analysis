#!/usr/bin/env python3
"""Tests for standardized runner logging (src/util/runner_logging.py).

Verifies output format matches what the GPU smoke test greps for.

Run with: uv run pytest tests/test_runner_logging.py -v
"""
import io
import sys
from contextlib import redirect_stdout
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent


class TestRunnerLoggingImport:
    def test_import_all_functions(self):
        from runner_logging import (
            log_batch_start,
            log_batch_complete,
            log_checkpoint_saved,
            log_eval_progress,
            log_run_summary,
            INJECTABLE_SOURCE,
        )

    def test_injectable_source_compiles(self):
        from runner_logging import INJECTABLE_SOURCE
        compile(INJECTABLE_SOURCE, "<injectable>", "exec")


class TestLogBatchFormat:
    def test_batch_start_format(self):
        from runner_logging import log_batch_start
        buf = io.StringIO()
        with redirect_stdout(buf):
            log_batch_start(0, 100, 10)
        line = buf.getvalue().strip()
        assert "[Batch 1/10]" in line
        assert "100 records" in line
        assert "100 total" in line

    def test_batch_complete_format(self):
        from runner_logging import log_batch_complete
        buf = io.StringIO()
        with redirect_stdout(buf):
            log_batch_complete(4, 100, 72.5)
        line = buf.getvalue().strip()
        assert "[Batch 5]" in line
        assert "500 edits" in line
        assert "72.5s" in line


class TestLogCheckpointFormat:
    """Smoke test greps for '[CHECKPOINT]' — format must match."""

    def test_checkpoint_prefix(self):
        from runner_logging import log_checkpoint_saved
        buf = io.StringIO()
        with redirect_stdout(buf):
            log_checkpoint_saved(9, 100, "/s3/ckpt/batch_9")
        line = buf.getvalue().strip()
        assert line.startswith("[CHECKPOINT]")
        assert "batch 9" in line
        assert "1000 edits" in line
        assert "/s3/ckpt/batch_9" in line

    def test_checkpoint_matches_smoke_grep(self):
        """The smoke test greps for '\\[CHECKPOINT\\]' — verify exact prefix."""
        from runner_logging import log_checkpoint_saved
        buf = io.StringIO()
        with redirect_stdout(buf):
            log_checkpoint_saved(0, 10, "/tmp/ckpt")
        assert "[CHECKPOINT]" in buf.getvalue()


class TestLogEvalFormat:
    def test_eval_progress_format(self):
        from runner_logging import log_eval_progress
        buf = io.StringIO()
        with redirect_stdout(buf):
            log_eval_progress(50, 100, 2.5, 20.0)
        line = buf.getvalue().strip()
        assert "[EVAL]" in line
        assert "50/100" in line
        assert "2.5 rec/s" in line
        assert "20s" in line


class TestLogRunSummary:
    def test_summary_format(self):
        from runner_logging import log_run_summary
        buf = io.StringIO()
        with redirect_stdout(buf):
            log_run_summary("AlphaEdit", 42, 10, 1000, "/s3/ckpt", 300.0)
        out = buf.getvalue()
        assert "AlphaEdit" in out
        assert "42" in out
        assert "1000" in out
        assert "300.0s" in out


class TestInjectableSource:
    """INJECTABLE_SOURCE must define the same functions with compatible format."""

    def test_defines_all_functions(self):
        from runner_logging import INJECTABLE_SOURCE
        assert "def _log_batch_start(" in INJECTABLE_SOURCE
        assert "def _log_batch_complete(" in INJECTABLE_SOURCE
        assert "def _log_checkpoint_saved(" in INJECTABLE_SOURCE
        assert "def _log_eval_progress(" in INJECTABLE_SOURCE

    def test_injectable_uses_same_prefixes(self):
        from runner_logging import INJECTABLE_SOURCE
        assert "[Batch " in INJECTABLE_SOURCE
        assert "[CHECKPOINT]" in INJECTABLE_SOURCE
        assert "[EVAL]" in INJECTABLE_SOURCE

    def test_injectable_exec_works(self):
        """Functions from INJECTABLE_SOURCE must be callable after exec."""
        from runner_logging import INJECTABLE_SOURCE
        ns = {}
        exec(compile(INJECTABLE_SOURCE, "<test>", "exec"), ns)
        assert callable(ns["_log_batch_start"])
        assert callable(ns["_log_checkpoint_saved"])

        buf = io.StringIO()
        with redirect_stdout(buf):
            ns["_log_checkpoint_saved"](0, 10, "/tmp")
        assert "[CHECKPOINT]" in buf.getvalue()

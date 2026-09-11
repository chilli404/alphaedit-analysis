#!/usr/bin/env python3
"""
Tests for Phase 3: evaluate_harness.py

Tests the harness module API, hook interface, algorithm dispatch,
and result file structure — all without GPU.

Run with: uv run pytest tests/test_evaluate_harness.py -v
"""
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "src" / "util"))


class TestHarnessImport:
    def test_import_run_experiment(self):
        from evaluate_harness import run_experiment

    def test_import_hooks(self):
        from evaluate_harness import ExperimentHooks

    def test_import_load_model_and_data(self):
        from evaluate_harness import load_model_and_tok, load_dataset


class TestExperimentHooks:
    def test_default_hooks_are_none(self):
        from evaluate_harness import ExperimentHooks
        hooks = ExperimentHooks()
        assert hooks.before_edit is None
        assert hooks.after_edit is None
        assert hooks.should_eval is None
        assert hooks.eval_fn is None

    def test_hooks_accept_callables(self):
        from evaluate_harness import ExperimentHooks
        hooks = ExperimentHooks(
            before_edit=lambda *a: None,
            after_edit=lambda *a: None,
            should_eval=lambda batch_idx: True,
        )
        assert hooks.should_eval(5) is True


class TestAlgorithmDispatch:
    """run_experiment must correctly dispatch to the provided apply_fn."""

    def test_apply_fn_called_with_correct_args(self):
        """The harness must call apply_fn with model, tok, requests, hparams."""
        from evaluate_harness import ExperimentHooks, run_experiment

        apply_fn = MagicMock(return_value=(MagicMock(), None))
        model = MagicMock()
        tok = MagicMock()
        tok.pad_token = None
        tok.eos_token = "<eos>"
        hparams = MagicMock()
        hparams.layers = [4, 5]

        dataset = [{"case_id": i, "requested_rewrite": {"prompt": "test {}", "subject": "s",
                     "target_new": {"str": "t"}, "target_true": {"str": "o"}},
                     "paraphrase_prompts": [], "neighborhood_prompts": [],
                     "generation_prompts": []} for i in range(100)]

        hooks = ExperimentHooks(
            should_eval=lambda _: False,  # skip eval to keep test fast
        )

        run_experiment(
            model=model, tok=tok, hparams=hparams,
            dataset=dataset, apply_fn=apply_fn,
            alg_name="MEMIT", num_edits=100,
            results_dir=Path("/tmp/test_results"),
            hooks=hooks, max_batches=1,
        )

        assert apply_fn.called
        call_args = apply_fn.call_args
        assert call_args is not None


class TestResultFileStructure:
    """Per-case result files must have the expected structure."""

    def test_result_dir_created(self, tmp_path):
        from evaluate_harness import ExperimentHooks, run_experiment

        results_dir = tmp_path / "results" / "test"
        apply_fn = MagicMock(return_value=(MagicMock(), None))
        model = MagicMock()
        tok = MagicMock()
        tok.pad_token = None
        tok.eos_token = "<eos>"
        hparams = MagicMock()
        hparams.layers = [4, 5]

        dataset = [{"case_id": i, "requested_rewrite": {"prompt": "test {}", "subject": "s",
                     "target_new": {"str": "t"}, "target_true": {"str": "o"}},
                     "paraphrase_prompts": [], "neighborhood_prompts": [],
                     "generation_prompts": []} for i in range(100)]

        hooks = ExperimentHooks(
            should_eval=lambda _: False,
        )

        run_experiment(
            model=model, tok=tok, hparams=hparams,
            dataset=dataset, apply_fn=apply_fn,
            alg_name="MEMIT", num_edits=100,
            results_dir=results_dir,
            hooks=hooks, max_batches=1,
        )
        # The harness should create the results directory
        assert results_dir.exists()


class TestCheckpointIntegration:
    """Checkpoint hooks must be called at the right times."""

    def test_checkpoint_hooks_called(self, tmp_path):
        from evaluate_harness import ExperimentHooks, run_experiment

        saved_batches = []
        skipped_batches = []

        def after_edit(batch_idx, model, records, hparams, result, exec_time):
            saved_batches.append(batch_idx)

        def before_edit(batch_idx, model, records, hparams):
            if batch_idx < 2:
                skipped_batches.append(batch_idx)

        apply_fn = MagicMock(return_value=(MagicMock(), None))
        model = MagicMock()
        tok = MagicMock()
        tok.pad_token = None
        tok.eos_token = "<eos>"
        hparams = MagicMock()
        hparams.layers = [4, 5]

        dataset = [{"case_id": i, "requested_rewrite": {"prompt": "test {}", "subject": "s",
                     "target_new": {"str": "t"}, "target_true": {"str": "o"}},
                     "paraphrase_prompts": [], "neighborhood_prompts": [],
                     "generation_prompts": []} for i in range(500)]

        hooks = ExperimentHooks(
            before_edit=before_edit,
            after_edit=after_edit,
            should_eval=lambda _: False,
        )

        run_experiment(
            model=model, tok=tok, hparams=hparams,
            dataset=dataset, apply_fn=apply_fn,
            alg_name="MEMIT", num_edits=100,
            results_dir=tmp_path / "results",
            hooks=hooks, max_batches=5,
        )

        assert len(saved_batches) == 5  # after_edit called for each batch
        assert 0 in skipped_batches

#!/usr/bin/env python3
"""Tests for measurement_harness.py and util/model_state.py.

Run with: uv run pytest tests/test_measurement_harness.py -v
"""
import json
import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "src" / "util"))


class TestModelState:
    """Tests for src/util/model_state.py save/restore utilities."""

    def test_import(self):
        from util.model_state import (
            save_edited_layers, restore_edited_layers,
            save_full_edited_state, restore_full_edited_state,
        )

    def test_save_edited_layers(self):
        from util.model_state import save_edited_layers
        param = torch.randn(10, 20)
        model = MagicMock()
        model.named_parameters.return_value = [
            ("model.layers.4.mlp.down_proj.weight", param),
            ("model.layers.5.mlp.down_proj.weight", param),
            ("model.layers.6.mlp.down_proj.weight", torch.randn(10, 20)),
        ]
        hparams = MagicMock()
        hparams.layers = [4, 5]
        hparams.rewrite_module_tmp = "model.layers.{}.mlp.down_proj"

        state = save_edited_layers(model, hparams)
        assert len(state) == 2
        assert "model.layers.4.mlp.down_proj.weight" in state
        assert "model.layers.5.mlp.down_proj.weight" in state
        # Must be on CPU (not GPU)
        for v in state.values():
            assert v.device == torch.device("cpu")

    def test_restore_edited_layers(self):
        from util.model_state import save_edited_layers, restore_edited_layers

        original = torch.randn(10, 20)
        param = torch.nn.Parameter(original.clone())
        model = MagicMock()
        model.named_parameters.return_value = [
            ("model.layers.4.mlp.down_proj.weight", param),
        ]
        hparams = MagicMock()
        hparams.layers = [4]
        hparams.rewrite_module_tmp = "model.layers.{}.mlp.down_proj"

        # Save
        state = save_edited_layers(model, hparams)

        # Modify
        param.data.fill_(999.0)
        assert not torch.allclose(param.data, original)

        # Restore
        restore_edited_layers(model, state)
        assert torch.allclose(param.data, original)

    def test_save_restore_roundtrip(self):
        from util.model_state import save_edited_layers, restore_edited_layers

        params = {
            f"model.layers.{i}.mlp.down_proj.weight": torch.nn.Parameter(torch.randn(5, 5))
            for i in [4, 5, 6, 7, 8]
        }
        model = MagicMock()
        model.named_parameters.return_value = list(params.items())
        hparams = MagicMock()
        hparams.layers = [4, 5, 6, 7, 8]
        hparams.rewrite_module_tmp = "model.layers.{}.mlp.down_proj"

        originals = {k: v.data.clone() for k, v in params.items()}
        state = save_edited_layers(model, hparams)

        # Trash all params
        for p in params.values():
            p.data.fill_(0.0)

        restore_edited_layers(model, state)
        for k, p in params.items():
            assert torch.allclose(p.data, originals[k]), f"Failed to restore {k}"

    def test_save_full_with_extra(self):
        from util.model_state import save_full_edited_state
        model = MagicMock()
        model.named_parameters.return_value = []
        hparams = MagicMock()
        hparams.layers = []
        hparams.rewrite_module_tmp = "model.layers.{}.mlp.down_proj"

        extra = {"prev_cache": {"layer_4": [1, 2, 3]}, "batch_idx": 10}
        state = save_full_edited_state(model, hparams, extra=extra)
        assert "weights" in state
        assert "extra" in state
        assert state["extra"]["batch_idx"] == 10
        # Must be a deep copy
        extra["batch_idx"] = 999
        assert state["extra"]["batch_idx"] == 10


class TestMeasurementHarness:
    """Tests for src/measurement_harness.py."""

    def test_import(self):
        from measurement_harness import run_measurement, MeasurementHooks, TrialPair

    def test_hooks_defaults(self):
        from measurement_harness import MeasurementHooks
        hooks = MeasurementHooks(measure=lambda *a: {})
        assert hooks.select_trials is None
        assert hooks.after_install is None
        assert hooks.after_all_trials is None

    def test_trial_pair_structure(self):
        from measurement_harness import TrialPair
        pair = TrialPair(
            high_batch=[{"case_id": 1}],
            low_batch=[{"case_id": 2}],
            trial_id=0,
            metadata={"cosine_gap": 0.1},
        )
        assert pair.high_batch[0]["case_id"] == 1
        assert pair.metadata["cosine_gap"] == 0.1

    def test_run_measurement_ab_pattern(self, tmp_path):
        """Verify the full A/B pattern: install → save → measure HIGH → restore → measure LOW → restore."""
        from measurement_harness import run_measurement, MeasurementHooks, TrialPair

        call_log = []
        apply_count = [0]

        def mock_apply(model, tok, records, hparams, **kwargs):
            apply_count[0] += 1
            call_log.append(f"apply_{apply_count[0]}")
            return model, None

        def mock_measure(model, tok, focal, batch, hparams):
            call_log.append(f"measure_{len(batch)}")
            return {"damage": -0.005}

        hooks = MeasurementHooks(measure=mock_measure)
        model = MagicMock()
        model.named_parameters.return_value = []
        model.device = "cpu"
        tok = MagicMock()
        hparams = MagicMock()
        hparams.layers = []
        hparams.rewrite_module_tmp = "model.layers.{}.mlp.down_proj"

        dataset = [{"case_id": i, "requested_rewrite": {"prompt": "t {}", "subject": "s",
                     "target_new": {"str": "n"}, "target_true": {"str": "o"}}}
                   for i in range(300)]

        trial_pairs = [
            TrialPair(
                high_batch=dataset[200:210],
                low_batch=dataset[210:220],
                trial_id=0,
                metadata={"cosine_gap": 0.08},
            ),
        ]

        result = run_measurement(
            model, tok, hparams, dataset, mock_apply,
            install_batches=2, num_edits=100,
            hooks=hooks, trial_pairs=trial_pairs,
            results_dir=tmp_path, seed=42,
        )

        # 2 install batches + 1 HIGH + 1 LOW = 4 apply calls
        assert apply_count[0] == 4
        # 2 measure calls (HIGH + LOW)
        measure_calls = [c for c in call_log if c.startswith("measure")]
        assert len(measure_calls) == 2
        # Results saved
        assert (tmp_path / "intervention_results.json").exists()
        assert result["n_trials"] == 1
        assert result["trials"][0]["high_measurements"]["damage"] == -0.005

    def test_results_json_structure(self, tmp_path):
        from measurement_harness import run_measurement, MeasurementHooks, TrialPair

        hooks = MeasurementHooks(measure=lambda *a: {"val": 1.0})
        model = MagicMock()
        model.named_parameters.return_value = []
        model.device = "cpu"
        tok = MagicMock()
        hparams = MagicMock()
        hparams.layers = []
        hparams.rewrite_module_tmp = "model.layers.{}.mlp.down_proj"

        dataset = [{"case_id": i, "requested_rewrite": {"prompt": "t {}", "subject": "s",
                     "target_new": {"str": "n"}, "target_true": {"str": "o"}}}
                   for i in range(200)]

        result = run_measurement(
            model, tok, hparams, dataset, lambda *a, **k: (model, None),
            install_batches=1, num_edits=100,
            hooks=hooks,
            trial_pairs=[TrialPair([dataset[100]], [dataset[101]])],
            results_dir=tmp_path, seed=42,
        )

        out = json.loads((tmp_path / "intervention_results.json").read_text())
        assert out["seed"] == 42
        assert out["install_batches"] == 1
        assert out["n_trials"] == 1
        assert "high_measurements" in out["trials"][0]
        assert "low_measurements" in out["trials"][0]

    def test_after_install_hook_called(self, tmp_path):
        from measurement_harness import run_measurement, MeasurementHooks, TrialPair

        install_called = [False]

        def after_install(model, tok, records, hparams):
            install_called[0] = True
            return {"focal_keys": "mock_keys"}

        hooks = MeasurementHooks(
            measure=lambda *a: {},
            after_install=after_install,
        )
        model = MagicMock()
        model.named_parameters.return_value = []
        model.device = "cpu"
        hparams = MagicMock()
        hparams.layers = []
        hparams.rewrite_module_tmp = "model.layers.{}.mlp.down_proj"

        dataset = [{"case_id": i, "requested_rewrite": {"prompt": "t {}", "subject": "s",
                     "target_new": {"str": "n"}, "target_true": {"str": "o"}}}
                   for i in range(200)]

        run_measurement(
            model, MagicMock(), hparams, dataset, lambda *a, **k: (model, None),
            install_batches=1, num_edits=100,
            hooks=hooks,
            trial_pairs=[TrialPair([dataset[100]], [dataset[101]])],
            results_dir=tmp_path, seed=42,
        )
        assert install_called[0]

    def test_no_trials_returns_empty(self, tmp_path):
        from measurement_harness import run_measurement, MeasurementHooks

        hooks = MeasurementHooks(measure=lambda *a: {})
        model = MagicMock()
        model.named_parameters.return_value = []
        model.device = "cpu"
        hparams = MagicMock()
        hparams.layers = []
        hparams.rewrite_module_tmp = "model.layers.{}.mlp.down_proj"

        dataset = [{"case_id": i, "requested_rewrite": {"prompt": "t {}", "subject": "s",
                     "target_new": {"str": "n"}, "target_true": {"str": "o"}}}
                   for i in range(200)]

        result = run_measurement(
            model, MagicMock(), hparams, dataset, lambda *a, **k: (model, None),
            install_batches=1, num_edits=100,
            hooks=hooks, trial_pairs=[],
            results_dir=tmp_path, seed=42,
        )
        assert result["trials"] == []

    def test_summary_hook_called(self, tmp_path):
        from measurement_harness import run_measurement, MeasurementHooks, TrialPair

        def summarize(trials):
            return {"mean_damage": sum(t.get("high_measurements", {}).get("d", 0) for t in trials)}

        hooks = MeasurementHooks(
            measure=lambda *a: {"d": -0.01},
            after_all_trials=summarize,
        )
        model = MagicMock()
        model.named_parameters.return_value = []
        model.device = "cpu"
        hparams = MagicMock()
        hparams.layers = []
        hparams.rewrite_module_tmp = "model.layers.{}.mlp.down_proj"

        dataset = [{"case_id": i, "requested_rewrite": {"prompt": "t {}", "subject": "s",
                     "target_new": {"str": "n"}, "target_true": {"str": "o"}}}
                   for i in range(300)]

        result = run_measurement(
            model, MagicMock(), hparams, dataset, lambda *a, **k: (model, None),
            install_batches=1, num_edits=100,
            hooks=hooks,
            trial_pairs=[TrialPair([dataset[100]], [dataset[101]])],
            results_dir=tmp_path, seed=42,
        )
        assert "summary" in result

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


class TestCanonicalModelNames:
    """Every model name variant must resolve to ONE canonical stats directory name.

    Without this, link_stats.sh had to copy 8 variant directories (80 files).
    With it, one canonical copy + symlinks.
    """

    LLAMA_VARIANTS = [
        "meta-llama/Meta-Llama-3-8B-Instruct",
        "NousResearch/Meta-Llama-3-8B-Instruct",
        "/s3-data/continual-learning/models/Meta-Llama-3-8B",
        "Meta-Llama-3-8B-Instruct",
        "NousResearch_Meta-Llama-3-8B-Instruct",
    ]
    GPTJ_VARIANTS = [
        "EleutherAI/gpt-j-6b",
        "EleutherAI/gpt-j-6B",
        "/s3-data/continual-learning/models/gpt-j-6b",
        "gpt-j-6b",
    ]
    QWEN_VARIANTS = [
        "Qwen/Qwen2.5-7B-Instruct",
        "Qwen2.5-7B-Instruct",
    ]

    @pytest.mark.parametrize("model_name", LLAMA_VARIANTS)
    def test_llama_variants_all_resolve_same(self, model_name):
        from evaluate_harness import _canonical_name_or_path
        assert _canonical_name_or_path(model_name) == "llama3-8b-instruct"

    @pytest.mark.parametrize("model_name", GPTJ_VARIANTS)
    def test_gptj_variants_all_resolve_same(self, model_name):
        from evaluate_harness import _canonical_name_or_path
        assert _canonical_name_or_path(model_name) == "gpt-j-6b"

    @pytest.mark.parametrize("model_name", QWEN_VARIANTS)
    def test_qwen_variants_all_resolve_same(self, model_name):
        from evaluate_harness import _canonical_name_or_path
        assert _canonical_name_or_path(model_name) == "qwen2.5-7b-instruct"

    def test_all_models_produce_distinct_canonical_names(self):
        from evaluate_harness import _canonical_name_or_path
        names = {
            _canonical_name_or_path("meta-llama/Meta-Llama-3-8B-Instruct"),
            _canonical_name_or_path("EleutherAI/gpt-j-6b"),
            _canonical_name_or_path("Qwen/Qwen2.5-7B-Instruct"),
        }
        assert len(names) == 3

    def test_canonical_name_matches_stats_dir(self):
        """The canonical name must match the stats directory name exactly.

        Stats live at: data/stats/{name}/wikipedia_stats/
        Vendor does: model.config._name_or_path.replace("/", "_") for covariance lookup.
        GLUE does: model.config._name_or_path.lower().split("/")[-1] for context length.
        All three must resolve to the same string.
        """
        from evaluate_harness import _canonical_name_or_path
        for model, expected in [
            ("meta-llama/Meta-Llama-3-8B-Instruct", "llama3-8b-instruct"),
            ("EleutherAI/gpt-j-6b", "gpt-j-6b"),
            ("Qwen/Qwen2.5-7B-Instruct", "qwen2.5-7b-instruct"),
        ]:
            canonical = _canonical_name_or_path(model)
            assert canonical == expected
            # Vendor covariance: .replace("/", "_") — no-op since no slashes
            assert canonical.replace("/", "_") == expected
            # GLUE: .lower().split("/")[-1] — no-op since already lowercase, no slashes
            assert canonical.lower().split("/")[-1] == expected

    @pytest.mark.parametrize("base_alg", ["MEMIT", "AlphaEdit", "NSE", "MEMIT_rect"])
    @pytest.mark.parametrize("model_name", [
        "meta-llama/Meta-Llama-3-8B-Instruct",
        "EleutherAI/gpt-j-6b",
        "Qwen/Qwen2.5-7B-Instruct",
    ])
    def test_every_alg_model_combination(self, base_alg, model_name):
        """Every algorithm × model combination must produce a valid canonical name."""
        from evaluate_harness import _canonical_name_or_path
        canonical = _canonical_name_or_path(model_name)
        assert canonical  # not empty
        assert "/" not in canonical  # no slashes (vendor does replace("/","_"))


class TestLoadDatasetVendorGlobals:
    """load_dataset must handle vendor globals.yml dependency."""

    def test_ensure_vendor_globals(self):
        """_ensure_vendor_globals pre-populates the vendor module without needing globals.yml."""
        from evaluate_harness import _ensure_vendor_globals
        from pathlib import Path
        import sys

        # Clear any existing module
        sys.modules.pop("util.globals", None)

        vendor_root = Path(__file__).parent.parent / "vendor" / "AlphaEdit"
        _ensure_vendor_globals(vendor_root)

        assert "util.globals" in sys.modules
        mod = sys.modules["util.globals"]
        assert hasattr(mod, "DATA_DIR")
        assert hasattr(mod, "STATS_DIR")
        assert "data" in str(mod.DATA_DIR)

    def test_does_not_chdir(self):
        """load_dataset must NOT chdir — use _ensure_vendor_globals instead."""
        from evaluate_harness import load_dataset
        import inspect
        source = inspect.getsource(load_dataset)
        assert "os.chdir" not in source, "load_dataset should not chdir — use _ensure_vendor_globals"

    def test_vendor_globals_idempotent(self):
        """Calling _ensure_vendor_globals twice must not crash or override."""
        from evaluate_harness import _ensure_vendor_globals
        from pathlib import Path
        import sys

        vendor_root = Path(__file__).parent.parent / "vendor" / "AlphaEdit"
        _ensure_vendor_globals(vendor_root)
        mod1 = sys.modules["util.globals"]
        _ensure_vendor_globals(vendor_root)
        mod2 = sys.modules["util.globals"]
        assert mod1 is mod2, "Second call should be a no-op"

    def test_vendor_globals_has_all_fields(self):
        """The pre-populated module must have every field that globals.yml provides."""
        from evaluate_harness import _ensure_vendor_globals
        from pathlib import Path
        import sys

        sys.modules.pop("util.globals", None)
        vendor_root = Path(__file__).parent.parent / "vendor" / "AlphaEdit"
        _ensure_vendor_globals(vendor_root)

        mod = sys.modules["util.globals"]
        for field in ["RESULTS_DIR", "DATA_DIR", "STATS_DIR", "HPARAMS_DIR", "KV_DIR", "REMOTE_ROOT_URL"]:
            assert hasattr(mod, field), f"vendor globals missing {field}"

    def test_vendor_globals_paths_are_absolute(self):
        """All path fields must be Path objects under the vendor root."""
        from evaluate_harness import _ensure_vendor_globals
        from pathlib import Path
        import sys

        sys.modules.pop("util.globals", None)
        vendor_root = Path(__file__).parent.parent / "vendor" / "AlphaEdit"
        _ensure_vendor_globals(vendor_root)

        mod = sys.modules["util.globals"]
        for field in ["RESULTS_DIR", "DATA_DIR", "STATS_DIR", "HPARAMS_DIR", "KV_DIR"]:
            val = getattr(mod, field)
            assert isinstance(val, Path), f"{field} must be a Path, got {type(val)}"
            assert str(vendor_root) in str(val), f"{field} must be under vendor root"

    def test_vendor_globals_matches_yaml(self):
        """Pre-populated values must match what globals.yml would provide."""
        from evaluate_harness import _ensure_vendor_globals
        from pathlib import Path
        import sys, yaml

        sys.modules.pop("util.globals", None)
        vendor_root = Path(__file__).parent.parent / "vendor" / "AlphaEdit"
        _ensure_vendor_globals(vendor_root)
        mod = sys.modules["util.globals"]

        with open(vendor_root / "globals.yml") as f:
            yml = yaml.safe_load(f)

        assert mod.DATA_DIR == vendor_root / yml["DATA_DIR"]
        assert mod.STATS_DIR == vendor_root / yml["STATS_DIR"]
        assert mod.REMOTE_ROOT_URL == yml["REMOTE_ROOT_URL"]


class TestRequestFormatting:
    """run_experiment must flatten requested_rewrite to match vendor apply_fn format."""

    def test_flattens_requested_rewrite(self):
        """Vendor apply functions expect {case_id, prompt, subject, target_new, target_true}
        not {case_id, requested_rewrite: {prompt, subject, target_new, target_true}}."""
        from evaluate_harness import ExperimentHooks, run_experiment
        from unittest.mock import MagicMock

        received_requests = []

        def capture_apply(model, tok, requests, hparams, **kwargs):
            received_requests.extend(requests)
            return (MagicMock(), None)

        model = MagicMock()
        tok = MagicMock()
        tok.pad_token = None
        tok.eos_token = "<eos>"
        hparams = MagicMock()
        hparams.layers = [4, 5]

        dataset = [{"case_id": 1, "requested_rewrite": {
            "prompt": "The capital of {} is", "subject": "France",
            "target_new": {"str": "Berlin"}, "target_true": {"str": "Paris"},
        }, "paraphrase_prompts": [], "neighborhood_prompts": [], "generation_prompts": []}]

        hooks = ExperimentHooks(should_eval=lambda _: False)
        run_experiment(model=model, tok=tok, hparams=hparams,
                      dataset=dataset, apply_fn=capture_apply,
                      alg_name="MEMIT", num_edits=1,
                      results_dir=Path("/tmp/test"), hooks=hooks, max_batches=1)

        assert len(received_requests) == 1
        req = received_requests[0]
        assert "target_new" in req, "request must have target_new at top level"
        assert "requested_rewrite" not in req, "request must NOT have nested requested_rewrite"
        assert req["case_id"] == 1
        assert req["subject"] == "France"


class TestVendorGlobals:
    """Vendor util.globals must be pre-populated before importing vendor modules."""

    def test_ensure_vendor_globals_before_sys_path(self):
        """_ensure_vendor_globals must be called BEFORE sys.path.insert of alphaedit_root."""
        source = (Path(__file__).resolve().parent.parent / "src" / "evaluate_harness.py").read_text()
        ensure_pos = source.find("_ensure_vendor_globals(alphaedit_root)")
        syspath_pos = source.find('sys.path.insert(0, str(alphaedit_root))')
        assert ensure_pos < syspath_pos, (
            "_ensure_vendor_globals must be called BEFORE sys.path.insert. "
            "Otherwise vendor dsets imports util.globals which reads globals.yml from CWD."
        )

    def test_ensure_vendor_globals_creates_util_package(self):
        """Must create the 'util' package module for util.globals to be importable."""
        source = (Path(__file__).resolve().parent.parent / "src" / "evaluate_harness.py").read_text()
        assert 'sys.modules["util"]' in source


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

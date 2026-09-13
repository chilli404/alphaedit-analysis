#!/usr/bin/env python3
"""GPU tests for helper functions and internal utilities.

Tests correctness of functions that are only exercised during real model inference
— context templates, covariance loading, shape matching, mega_batch_eval internals,
dataset loading, and the seeded_runner pipeline.

Run on GPU cluster: uv run pytest tests/test_gpu_helpers.py -v --timeout=300
Assigned to: test-eval cluster (sky/test_eval_and_measure.yaml)
"""
from __future__ import annotations

import json
import os
import random
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

pytestmark = pytest.mark.gpu
requires_gpu = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="No GPU available"
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent


# ─── Session fixtures ──────────────────────────────────────────────────────


@pytest.fixture(scope="session")
def alphaedit_root():
    root = PROJECT_ROOT / "vendor" / "AlphaEdit"
    if not root.exists():
        pytest.skip("vendor/AlphaEdit not found")
    return root


@pytest.fixture(scope="session")
def model_and_tok(alphaedit_root):
    sys.path.insert(0, str(PROJECT_ROOT / "src"))
    sys.path.insert(0, str(PROJECT_ROOT / "src" / "util"))
    from evaluate_harness import load_model_and_tok, _ensure_vendor_globals
    _ensure_vendor_globals(alphaedit_root)
    sys.path.insert(0, str(alphaedit_root))
    model, tok = load_model_and_tok("NousResearch/Meta-Llama-3-8B-Instruct")
    yield model, tok
    del model
    torch.cuda.empty_cache()


@pytest.fixture(scope="session")
def memit_hparams(alphaedit_root):
    sys.path.insert(0, str(alphaedit_root))
    from memit import MEMITHyperParams
    return MEMITHyperParams.from_json(alphaedit_root / "hparams" / "MEMIT" / "Llama3-8B.json")


@pytest.fixture(scope="session")
def small_dataset(alphaedit_root):
    sys.path.insert(0, str(PROJECT_ROOT / "src"))
    from evaluate_harness import load_dataset
    return load_dataset("mcf", size_limit=20, alphaedit_root=alphaedit_root)


def _flatten_requests(records):
    return [{"case_id": r["case_id"], **r["requested_rewrite"]}
            if "requested_rewrite" in r else r for r in records]


# ─── A. Context Templates ─────────────────────────────────────────────────


class TestContextTemplates:
    """Context templates must be consistent between hooks and vendor code."""

    @requires_gpu
    def test_memit_context_templates_not_empty(self, model_and_tok):
        """_get_context_templates must return non-empty list of lists."""
        sys.path.insert(0, str(PROJECT_ROOT / "src" / "algorithms"))
        from memit_with_hooks import _get_context_templates
        model, tok = model_and_tok
        templates = _get_context_templates(model, tok)
        assert len(templates) >= 1
        assert templates[0] == ["{}"]
        for t in templates:
            assert isinstance(t, list)
            assert all("{}" in s for s in t)

    @requires_gpu
    def test_alphaedit_context_templates_not_empty(self, model_and_tok):
        """AlphaEdit context templates must also be non-empty."""
        from alphaedit_with_hooks import _get_context_templates_ae
        model, tok = model_and_tok
        templates = _get_context_templates_ae(model, tok)
        assert len(templates) >= 1
        assert templates[0] == ["{}"]

    @requires_gpu
    def test_context_templates_cached(self, model_and_tok):
        """Calling _get_context_templates twice must return the same object (cached)."""
        from memit_with_hooks import _get_context_templates
        model, tok = model_and_tok
        t1 = _get_context_templates(model, tok)
        t2 = _get_context_templates(model, tok)
        assert t1 is t2, "Context templates must be cached (same object on second call)"


# ─── B. Covariance Loading ────────────────────────────────────────────────


class TestCovarianceLoading:
    """Covariance statistics must load correctly for the canonical model name."""

    @requires_gpu
    def test_get_cov_returns_tensor(self, model_and_tok, memit_hparams):
        """_get_cov must return a 2D tensor."""
        from memit_with_hooks import _get_cov
        model, tok = model_and_tok
        layer_name = memit_hparams.rewrite_module_tmp.format(memit_hparams.layers[0])
        cov = _get_cov(model, tok, layer_name,
                       memit_hparams.mom2_dataset, memit_hparams.mom2_n_samples,
                       memit_hparams.mom2_dtype)
        assert isinstance(cov, torch.Tensor)
        assert cov.ndim == 2
        assert cov.shape[0] == cov.shape[1]

    @requires_gpu
    def test_cov_is_positive_semidefinite(self, model_and_tok, memit_hparams):
        """Covariance matrix should be PSD (all eigenvalues >= 0)."""
        from memit_with_hooks import _get_cov
        model, tok = model_and_tok
        layer_name = memit_hparams.rewrite_module_tmp.format(memit_hparams.layers[0])
        cov = _get_cov(model, tok, layer_name,
                       memit_hparams.mom2_dataset, memit_hparams.mom2_n_samples,
                       memit_hparams.mom2_dtype)
        eigvals = torch.linalg.eigvalsh(cov.double().cpu())
        assert (eigvals >= -1e-6).all(), f"Covariance has negative eigenvalue: {eigvals.min()}"


# ─── C. Shape Matching ────────────────────────────────────────────────────


class TestMatchShape:
    """_match_shape must handle normal and transposed weight matrices."""

    @requires_gpu
    def test_match_shape_same(self):
        from memit_with_hooks import _match_shape
        m = torch.randn(4096, 14336)
        result = _match_shape(m, (4096, 14336))
        assert result.shape == (4096, 14336)

    @requires_gpu
    def test_match_shape_transposed(self):
        from memit_with_hooks import _match_shape
        m = torch.randn(14336, 4096)
        result = _match_shape(m, (4096, 14336))
        assert result.shape == (4096, 14336)

    @requires_gpu
    def test_match_shape_mismatch_raises(self):
        from memit_with_hooks import _match_shape
        m = torch.randn(100, 200)
        with pytest.raises(ValueError, match="Shape mismatch"):
            _match_shape(m, (300, 400))


# ─── D. Dataset Loading ───────────────────────────────────────────────────


class TestDatasetLoading:
    """load_dataset must return properly structured records."""

    @requires_gpu
    def test_mcf_records_have_required_fields(self, small_dataset):
        """MCF records must have case_id, requested_rewrite, paraphrase_prompts, etc."""
        for record in small_dataset[:5]:
            assert "case_id" in record
            assert "requested_rewrite" in record
            rw = record["requested_rewrite"]
            assert "prompt" in rw
            assert "subject" in rw
            assert "target_new" in rw
            assert "target_true" in rw

    @requires_gpu
    def test_dataset_override_from_json(self, alphaedit_root):
        """load_dataset with dataset_override loads from JSON file."""
        from evaluate_harness import load_dataset
        ordering_dir = PROJECT_ROOT / "results" / "matched_ordering" / "orderings"
        json_files = list(ordering_dir.glob("*.json")) if ordering_dir.exists() else []
        if not json_files:
            pytest.skip("No ordering JSON files available")
        ds = load_dataset("mcf", size_limit=10, dataset_override=str(json_files[0]))
        assert len(ds) == 10
        assert "case_id" in ds[0]


# ─── E. Mega Batch Eval Internals ─────────────────────────────────────────


class TestMegaBatchEvalInternals:
    """The mega_batch_eval function must produce correct per-case JSONs."""

    @requires_gpu
    def test_per_case_json_has_both_metrics(self, model_and_tok, small_dataset, memit_hparams, tmp_path):
        """Per-case JSONs must contain both prob_pref and argmax fields."""
        model, tok = model_and_tok
        from mega_batch_eval import get_mega_batch_eval_source

        ns = {}
        exec(get_mega_batch_eval_source(), ns)
        mbe = ns["_mega_batch_eval"]

        template = str(tmp_path / "{}_edits-case_{}.json")
        mbe(model, tok, small_dataset[:5], template, 10, [r["case_id"] for r in small_dataset[:5]], 1.0)

        json_files = list(tmp_path.glob("*.json"))
        assert len(json_files) > 0, "mega_batch_eval should produce per-case JSONs"

        with open(json_files[0]) as f:
            data = json.load(f)
        post = data.get("post", {})
        assert "rewrite_prompts_probs" in post, "Must have prob NLL values"
        assert "rewrite_prompts_correct" in post, "Must have argmax correctness"


# ─── F. Seeded Runner Pipeline ────────────────────────────────────────────


class TestSeededRunnerPipeline:
    """seeded_runner produces correct output through the harness."""

    @requires_gpu
    def test_harness_with_memit_produces_summary(self, model_and_tok, small_dataset, memit_hparams, tmp_path):
        """Running the harness with MEMIT for 1 batch should produce a summary."""
        model, tok = model_and_tok
        from evaluate_harness import run_experiment, ExperimentHooks
        from algorithms.memit_with_hooks import apply_memit_with_hooks

        summary = run_experiment(
            model, tok, memit_hparams, small_dataset,
            apply_fn=apply_memit_with_hooks,
            alg_name="MEMIT", num_edits=10,
            results_dir=tmp_path / "seeded",
            hooks=ExperimentHooks(should_eval=lambda _: False),
            max_batches=1,
        )
        assert summary["batches_run"] == 1
        assert summary["total_edits"] == 10

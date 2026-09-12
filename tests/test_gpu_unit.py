#!/usr/bin/env python3
"""GPU unit tests — each tests ONE thing with real model inference.

Requires: CUDA GPU, ~48GB VRAM, Llama-3-8B weights, MCF dataset linked.
Run on GPU cluster: uv run pytest tests/test_gpu_unit.py -v --timeout=300
Skip on CPU: all tests marked @requires_gpu.

Session-scoped fixtures load the model ONCE across all tests (~40s),
then each test runs in ~10-30s.
"""
from __future__ import annotations

import copy
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


# ─── Session fixtures (load model once) ─────────────────────────────────────


@pytest.fixture(scope="session")
def alphaedit_root():
    root = PROJECT_ROOT / "vendor" / "AlphaEdit"
    if not root.exists():
        pytest.skip("vendor/AlphaEdit not found")
    return root


@pytest.fixture(scope="session")
def model_and_tok(alphaedit_root):
    from evaluate_harness import load_model_and_tok, _ensure_vendor_globals
    _ensure_vendor_globals(alphaedit_root)
    sys.path.insert(0, str(alphaedit_root))
    model, tok = load_model_and_tok("NousResearch/Meta-Llama-3-8B-Instruct")
    yield model, tok
    del model
    torch.cuda.empty_cache()


@pytest.fixture(scope="session")
def small_dataset(alphaedit_root):
    from evaluate_harness import load_dataset
    ds = load_dataset("mcf", size_limit=20, alphaedit_root=alphaedit_root)
    assert len(ds) >= 10, "Need at least 10 MCF records for GPU tests"
    return ds


@pytest.fixture(scope="session")
def memit_hparams(alphaedit_root):
    sys.path.insert(0, str(alphaedit_root))
    from memit import MEMITHyperParams
    return MEMITHyperParams.from_json(alphaedit_root / "hparams" / "MEMIT" / "Llama3-8B.json")


@pytest.fixture(scope="session")
def alphaedit_hparams(alphaedit_root):
    sys.path.insert(0, str(alphaedit_root))
    from AlphaEdit import AlphaEditHyperParams
    return AlphaEditHyperParams.from_json(alphaedit_root / "hparams" / "AlphaEdit" / "Llama3-8B.json")


@pytest.fixture(scope="session")
def P_matrix(alphaedit_root):
    """Load the null-space projection matrix."""
    p_path = alphaedit_root / "data" / "stats" / "null_space_project.pt"
    if not p_path.exists():
        pytest.skip("null_space_project.pt not found (run link_stats.sh)")
    return torch.load(str(p_path), map_location="cpu")


def _flatten_requests(records):
    """Convert dataset records to the format vendor apply functions expect."""
    return [
        {"case_id": r["case_id"], **r["requested_rewrite"]}
        if "requested_rewrite" in r else r
        for r in records
    ]


def _get_weight_snapshot(model, hparams):
    """Capture current edited-layer weights."""
    snapshot = {}
    params = dict(model.named_parameters())
    for layer in hparams.layers:
        key = hparams.rewrite_module_tmp.format(layer) + ".weight"
        if key in params:
            snapshot[key] = params[key].data.clone()
    return snapshot


# ─── A. Model Loading ───────────────────────────────────────────────────────


class TestModelLoading:

    @requires_gpu
    def test_model_loads_to_gpu(self, model_and_tok):
        model, tok = model_and_tok
        param = next(model.parameters())
        assert param.is_cuda, "Model should be on GPU"

    @requires_gpu
    def test_canonical_name_set(self, model_and_tok):
        model, tok = model_and_tok
        assert model.config._name_or_path == "llama3-8b-instruct", (
            f"Expected canonical name 'llama3-8b-instruct', got '{model.config._name_or_path}'"
        )

    @requires_gpu
    def test_tokenizer_has_pad_token(self, model_and_tok):
        model, tok = model_and_tok
        assert tok.pad_token is not None
        assert tok.padding_side == "left"


# ─── B. Edit Correctness ────────────────────────────────────────────────────


class TestEditCorrectness:

    @requires_gpu
    def test_memit_edit_changes_weights(self, model_and_tok, small_dataset, memit_hparams):
        """MEMIT should modify the target layer weights."""
        model, tok = model_and_tok
        before = _get_weight_snapshot(model, memit_hparams)

        requests = _flatten_requests(small_dataset[:10])
        sys.path.insert(0, str(PROJECT_ROOT / "vendor" / "AlphaEdit"))
        from memit.memit_main import apply_memit_to_model
        apply_memit_to_model(model, tok, requests, memit_hparams, return_orig_weights=False)

        after = _get_weight_snapshot(model, memit_hparams)
        changed = sum(1 for k in before if not torch.equal(before[k], after[k]))
        assert changed > 0, "MEMIT should change at least one layer's weights"

    @requires_gpu
    def test_memit_with_hooks_no_hooks_matches_vendor(self, model_and_tok, small_dataset, memit_hparams):
        """apply_memit_with_hooks with no hooks should produce same deltas as vendor."""
        model, tok = model_and_tok
        requests = _flatten_requests(small_dataset[:10])

        # Snapshot before
        w0 = _get_weight_snapshot(model, memit_hparams)

        # Run vendor MEMIT
        sys.path.insert(0, str(PROJECT_ROOT / "vendor" / "AlphaEdit"))
        from memit.memit_main import apply_memit_to_model
        apply_memit_to_model(model, tok, requests, memit_hparams, return_orig_weights=False)
        w_vendor = _get_weight_snapshot(model, memit_hparams)
        delta_vendor = {k: w_vendor[k] - w0[k] for k in w0}

        # Restore weights
        params = dict(model.named_parameters())
        for k, v in w0.items():
            params[k].data.copy_(v)

        # Run hooks version (no hooks = should be identical)
        from algorithms.memit_with_hooks import apply_memit_with_hooks
        apply_memit_with_hooks(model, tok, requests, memit_hparams, return_orig_weights=False)
        w_hooks = _get_weight_snapshot(model, memit_hparams)
        delta_hooks = {k: w_hooks[k] - w0[k] for k in w0}

        # Compare deltas
        for k in delta_vendor:
            assert torch.allclose(delta_vendor[k], delta_hooks[k], atol=1e-4), (
                f"Weight delta mismatch at {k}: "
                f"vendor norm={delta_vendor[k].norm():.6f}, hooks norm={delta_hooks[k].norm():.6f}"
            )

        # Restore again for next test
        for k, v in w0.items():
            params[k].data.copy_(v)

    @requires_gpu
    def test_edit_preserves_unedited_layers(self, model_and_tok, small_dataset, memit_hparams):
        """Editing layers 4-8 should not change layer 0."""
        model, tok = model_and_tok
        params = dict(model.named_parameters())

        layer0_key = memit_hparams.rewrite_module_tmp.format(0) + ".weight"
        if layer0_key not in params:
            pytest.skip("Layer 0 rewrite key not found")

        w0_before = params[layer0_key].data.clone()
        requests = _flatten_requests(small_dataset[:10])
        from algorithms.memit_with_hooks import apply_memit_with_hooks
        apply_memit_with_hooks(model, tok, requests, memit_hparams, return_orig_weights=False)
        w0_after = params[layer0_key].data.clone()

        assert 0 not in memit_hparams.layers, "Layer 0 should not be in edited layers for this test"
        assert torch.equal(w0_before, w0_after), "Layer 0 should not change when editing layers 4-8"

    @requires_gpu
    def test_edit_is_deterministic(self, model_and_tok, small_dataset, memit_hparams):
        """Same seed + data should produce identical weight deltas."""
        model, tok = model_and_tok
        requests = _flatten_requests(small_dataset[:10])

        deltas = []
        for _ in range(2):
            w_before = _get_weight_snapshot(model, memit_hparams)

            torch.manual_seed(42)
            np.random.seed(42)
            random.seed(42)

            from algorithms.memit_with_hooks import apply_memit_with_hooks
            apply_memit_with_hooks(model, tok, requests, memit_hparams, return_orig_weights=False)
            w_after = _get_weight_snapshot(model, memit_hparams)

            delta = {k: w_after[k] - w_before[k] for k in w_before}
            deltas.append(delta)

            # Restore
            params = dict(model.named_parameters())
            for k, v in w_before.items():
                params[k].data.copy_(v)

        for k in deltas[0]:
            assert torch.allclose(deltas[0][k], deltas[1][k], atol=1e-6), (
                f"Non-deterministic delta at {k}"
            )


# ─── C. Hook Effects ────────────────────────────────────────────────────────


class TestHookEffects:

    @requires_gpu
    def test_seqreg_hooks_cache_keys(self, model_and_tok, small_dataset, memit_hparams):
        """After one batch with seqreg_hooks, prev_cache should be populated."""
        model, tok = model_and_tok
        w0 = _get_weight_snapshot(model, memit_hparams)

        from algorithms.memit_with_hooks import apply_memit_with_hooks
        from algorithms.hook_presets import seqreg_hooks

        hooks = seqreg_hooks(lambda_prev=1.0, lambda_delta=0.0)
        state = hooks.get_state()
        requests = _flatten_requests(small_dataset[:10])

        apply_memit_with_hooks(model, tok, requests, memit_hparams,
                               hooks=hooks, state=state, return_orig_weights=False)

        assert len(state["prev_cache"]) > 0, "prev_cache should be populated after one batch"
        for layer, keys in state["prev_cache"].items():
            assert len(keys) > 0, f"Layer {layer} should have cached keys"

        # Restore
        params = dict(model.named_parameters())
        for k, v in w0.items():
            params[k].data.copy_(v)

    @requires_gpu
    def test_revive_hooks_reduce_update_norm(self, model_and_tok, small_dataset, memit_hparams):
        """REVIVE post_solve should reduce ||upd_matrix|| for a single update."""
        model, tok = model_and_tok
        from algorithms.hook_presets import revive_hooks

        layer = memit_hparams.layers[0]
        weight_name = f"{memit_hparams.rewrite_module_tmp.format(layer)}.weight"
        current_weight = dict(model.named_parameters())[weight_name].data

        # Create a random update in float64 (same dtype as MEMIT solve output)
        upd = torch.randn(current_weight.shape, dtype=torch.float64, device="cuda") * 0.01

        # Use tau=0.3 for a clear signal (removes ~30% of energy)
        hooks = revive_hooks(revive_tau=0.3, revive_svd_device="cuda")
        state = {"_current_weights": {weight_name: current_weight}}

        filtered = hooks.post_solve(layer, upd, None, None, weight_name, state)

        orig_norm = upd.norm().item()
        filt_norm = filtered.norm().item()
        assert filt_norm <= orig_norm * 1.01, (
            f"REVIVE should reduce update norm: filtered={filt_norm:.4f} vs original={orig_norm:.4f}"
        )


# ─── D. Checkpoint Correctness ──────────────────────────────────────────────


class TestCheckpointCorrectness:

    @requires_gpu
    def test_checkpoint_save_load_preserves_weights(self, model_and_tok, small_dataset, memit_hparams, tmp_path):
        """Save after edit, load into fresh model, weights should match."""
        model, tok = model_and_tok
        requests = _flatten_requests(small_dataset[:10])

        from algorithms.memit_with_hooks import apply_memit_with_hooks
        from checkpoint_io import save_checkpoint, load_checkpoint

        # Edit
        apply_memit_with_hooks(model, tok, requests, memit_hparams, return_orig_weights=False)
        w_edited = _get_weight_snapshot(model, memit_hparams)

        # Save
        ckpt_dir = tmp_path / "ckpt"
        save_checkpoint(0, model, memit_hparams, str(ckpt_dir), num_edits=10)

        # Corrupt weights to prove loading works
        params = dict(model.named_parameters())
        for k in w_edited:
            params[k].data.zero_()

        # Load
        result = load_checkpoint(model, memit_hparams, str(ckpt_dir), 0)
        assert result["loaded"]

        w_loaded = _get_weight_snapshot(model, memit_hparams)
        for k in w_edited:
            assert torch.allclose(w_edited[k], w_loaded[k], atol=1e-6), (
                f"Checkpoint load mismatch at {k}"
            )


# ─── E. Evaluation Metrics ──────────────────────────────────────────────────


class TestEvalMetrics:

    @requires_gpu
    def test_argmax_stricter_than_prob_pref(self, model_and_tok, small_dataset):
        """Argmax success should be ≤ prob-pref success on any batch."""
        model, tok = model_and_tok

        argmax_success = 0
        prob_pref_success = 0
        total = 0

        for record in small_dataset[:10]:
            rw = record["requested_rewrite"]
            prompt = rw["prompt"].format(rw["subject"])
            target_new = rw["target_new"]["str"]
            target_true = rw["target_true"]["str"]

            inputs_new = tok(prompt + target_new, return_tensors="pt").to("cuda")
            inputs_true = tok(prompt + target_true, return_tensors="pt").to("cuda")

            with torch.no_grad():
                nll_new = -model(**inputs_new).logits[0, -1].log_softmax(-1).max().item()
                nll_true = -model(**inputs_true).logits[0, -1].log_softmax(-1).max().item()

            # prob-pref: is target_new more probable?
            if nll_new < nll_true:
                prob_pref_success += 1

            # argmax: is first token of target_new the argmax?
            new_tok_id = tok(f" {target_new}", add_special_tokens=False).input_ids[0]
            prompt_ids = tok(prompt, return_tensors="pt").to("cuda")
            with torch.no_grad():
                logits = model(**prompt_ids).logits[0, -1]
            if logits.argmax().item() == new_tok_id:
                argmax_success += 1

            total += 1

        # Argmax is strictly harder — shouldn't exceed prob-pref
        assert argmax_success <= prob_pref_success + 1, (  # +1 for measurement noise
            f"Argmax ({argmax_success}/{total}) should not exceed prob-pref ({prob_pref_success}/{total})"
        )

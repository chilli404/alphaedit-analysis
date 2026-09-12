#!/usr/bin/env python3
"""Tests for the per-layer algorithm hooks system.

Verifies hooks API, wrapper function structure, preset configurations,
hook composition, and that every legacy runner maps to a preset.

No GPU needed — tests structure, imports, and hook logic with mock tensors.
"""
import ast
import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent


class TestHooksImport:
    def test_import_hooks(self):
        from algorithms.hooks import AlgorithmHooks

    def test_import_compose(self):
        from algorithms.hooks import compose_hooks

    def test_import_memit_wrapper(self):
        from algorithms.memit_with_hooks import apply_memit_with_hooks

    def test_import_alphaedit_wrapper(self):
        from algorithms.alphaedit_with_hooks import apply_alphaedit_with_hooks

    def test_import_presets(self):
        from algorithms.hook_presets import seqreg_hooks, revive_hooks, pathguard_hooks, polykernel_hooks


class TestAlgorithmHooks:
    def test_default_none(self):
        from algorithms.hooks import AlgorithmHooks
        h = AlgorithmHooks()
        assert h.build_lhs is None
        assert h.post_solve is None
        assert h.post_update is None
        assert h.init_state is None

    def test_get_state_default(self):
        from algorithms.hooks import AlgorithmHooks
        h = AlgorithmHooks()
        assert h.get_state() == {}

    def test_get_state_custom(self):
        from algorithms.hooks import AlgorithmHooks
        h = AlgorithmHooks(init_state=lambda: {"foo": 42})
        assert h.get_state() == {"foo": 42}

    def test_accepts_callables(self):
        from algorithms.hooks import AlgorithmHooks
        h = AlgorithmHooks(
            build_lhs=lambda *a: None,
            post_solve=lambda *a: a[1],
            post_update=lambda *a: None,
        )
        assert h.build_lhs is not None
        assert h.post_solve is not None


class TestComposeHooks:
    def test_compose_empty(self):
        from algorithms.hooks import AlgorithmHooks, compose_hooks
        h = compose_hooks(AlgorithmHooks(), AlgorithmHooks())
        assert h.build_lhs is None
        assert h.post_solve is None

    def test_compose_post_solve_chains(self):
        from algorithms.hooks import AlgorithmHooks, compose_hooks
        h1 = AlgorithmHooks(post_solve=lambda li, u, *a: u * 2)
        h2 = AlgorithmHooks(post_solve=lambda li, u, *a: u + 1)
        composed = compose_hooks(h1, h2)
        result = composed.post_solve(0, torch.tensor(3.0), None, None, "", {})
        assert result == torch.tensor(7.0)  # (3*2) + 1

    def test_compose_init_state_merges(self):
        from algorithms.hooks import AlgorithmHooks, compose_hooks
        h1 = AlgorithmHooks(init_state=lambda: {"a": 1})
        h2 = AlgorithmHooks(init_state=lambda: {"b": 2})
        composed = compose_hooks(h1, h2)
        state = composed.get_state()
        assert state == {"a": 1, "b": 2}

    def test_compose_post_update_calls_all(self):
        from algorithms.hooks import AlgorithmHooks, compose_hooks
        calls = []
        h1 = AlgorithmHooks(post_update=lambda *a: calls.append("h1"))
        h2 = AlgorithmHooks(post_update=lambda *a: calls.append("h2"))
        composed = compose_hooks(h1, h2)
        composed.post_update(0, "w", torch.zeros(1), {})
        assert calls == ["h1", "h2"]


class TestWrapperStructure:
    """Verify wrapper files parse and have correct structure."""

    def test_memit_wrapper_parses(self):
        source = (PROJECT_ROOT / "src" / "algorithms" / "memit_with_hooks.py").read_text()
        ast.parse(source)

    def test_alphaedit_wrapper_parses(self):
        source = (PROJECT_ROOT / "src" / "algorithms" / "alphaedit_with_hooks.py").read_text()
        ast.parse(source)

    def test_memit_wrapper_has_hook_calls(self):
        source = (PROJECT_ROOT / "src" / "algorithms" / "memit_with_hooks.py").read_text()
        assert "hooks.build_lhs" in source
        assert "hooks.post_solve" in source
        assert "hooks.post_update" in source

    def test_alphaedit_wrapper_has_hook_calls(self):
        source = (PROJECT_ROOT / "src" / "algorithms" / "alphaedit_with_hooks.py").read_text()
        assert "hooks.build_lhs" in source
        assert "hooks.post_solve" in source
        assert "hooks.post_update" in source

    def test_memit_wrapper_imports_vendor(self):
        source = (PROJECT_ROOT / "src" / "algorithms" / "memit_with_hooks.py").read_text()
        assert "compute_ks" in source
        assert "compute_z" in source
        assert "nethook" in source

    def test_alphaedit_wrapper_handles_cache_c(self):
        source = (PROJECT_ROOT / "src" / "algorithms" / "alphaedit_with_hooks.py").read_text()
        assert "cache_c" in source
        assert "P" in source

    def test_memit_wrapper_cleans_gpu_memory(self):
        source = (PROJECT_ROOT / "src" / "algorithms" / "memit_with_hooks.py").read_text()
        assert "empty_cache()" in source

    def test_memit_wrapper_restores_weights(self):
        """MEMIT must restore original weights after computing deltas."""
        source = (PROJECT_ROOT / "src" / "algorithms" / "memit_with_hooks.py").read_text()
        assert "weights_copy" in source

    def test_alphaedit_wrapper_updates_cache_c(self):
        """AlphaEdit must update cache_c after editing."""
        source = (PROJECT_ROOT / "src" / "algorithms" / "alphaedit_with_hooks.py").read_text()
        assert "cache_c[i" in source


class TestHookPresets:
    def test_seqreg_returns_hooks(self):
        from algorithms.hook_presets import seqreg_hooks
        h = seqreg_hooks(lambda_prev=1.0, lambda_delta=0.0)
        assert h.build_lhs is not None
        assert h.post_solve is not None
        assert h.init_state is not None

    def test_revive_returns_hooks(self):
        from algorithms.hook_presets import revive_hooks
        h = revive_hooks(revive_tau=0.1)
        assert h.post_solve is not None
        assert h.build_lhs is None  # REVIVE doesn't modify LHS

    def test_pathguard_returns_hooks(self):
        from algorithms.hook_presets import pathguard_hooks
        h = pathguard_hooks()
        assert h.post_update is not None
        assert h.init_state is not None

    def test_polykernel_returns_hooks(self):
        from algorithms.hook_presets import polykernel_hooks
        h = polykernel_hooks(kernel_degree=2)
        assert h.build_lhs is not None

    def test_seqreg_state_has_prev_cache(self):
        from algorithms.hook_presets import seqreg_hooks
        h = seqreg_hooks()
        state = h.get_state()
        assert "prev_cache" in state
        assert "mechanism_log" in state
        assert "batch_idx" in state

    def test_revive_plus_seqreg_composable(self):
        from algorithms.hook_presets import seqreg_hooks, revive_hooks
        from algorithms.hooks import compose_hooks
        h = compose_hooks(seqreg_hooks(), revive_hooks())
        assert h.build_lhs is not None  # from seqreg
        assert h.post_solve is not None  # from both (chained)

    def test_runner_presets_map_exists(self):
        from algorithms.hook_presets import RUNNER_PRESETS
        assert "memit_sequential_runner" in RUNNER_PRESETS
        assert "polykernel_seqreg_runner" in RUNNER_PRESETS
        assert "pathguard_runner" in RUNNER_PRESETS
        assert "polykernel_editor_runner" in RUNNER_PRESETS


class TestSeqRegBuildLhs:
    """Test the seqreg build_lhs hook with mock tensors."""

    def test_adds_kprev_regularization(self):
        from algorithms.hook_presets import seqreg_hooks
        h = seqreg_hooks(lambda_prev=1.0, lambda_delta=0.0)
        state = h.get_state()

        d = 4
        layer_ks = torch.randn(d, 3, dtype=torch.float64)
        cov = torch.eye(d, dtype=torch.float32)
        hparams = MagicMock()
        hparams.mom2_update_weight = 1.0

        # No prev_cache yet — should return base LHS
        lhs = h.build_lhs(0, layer_ks, cov, hparams, state)
        assert lhs.shape == (d, d)

        # Add some prev keys
        state["prev_cache"][0] = [torch.randn(d, 5)]
        lhs_with_prev = h.build_lhs(0, layer_ks, cov, hparams, state)
        # LHS should be larger (more regularization)
        assert torch.linalg.norm(lhs_with_prev) > torch.linalg.norm(lhs)

    def test_lambda_delta_adds_ridge(self):
        from algorithms.hook_presets import seqreg_hooks
        h = seqreg_hooks(lambda_prev=0.0, lambda_delta=1.0)
        state = h.get_state()

        d = 4
        layer_ks = torch.randn(d, 3, dtype=torch.float64)
        cov = torch.zeros(d, d, dtype=torch.float32)  # zero cov
        hparams = MagicMock()
        hparams.mom2_update_weight = 0.0

        lhs = h.build_lhs(0, layer_ks, cov, hparams, state)
        # Should have K@K^T + lambda_delta*I
        diag = lhs.diag()
        assert (diag > 0).all()  # ridge ensures positive diagonal


class TestRevivePostSolve:
    """Test REVIVE spectral filter with small mock matrices."""

    def test_filters_update(self):
        from algorithms.hook_presets import revive_hooks
        h = revive_hooks(revive_tau=0.5, revive_svd_device="cpu")
        state = {
            "_current_weights": {
                "test.weight": torch.randn(8, 16)
            }
        }
        upd = torch.randn(8, 16)
        filtered = h.post_solve(0, upd, None, None, "test.weight", state)
        # Filtered should have smaller norm (some components removed)
        assert torch.linalg.norm(filtered) <= torch.linalg.norm(upd) * 1.01

    def test_no_filter_without_weight(self):
        from algorithms.hook_presets import revive_hooks
        h = revive_hooks()
        state = {}  # no _current_weights
        upd = torch.randn(4, 8)
        result = h.post_solve(0, upd, None, None, "test.weight", state)
        assert torch.equal(result, upd)


class TestPolykernelBuildLhs:
    """Test polykernel kernel-weighted LHS."""

    def test_kernel_modifies_lhs(self):
        from algorithms.hook_presets import polykernel_hooks
        h = polykernel_hooks(kernel_degree=2)
        state = {}

        d = 4
        layer_ks = torch.randn(d, 3, dtype=torch.float64)
        cov = torch.eye(d, dtype=torch.float32)
        hparams = MagicMock()
        hparams.mom2_update_weight = 1.0

        lhs = h.build_lhs(0, layer_ks, cov, hparams, state)
        # Should differ from standard K@K^T
        standard = 1.0 * cov.double() + layer_ks @ layer_ks.T
        assert not torch.allclose(lhs, standard, atol=1e-6)

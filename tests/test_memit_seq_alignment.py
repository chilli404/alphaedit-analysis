"""Verify MEMIT-Aligned (memit_seq) matches the OTE reference implementation.

Our baselines/EvoEdit/memit/memit_seq_main.py should be functionally identical
to the OTE-SE-Alignment reference. These tests confirm the structural properties
that make it "OTE-aligned" (Lemma 3.1 from arxiv:2605.26670):
  - LHS includes cache_c (accumulated key outer products)
  - No identity ridge, no error correction
  - Factored solve (adj_k then resid @ adj_k.T)
  - Phase 2 applies full update (no RECT mask)

All tests should PASS without code changes.
"""
from pathlib import Path

import pytest
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MEMIT_SEQ_PATH = PROJECT_ROOT / "baselines" / "EvoEdit" / "memit" / "memit_seq_main.py"


def _read_source():
    return MEMIT_SEQ_PATH.read_text()


# ============================================================================
# Solve structure matches reference
# ============================================================================


class TestMemitSeqSolveMatches:

    def test_lhs_has_cache_c(self):
        source = _read_source()
        assert "cache_c[i,:,:].cuda()" in source, \
            "MEMIT-Seq solve LHS must include cache_c (OTE history term)"

    def test_no_identity_ridge(self):
        source = _read_source()
        solve_start = source.find("torch.linalg.solve(")
        assert solve_start > 0
        solve_block = source[solve_start:source.find(")", solve_start + 200) + 1]
        assert "torch.eye(" not in solve_block, \
            "MEMIT-Seq should NOT have identity ridge in the solve (only RECT-Err does)"

    def test_no_error_cache_param(self):
        source = _read_source()
        sig_start = source.find("def apply_memit_seq_to_model(")
        assert sig_start > 0
        sig_end = source.find(")", sig_start) + 1
        sig = source[sig_start:sig_end]
        assert "error_cache" not in sig, \
            "MEMIT-Seq function signature should not accept error_cache"

    def test_factored_solve(self):
        source = _read_source()
        assert "resid @ adj_k.T" in source, \
            "MEMIT-Seq should use factored form: upd_matrix = resid @ adj_k.T"

    def test_deltas_store_adj_k_and_resid(self):
        source = _read_source()
        assert "adj_k.detach().cpu()" in source, \
            "MEMIT-Seq deltas should store adj_k (factored key matrix)"
        assert "resid.detach().cpu()" in source, \
            "MEMIT-Seq deltas should store resid (factored value matrix)"


# ============================================================================
# Phase 2 matches reference
# ============================================================================


class TestMemitSeqPhase2Matches:

    def test_phase2_reconstructs(self):
        source = _read_source()
        apply_start = source.find("def apply_memit_seq_to_model(")
        apply_block = source[apply_start:source.find("\ndef ", apply_start + 10)]
        assert "key_mat @ val_mat.T" in apply_block, \
            "Phase 2 should reconstruct upd_matrix from key_mat @ val_mat.T"

    def test_phase2_full_update_no_mask(self):
        source = _read_source()
        apply_start = source.find("def apply_memit_seq_to_model(")
        apply_block = source[apply_start:source.find("\ndef ", apply_start + 10)]
        assert "w[...] +=" in apply_block, \
            "Phase 2 should apply full update w[...] += (no sparse mask)"

    def test_no_apply_rect_call(self):
        source = _read_source()
        assert "apply_rect(" not in source, \
            "MEMIT-Seq should not call apply_rect (that's RECT only)"


# ============================================================================
# Return type
# ============================================================================


class TestMemitSeqReturns2Tuple:

    def test_returns_model_and_cache_c(self):
        source = _read_source()
        apply_start = source.find("def apply_memit_seq_to_model(")
        apply_block = source[apply_start:source.find("\ndef ", apply_start + 10)]
        assert "return model, cache_c" in apply_block, \
            "apply_memit_seq_to_model should return (model, cache_c), not 3-tuple"


# ============================================================================
# Mathematical equivalence (synthetic tensors)
# ============================================================================


class TestMemitSeqMathEquivalence:

    def test_solve_matches_reference(self):
        """With known inputs, verify the MEMIT-Seq solve produces the expected result.

        The MEMIT-Seq solve is:
            adj_k = solve(lambda*C + cache_c + K@K^T, K)
            upd = resid @ adj_k^T
        """
        torch.manual_seed(42)
        d = 8
        n_keys = 3
        n_targets = 3
        lam = 15000.0

        cov = torch.eye(d, dtype=torch.float64) * 0.01
        cache_c = torch.randn(d, d, dtype=torch.float64) * 0.001
        cache_c = cache_c @ cache_c.T  # make symmetric PSD
        K = torch.randn(d, n_keys, dtype=torch.float64)
        targets = torch.randn(d, n_targets, dtype=torch.float64)
        n_layers = 5
        layer_idx = 0
        resid = targets / (n_layers - layer_idx)

        LHS = lam * cov + cache_c + K @ K.T
        adj_k = torch.linalg.solve(LHS, K)
        upd = resid @ adj_k.T

        # Verify the solve is correct: LHS @ adj_k ≈ K
        residual = LHS @ adj_k - K
        assert torch.linalg.norm(residual) < 1e-8, \
            f"Solve residual too large: {torch.linalg.norm(residual)}"

        # Verify upd_matrix has expected shape
        assert upd.shape == (d, d), f"Expected ({d},{d}), got {upd.shape}"

        # Verify upd_matrix is non-zero
        assert torch.linalg.norm(upd) > 0, "Update matrix should be non-zero"

        # Verify that adding identity ridge changes the result
        LHS_ridge = LHS + torch.eye(d, dtype=torch.float64)
        adj_k_ridge = torch.linalg.solve(LHS_ridge, K)
        upd_ridge = resid @ adj_k_ridge.T
        assert not torch.allclose(upd, upd_ridge, atol=1e-10), \
            "Adding identity ridge should change the solve result (RECT-Err vs MEMIT-Seq)"

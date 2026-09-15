"""TDD tests for the OTE-aligned RECT-Err implementation.

Verifies that baselines/EvoEdit/memit/memit_seq_rect_err_main.py matches
the reference from 'The Labyrinth and the Thread' (ICML 2026, arxiv:2605.26670).

Source-presence tests are xfail until the file is ported.
Synthetic math tests validate the mathematical properties directly.
"""
import pytest
import torch
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RECT_ERR_PATH = PROJECT_ROOT / "baselines" / "EvoEdit" / "memit" / "memit_seq_rect_err_main.py"
RECT_OLD_PATH = PROJECT_ROOT / "baselines" / "EvoEdit" / "memit" / "memit_seq_rect_main.py"


def _read_rect_err():
    return RECT_ERR_PATH.read_text()


def _read_rect_old():
    return RECT_OLD_PATH.read_text()


# ============================================================================
# Source file existence and structural properties
# ============================================================================

class TestRectErrSourcePresent:

    def test_file_exists(self):
        assert RECT_ERR_PATH.exists(), \
            f"Missing {RECT_ERR_PATH.relative_to(PROJECT_ROOT)}"

    def test_has_torch_eye_in_solve(self):
        source = _read_rect_err()
        assert "torch.eye(" in source, \
            "RECT-Err solve must include identity ridge regularizer (+I)"

    def test_error_cache_replacement_not_accumulation(self):
        source = _read_rect_err()
        assert "error_cache[i,:,:] = error_temp" in source, \
            "RECT-Err must REPLACE error_cache (=), not accumulate (+=)"
        lines = source.splitlines()
        for line in lines:
            stripped = line.strip()
            if "error_cache[i,:,:]" in stripped and "error_temp" in stripped:
                assert "+=" not in stripped, \
                    f"Found accumulation (+=) where replacement (=) expected: {stripped}"

    def test_has_small_delta(self):
        source = _read_rect_err()
        assert "small_delta" in source, \
            "RECT-Err must compute small_delta = (masked - unmasked) update"

    def test_has_direct_upd_matrix_in_deltas(self):
        source = _read_rect_err()
        assert "upd_matrix.detach().cpu()" in source, \
            "RECT-Err deltas must store upd_matrix directly (not adj_k)"


# ============================================================================
# Function signature
# ============================================================================

class TestRectErrFunctionSignature:

    def test_function_name(self):
        source = _read_rect_err()
        assert "def apply_memit_seq_rect_err_to_model(" in source

    def test_accepts_kwargs(self):
        source = _read_rect_err()
        func_start = source.find("def apply_memit_seq_rect_err_to_model(")
        assert func_start > 0
        sig_end = source.find(")", func_start) + 1
        sig = source[func_start:sig_end]
        assert "**_kwargs" in sig or "**kwargs" in sig, \
            f"Function must accept **_kwargs for runner compatibility. Sig: {sig[:200]}"

    def test_accepts_error_cache_param(self):
        source = _read_rect_err()
        func_start = source.find("def apply_memit_seq_rect_err_to_model(")
        sig_end = source.find(")", func_start) + 1
        sig = source[func_start:sig_end]
        assert "error_cache" in sig

    def test_accepts_cache_c_param(self):
        source = _read_rect_err()
        func_start = source.find("def apply_memit_seq_rect_err_to_model(")
        sig_end = source.find(")", func_start) + 1
        sig = source[func_start:sig_end]
        assert "cache_c" in sig


# ============================================================================
# RECT k_percent parameter
# ============================================================================

class TestRectErrKPercent:


    def test_k_percent_is_20(self):
        source = _read_rect_err()
        assert "k_percent=20" in source, \
            "RECT-Err apply_rect must use k_percent=20 (reference value)"

    def test_old_file_had_k_percent_40(self):
        """Guard: the old RECT used k_percent=40 — verify it hasn't changed."""
        if not RECT_OLD_PATH.exists():
            pytest.skip("Old RECT file not present")
        source = _read_rect_old()
        assert "k_percent=40" in source, \
            "Old memit_seq_rect_main.py should have k_percent=40"


# ============================================================================
# Synthetic math: solve structure
# ============================================================================

class TestRectErrSolveStructure:
    """Validate mathematical properties of the OTE-aligned solve.

    These tests verify the math directly with small tensors —
    no file parsing, no GPU, no xfail.
    """

    def test_solve_with_identity_ridge(self):
        """Adding +I to the LHS changes the solution."""
        d = 4
        torch.manual_seed(42)
        C = torch.randn(d, d, dtype=torch.float64)
        C = C @ C.T + 0.1 * torch.eye(d, dtype=torch.float64)
        rhs = torch.randn(d, 3, dtype=torch.float64)

        sol_no_ridge = torch.linalg.solve(C, rhs)
        sol_with_ridge = torch.linalg.solve(C + torch.eye(d, dtype=torch.float64), rhs)

        assert not torch.allclose(sol_no_ridge, sol_with_ridge, atol=1e-6), \
            "Ridge regularizer must change the solution"

    def test_solve_with_error_correction_in_rhs(self):
        """Subtracting error_cache^T from the RHS changes the solution."""
        d = 4
        n_req = 3
        torch.manual_seed(42)
        LHS = torch.randn(d, d, dtype=torch.float64)
        LHS = LHS @ LHS.T + torch.eye(d, dtype=torch.float64)
        K = torch.randn(d, n_req, dtype=torch.float64)
        R = torch.randn(d, n_req, dtype=torch.float64)
        E = torch.randn(d, d, dtype=torch.float64)

        rhs_plain = K @ R.T
        rhs_corrected = K @ R.T - E.T

        sol_plain = torch.linalg.solve(LHS, rhs_plain)
        sol_corrected = torch.linalg.solve(LHS, rhs_corrected)

        assert not torch.allclose(sol_plain, sol_corrected, atol=1e-6), \
            "Error correction in RHS must change the solution"

    def test_direct_solve_vs_factored(self):
        """Direct solve for ΔW differs from factored solve(LHS, K) then R @ adj_k^T.

        Reference (direct): solve(LHS, K @ R^T) → ΔW
        Old code (factored): adj_k = solve(LHS, K); ΔW = R @ adj_k^T

        These produce different results because:
          direct: LHS @ ΔW = K @ R^T  →  ΔW = LHS^{-1} @ K @ R^T
          factored: adj_k = LHS^{-1} @ K; ΔW = R @ adj_k^T = R @ K^T @ LHS^{-T}
        The factored form transposes the inverse, so results differ when LHS ≠ LHS^T
        (though in practice LHS is symmetric, so we test with the identity ridge
        and error correction which make the overall equations different).
        """
        d = 4
        n_req = 3
        torch.manual_seed(42)
        C = torch.randn(d, d, dtype=torch.float64)
        C = C @ C.T + torch.eye(d, dtype=torch.float64)
        K = torch.randn(d, n_req, dtype=torch.float64)
        R = torch.randn(d, n_req, dtype=torch.float64)
        E = torch.randn(d, d, dtype=torch.float64)

        LHS_ref = C + torch.eye(d, dtype=torch.float64)
        rhs_ref = K @ R.T - E.T
        upd_direct = torch.linalg.solve(LHS_ref, rhs_ref)

        LHS_old = C
        adj_k_old = torch.linalg.solve(LHS_old, K)
        upd_factored = R @ adj_k_old.T

        assert not torch.allclose(upd_direct, upd_factored, atol=1e-4), \
            "Direct solve (with ridge + error correction) must differ from factored solve (without)"


# ============================================================================
# Synthetic math: error computation
# ============================================================================

class TestRectErrErrorComputation:

    def test_small_delta_is_masked_minus_full(self):
        """small_delta = mask * upd_matrix - upd_matrix = -(1-mask) * upd_matrix."""
        torch.manual_seed(42)
        upd = torch.randn(4, 4, dtype=torch.float64)
        mask = torch.tensor([
            [True, False, True, False],
            [False, True, False, True],
            [True, True, False, False],
            [False, False, True, True],
        ])
        masked = mask * upd
        small_delta = masked - upd

        expected = (~mask) * (-upd)
        assert torch.allclose(small_delta, expected), \
            "small_delta must equal -(1-mask)*upd"
        assert not torch.allclose(small_delta, torch.zeros_like(small_delta)), \
            "small_delta must be nonzero when mask is partial"

    def test_error_replace_vs_accumulate(self):
        """Replacing error_cache (=) vs accumulating (+=) diverges after 2 rounds."""
        torch.manual_seed(42)
        d = 4

        error_replace = torch.zeros(d, d, dtype=torch.float64)
        error_accum = torch.zeros(d, d, dtype=torch.float64)

        for _ in range(2):
            error_temp = torch.randn(d, d, dtype=torch.float64)
            error_replace = error_temp
            error_accum += error_temp

        assert not torch.allclose(error_replace, error_accum), \
            "Replace (=) and accumulate (+=) must diverge after multiple rounds"


# ============================================================================
# Phase 2 application
# ============================================================================

class TestRectErrPhase2:

    def test_phase2_does_not_reconstruct_from_factors(self):
        """The apply function must NOT do key_mat @ val_mat.T reconstruction."""
        source = _read_rect_err()
        apply_start = source.find("def apply_memit_seq_rect_err_to_model(")
        apply_end = source.find("\ndef ", apply_start + 1)
        if apply_end < 0:
            apply_end = len(source)
        apply_body = source[apply_start:apply_end]
        assert "key_mat @ val_mat.T" not in apply_body, \
            "RECT-Err Phase 2 must NOT reconstruct from (key_mat, val_mat) factors"

    def test_phase2_uses_upd_mat_directly(self):
        """Phase 2 should unpack (upd_mat, res_mat) and use upd_mat directly."""
        source = _read_rect_err()
        apply_start = source.find("def apply_memit_seq_rect_err_to_model(")
        apply_end = source.find("\ndef ", apply_start + 1)
        if apply_end < 0:
            apply_end = len(source)
        apply_body = source[apply_start:apply_end]
        assert "upd_mat" in apply_body or "upd_matrix = upd_mat" in apply_body, \
            "RECT-Err Phase 2 must use upd_mat directly from deltas"

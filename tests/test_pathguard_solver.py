"""
Unit tests for the PathGuard Woodbury solver and hazard selector.

Tests are written BEFORE implementation (TDD). All should fail initially,
then pass once pathguard_solver.py is implemented.

Run with: uv run pytest tests/test_pathguard_solver.py -v
"""

import sys
from pathlib import Path

import pytest
import torch


from mechanism.pathguard_solver import (
    select_vulnerable,
    woodbury_solve,
    compute_displacement,
    adapt_epsilon,
)


class TestSelectVulnerable:
    """Test top-M vulnerable edit selection by cosine exposure."""

    def test_returns_topM_by_cosine(self):
        """Known geometry: one historical key perfectly aligned with batch key."""
        torch.manual_seed(42)
        d = 64
        M = 3

        batch_keys = torch.randn(d, 2)
        batch_keys[:, 0] = torch.randn(d)

        # Historical keys: first one is a copy of batch key 0 (cosine ~1.0)
        hist_keys = torch.randn(d, 10)
        hist_keys[:, 0] = batch_keys[:, 0] + 0.01 * torch.randn(d)

        indices, hazard = select_vulnerable(hist_keys, batch_keys, M)

        assert len(indices) == M
        assert len(hazard) == M
        assert indices[0] == 0, "Most exposed key should be index 0 (nearly identical to batch key)"
        assert hazard[0] > 0.99, f"Top hazard should be ~1.0, got {hazard[0]}"

    def test_hazard_is_max_cosine(self):
        """Hazard weight equals max cosine across batch keys."""
        torch.manual_seed(7)
        d = 32
        M = 5
        n_hist = 20
        n_batch = 4

        hist_keys = torch.randn(d, n_hist)
        batch_keys = torch.randn(d, n_batch)

        indices, hazard = select_vulnerable(hist_keys, batch_keys, M)

        # Manually compute cosines to verify
        hist_normed = hist_keys / torch.clamp(torch.linalg.norm(hist_keys, dim=0, keepdim=True), min=1e-8)
        batch_normed = batch_keys / torch.clamp(torch.linalg.norm(batch_keys, dim=0, keepdim=True), min=1e-8)
        cos_matrix = hist_normed.T @ batch_normed  # [n_hist, n_batch]
        max_cos = cos_matrix.max(dim=1).values  # [n_hist]

        for rank, (idx, h) in enumerate(zip(indices, hazard)):
            assert abs(h - max_cos[idx].item()) < 1e-5, f"Hazard mismatch at rank {rank}"

    def test_sorted_descending(self):
        """Returned hazard weights are sorted in descending order."""
        torch.manual_seed(99)
        d = 32
        M = 8
        hist_keys = torch.randn(d, 30)
        batch_keys = torch.randn(d, 5)

        _, hazard = select_vulnerable(hist_keys, batch_keys, M)

        for i in range(len(hazard) - 1):
            assert hazard[i] >= hazard[i + 1], f"Not sorted at position {i}: {hazard[i]} < {hazard[i+1]}"

    def test_M_larger_than_history(self):
        """When M > n_hist, return all historical edits."""
        torch.manual_seed(42)
        d = 16
        hist_keys = torch.randn(d, 5)
        batch_keys = torch.randn(d, 3)

        indices, hazard = select_vulnerable(hist_keys, batch_keys, M=100)

        assert len(indices) == 5
        assert len(hazard) == 5

    def test_empty_history(self):
        """Empty history returns empty results."""
        d = 16
        hist_keys = torch.empty(d, 0)
        batch_keys = torch.randn(d, 3)

        indices, hazard = select_vulnerable(hist_keys, batch_keys, M=10)

        assert len(indices) == 0
        assert len(hazard) == 0


class TestWoodburySolve:
    """Test Woodbury identity solver for PathGuard's augmented LHS."""

    def test_matches_direct_solve(self):
        """Woodbury formula must match direct solve(A + lambda*G, K) to high precision."""
        torch.manual_seed(42)
        d = 64
        n_new = 5
        M = 10
        lam = 2.0

        # Build a well-conditioned base LHS
        A_half = torch.randn(d, d, dtype=torch.float64)
        A = A_half @ A_half.T + 10.0 * torch.eye(d, dtype=torch.float64)

        K_new = torch.randn(d, n_new, dtype=torch.float64)
        K_S = torch.randn(d, M, dtype=torch.float64)
        h = torch.rand(M, dtype=torch.float64) + 0.1  # positive hazard weights

        # Direct solve
        G = K_S @ torch.diag(h) @ K_S.T
        direct = torch.linalg.solve(A + lam * G, K_new)

        # Woodbury solve
        adj_k, lambda_used, info = woodbury_solve(
            A, K_new, K_S, h,
            lambda_candidates=[lam],
            epsilon_t=float('inf'),
            E_max=float('inf'),
        )

        assert torch.allclose(adj_k, direct, atol=1e-6), (
            f"Woodbury error: max diff = {(adj_k - direct).abs().max().item():.2e}"
        )
        assert lambda_used == lam

    def test_lambda_zero_recovers_base(self):
        """With lambda=0 and infinite budget, should match base solve A x = K."""
        torch.manual_seed(7)
        d = 32
        n_new = 4
        M = 5

        A_half = torch.randn(d, d, dtype=torch.float64)
        A = A_half @ A_half.T + 5.0 * torch.eye(d, dtype=torch.float64)
        K_new = torch.randn(d, n_new, dtype=torch.float64)
        K_S = torch.randn(d, M, dtype=torch.float64)
        h = torch.rand(M, dtype=torch.float64) + 0.1

        base_sol = torch.linalg.solve(A, K_new)

        adj_k, lambda_used, _ = woodbury_solve(
            A, K_new, K_S, h,
            lambda_candidates=[0.0],
            epsilon_t=float('inf'),
            E_max=float('inf'),
        )

        assert torch.allclose(adj_k, base_sol, atol=1e-8)

    def test_selects_smallest_satisfying_lambda(self):
        """Lambda search returns the smallest candidate that satisfies displacement budget."""
        torch.manual_seed(42)
        d = 64
        n_new = 5
        M = 10

        A_half = torch.randn(d, d, dtype=torch.float64)
        A = A_half @ A_half.T + 10.0 * torch.eye(d, dtype=torch.float64)
        K_new = torch.randn(d, n_new, dtype=torch.float64)
        K_S = torch.randn(d, M, dtype=torch.float64)
        h = torch.rand(M, dtype=torch.float64) + 0.1
        resid = torch.randn(d, n_new, dtype=torch.float64)

        candidates = [0.01, 0.1, 1.0, 10.0, 100.0]

        # Use a tight epsilon that requires some lambda
        adj_k_loose, _, _ = woodbury_solve(
            A, K_new, K_S, h,
            lambda_candidates=[0.01],
            epsilon_t=float('inf'),
            E_max=float('inf'),
        )
        D_loose = compute_displacement(adj_k_loose, K_S, h, resid)
        tight_eps = D_loose * 0.1  # require 10x reduction

        adj_k, lambda_used, info = woodbury_solve(
            A, K_new, K_S, h,
            lambda_candidates=candidates,
            epsilon_t=tight_eps,
            E_max=float('inf'),
            resid=resid,
        )

        assert lambda_used > 0.01, "Should need lambda > smallest candidate"

        # Verify chosen lambda actually satisfies the budget
        D_chosen = compute_displacement(adj_k, K_S, h, resid)
        assert D_chosen <= tight_eps * 1.01, f"Displacement {D_chosen} exceeds budget {tight_eps}"

    def test_empty_K_S_returns_base(self):
        """With no vulnerable keys, returns base solve."""
        torch.manual_seed(42)
        d = 32
        n_new = 4

        A_half = torch.randn(d, d, dtype=torch.float64)
        A = A_half @ A_half.T + 5.0 * torch.eye(d, dtype=torch.float64)
        K_new = torch.randn(d, n_new, dtype=torch.float64)
        K_S = torch.empty(d, 0, dtype=torch.float64)
        h = torch.empty(0, dtype=torch.float64)

        base_sol = torch.linalg.solve(A, K_new)

        adj_k, lambda_used, _ = woodbury_solve(
            A, K_new, K_S, h,
            lambda_candidates=[1.0],
            epsilon_t=float('inf'),
            E_max=float('inf'),
        )

        assert torch.allclose(adj_k, base_sol, atol=1e-8)
        assert lambda_used == 0.0


class TestComputeDisplacement:
    """Test displacement metric D_t computation."""

    def test_displacement_formula(self):
        """D_t = ||resid @ adj_k^T @ K_S @ diag(sqrt(h))||_F^2"""
        torch.manual_seed(42)
        d = 32
        n_new = 4
        M = 5

        adj_k = torch.randn(d, n_new, dtype=torch.float64)
        K_S = torch.randn(d, M, dtype=torch.float64)
        h = torch.rand(M, dtype=torch.float64) + 0.1
        resid = torch.randn(d, n_new, dtype=torch.float64)

        D = compute_displacement(adj_k, K_S, h, resid)

        # Manual computation
        h_sqrt = torch.sqrt(h)
        delta_W = resid @ adj_k.T  # [d, d]
        weighted = delta_W @ K_S @ torch.diag(h_sqrt)  # [d, M]
        expected = torch.sum(weighted ** 2).item()

        assert abs(D - expected) < 1e-8, f"Displacement mismatch: {D} vs {expected}"

    def test_displacement_zero_for_zero_update(self):
        """Zero adj_k produces zero displacement."""
        d = 16
        M = 3
        adj_k = torch.zeros(d, 2, dtype=torch.float64)
        K_S = torch.randn(d, M, dtype=torch.float64)
        h = torch.rand(M, dtype=torch.float64) + 0.1
        resid = torch.randn(d, 2, dtype=torch.float64)

        D = compute_displacement(adj_k, K_S, h, resid)
        assert D == 0.0

    def test_displacement_nonnegative(self):
        """Displacement is always >= 0."""
        torch.manual_seed(99)
        for _ in range(10):
            d, M = 16, 4
            adj_k = torch.randn(d, 3, dtype=torch.float64)
            K_S = torch.randn(d, M, dtype=torch.float64)
            h = torch.rand(M, dtype=torch.float64) + 0.1
            resid = torch.randn(d, 3, dtype=torch.float64)
            D = compute_displacement(adj_k, K_S, h, resid)
            assert D >= 0.0


class TestAdaptEpsilon:
    """Test adaptive displacement budget."""

    def test_warmup_returns_base(self):
        """During warmup period, epsilon equals base."""
        eps = adapt_epsilon(
            displacement_history=[0.1, 0.2, 0.15],
            epsilon_base=1.0,
            warmup_batches=10,
        )
        assert eps == 1.0

    def test_elevated_displacement_tightens(self):
        """Sustained high displacement should reduce epsilon below base."""
        stable = [0.1] * 20
        elevated = [0.1] * 15 + [1.0] * 5  # spike in last 5 batches

        eps_stable = adapt_epsilon(stable, epsilon_base=1.0, warmup_batches=5)
        eps_elevated = adapt_epsilon(elevated, epsilon_base=1.0, warmup_batches=5)

        assert eps_elevated < eps_stable, (
            f"Elevated displacement should tighten: {eps_elevated} vs {eps_stable}"
        )

    def test_epsilon_never_negative(self):
        """Epsilon must always be positive."""
        huge = [100.0] * 50
        eps = adapt_epsilon(huge, epsilon_base=1.0, warmup_batches=5)
        assert eps > 0.0

    def test_empty_history(self):
        """Empty history returns base epsilon."""
        eps = adapt_epsilon([], epsilon_base=2.0, warmup_batches=5)
        assert eps == 2.0

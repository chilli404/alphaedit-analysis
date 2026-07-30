"""
Unit tests for the REVIVE spectral subspace filter.

Run with: uv run pytest tests/test_revive.py -v
"""

import sys
import tempfile
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from revive.revive_filter import compute_protected_rank, revive_filter, revive_filter_disabled
from revive.svd_cache import SVDCache, _weight_fingerprint


class TestComputeProtectedRank:
    """Test protected rank computation from singular values."""

    def test_diagonal_matrix(self):
        """Test 1: diagonal matrix with known cumulative energy."""
        # W0 = diag([10, 5, 1, 0.5]) → S = [10, 5, 1, 0.5], sum = 16.5
        S = torch.tensor([10.0, 5.0, 1.0, 0.5])
        # cumsum = [10, 15, 16, 16.5]
        # cumsum/total = [0.606, 0.909, 0.970, 1.0]

        # tau=0.5: k=1 (10/16.5 = 0.606 >= 0.5)
        assert compute_protected_rank(S, 0.5) == 1

        # tau=0.6: k=1 (0.606 >= 0.6)
        assert compute_protected_rank(S, 0.6) == 1

        # tau=0.7: k=2 (0.909 >= 0.7)
        assert compute_protected_rank(S, 0.7) == 2

        # tau=0.91: k=2 (0.909 < 0.91) → k=3 (0.970 >= 0.91)
        assert compute_protected_rank(S, 0.91) == 3

        # tau=0.99: k=3 (0.970 < 0.99) → k=4 would be all, clamped to r-1=3
        assert compute_protected_rank(S, 0.99) == 3

    def test_invalid_tau(self):
        """tau must be in (0, 1)."""
        S = torch.tensor([1.0, 0.5])
        with pytest.raises(ValueError, match="tau must be in"):
            compute_protected_rank(S, 0.0)
        with pytest.raises(ValueError, match="tau must be in"):
            compute_protected_rank(S, 1.0)
        with pytest.raises(ValueError, match="tau must be in"):
            compute_protected_rank(S, -0.1)

    def test_empty_S(self):
        """Empty singular values should raise."""
        with pytest.raises(ValueError):
            compute_protected_rank(torch.tensor([]), 0.5)

    def test_monotone_with_tau(self):
        """Test 5: As tau increases, k must be non-decreasing."""
        S = torch.tensor([10.0, 5.0, 3.0, 2.0, 1.0, 0.5, 0.3, 0.1])
        taus = [0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90]
        ks = [compute_protected_rank(S, t) for t in taus]
        for i in range(len(ks) - 1):
            assert ks[i] <= ks[i + 1], f"k not monotone: {ks}"


class TestReviveFilter:
    """Test the main REVIVE filter function."""

    def _make_svd(self, m: int, n: int):
        """Create a random matrix and its compact SVD."""
        torch.manual_seed(42)
        W0 = torch.randn(m, n, dtype=torch.float64)
        U, S, Vh = torch.linalg.svd(W0, full_matrices=False)
        return U, S, Vh

    def test_exact_block_removal(self):
        """Test 2: Verify only tail-tail block survives."""
        m, n = 8, 8
        U, S, Vh = self._make_svd(m, n)
        tau = 0.3  # will protect first few singular values
        k = compute_protected_rank(S, tau)
        r = S.numel()

        # Construct update in SVD basis with known block structure
        A = torch.zeros(r, r, dtype=torch.float64)
        # dominant-dominant block
        A[:k, :k] = torch.randn(k, k, dtype=torch.float64)
        # dominant-tail block
        A[:k, k:] = torch.randn(k, r - k, dtype=torch.float64)
        # tail-dominant block
        A[k:, :k] = torch.randn(r - k, k, dtype=torch.float64)
        # tail-tail block
        A_tail_tail = torch.randn(r - k, r - k, dtype=torch.float64)
        A[k:, k:] = A_tail_tail

        # Construct delta_w_raw from full A
        delta_w_raw = U @ A @ Vh

        delta_w_safe, metrics = revive_filter(
            delta_w_raw, U, S, Vh, tau,
            compute_spectral_norm=False,
        )

        # The safe update should only contain tail-tail
        expected = U[:, k:] @ A_tail_tail @ Vh[k:, :]
        torch.testing.assert_close(delta_w_safe, expected, atol=1e-10, rtol=1e-10)

    def test_orthogonality(self):
        """Test 3: U_top.T @ delta_w_safe ≈ 0 and delta_w_safe @ V_top ≈ 0."""
        m, n = 16, 16
        U, S, Vh = self._make_svd(m, n)
        tau = 0.4
        k = compute_protected_rank(S, tau)

        delta_w_raw = torch.randn(m, n, dtype=torch.float64)
        delta_w_safe, _ = revive_filter(
            delta_w_raw, U, S, Vh, tau,
            compute_metrics=False,
        )

        U_top = U[:, :k]
        V_top = Vh[:k, :].T  # [n, k]

        left_residual = U_top.T @ delta_w_safe  # should be ~0
        right_residual = delta_w_safe @ V_top  # should be ~0

        assert torch.linalg.norm(left_residual).item() < 1e-10
        assert torch.linalg.norm(right_residual).item() < 1e-10

    def test_identity_when_disabled(self):
        """Test 4: disabled REVIVE returns input unchanged."""
        delta_w_raw = torch.randn(10, 10)
        result, metrics = revive_filter_disabled(delta_w_raw)
        assert result is delta_w_raw  # Same object, zero-copy
        assert metrics is None

    def test_tau_monotone_norm(self):
        """Test 5: As tau increases, retained norm generally non-increasing."""
        m, n = 16, 16
        U, S, Vh = self._make_svd(m, n)
        delta_w_raw = torch.randn(m, n, dtype=torch.float64)

        taus = [0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80]
        norms = []
        for t in taus:
            safe, _ = revive_filter(delta_w_raw, U, S, Vh, t, compute_metrics=False)
            norms.append(torch.linalg.norm(safe, ord="fro").item())

        # Non-increasing (allow small numerical tolerance)
        for i in range(len(norms) - 1):
            assert norms[i] >= norms[i + 1] - 1e-10, (
                f"Norm increased at tau={taus[i+1]}: {norms[i]:.6f} -> {norms[i+1]:.6f}"
            )

    def test_rectangular_m_greater_n(self):
        """Test 6a: Rectangular matrix with m > n."""
        m, n = 16, 8  # "tall" matrix
        U, S, Vh = self._make_svd(m, n)
        r = min(m, n)  # = 8
        assert U.shape == (m, r)
        assert Vh.shape == (r, n)

        delta_w_raw = torch.randn(m, n, dtype=torch.float64)
        delta_w_safe, metrics = revive_filter(
            delta_w_raw, U, S, Vh, 0.3,
            compute_spectral_norm=False,
        )
        assert delta_w_safe.shape == (m, n)
        assert metrics is not None
        assert metrics.removed_fraction > 0  # Something was removed

        # Verify orthogonality with top directions
        k = metrics.k
        U_top = U[:, :k]
        assert torch.linalg.norm(U_top.T @ delta_w_safe).item() < 1e-10

    def test_rectangular_m_less_n(self):
        """Test 6b: Rectangular matrix with m < n (like Llama down_proj)."""
        m, n = 8, 16  # "wide" matrix
        U, S, Vh = self._make_svd(m, n)
        r = min(m, n)  # = 8
        assert U.shape == (m, r)
        assert Vh.shape == (r, n)

        delta_w_raw = torch.randn(m, n, dtype=torch.float64)
        delta_w_safe, metrics = revive_filter(
            delta_w_raw, U, S, Vh, 0.3,
            compute_spectral_norm=False,
        )
        assert delta_w_safe.shape == (m, n)
        assert metrics is not None
        assert metrics.removed_fraction > 0

        # Verify orthogonality
        k = metrics.k
        V_top = Vh[:k, :].T  # [n, k]
        assert torch.linalg.norm(delta_w_safe @ V_top).item() < 1e-10

    def test_orientation_shape_mismatch(self):
        """Test 7: Shape assertion catches transposed update."""
        m, n = 8, 12
        U, S, Vh = self._make_svd(m, n)

        # Deliberately transpose the update
        delta_wrong = torch.randn(n, m, dtype=torch.float64)  # [12, 8] instead of [8, 12]

        with pytest.raises(ValueError, match="doesn't match expected"):
            revive_filter(delta_wrong, U, S, Vh, 0.3)

    def test_metrics_content(self):
        """Verify metrics dict has expected fields."""
        m, n = 8, 8
        U, S, Vh = self._make_svd(m, n)
        delta_w_raw = torch.randn(m, n, dtype=torch.float64)

        _, metrics = revive_filter(
            delta_w_raw, U, S, Vh, 0.3,
            param_name="test.weight",
            layer=5,
            batch=3,
        )

        d = metrics.to_dict()
        assert d["phase"] == "revive"
        assert d["param_name"] == "test.weight"
        assert d["layer"] == 5
        assert d["batch"] == 3
        assert 0 < d["removed_fraction"] <= 1.0
        assert -1 <= d["raw_safe_cosine"] <= 1
        assert d["k"] > 0
        assert d["k"] < d["r"]

    def test_non_finite_input_raises(self):
        """Non-finite delta_w_raw should raise."""
        m, n = 4, 4
        U, S, Vh = self._make_svd(m, n)
        delta_w_raw = torch.randn(m, n, dtype=torch.float64)
        delta_w_raw[0, 0] = float("nan")

        with pytest.raises(ValueError, match="non-finite"):
            revive_filter(delta_w_raw, U, S, Vh, 0.3)

    def test_full_svd_equivalence_square(self):
        """For square matrices, projection approach matches explicit A-zeroing."""
        m = n = 8
        U, S, Vh = self._make_svd(m, n)
        tau = 0.3
        k = compute_protected_rank(S, tau)

        delta_w_raw = torch.randn(m, n, dtype=torch.float64)

        # Method 1: projection (our implementation)
        safe1, _ = revive_filter(delta_w_raw, U, S, Vh, tau, compute_metrics=False)

        # Method 2: explicit A-zeroing
        V = Vh.T
        A = U.T @ delta_w_raw @ V
        A[:k, :] = 0
        A[:, :k] = 0
        safe2 = U @ A @ Vh

        torch.testing.assert_close(safe1, safe2, atol=1e-10, rtol=1e-10)


class TestSVDCache:
    """Test persistent SVD cache."""

    def test_compute_and_store(self):
        """Cache computes SVD on first call, returns cache hit on second."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cache = SVDCache(tmpdir, model_id="test-model")
            W = torch.randn(8, 8, dtype=torch.float32)

            U1, S1, Vh1, hit1 = cache.get_or_compute("layer.weight", W)
            assert not hit1  # First call: miss

            U2, S2, Vh2, hit2 = cache.get_or_compute("layer.weight", W)
            assert hit2  # Second call: memory hit
            torch.testing.assert_close(U1, U2)
            torch.testing.assert_close(S1, S2)

    def test_disk_persistence(self):
        """Cache persists across instances."""
        with tempfile.TemporaryDirectory() as tmpdir:
            W = torch.randn(8, 8, dtype=torch.float32)

            # Instance 1: compute and save
            cache1 = SVDCache(tmpdir, model_id="test-model")
            U1, S1, _, hit1 = cache1.get_or_compute("layer.weight", W)
            assert not hit1

            # Instance 2: load from disk
            cache2 = SVDCache(tmpdir, model_id="test-model")
            U2, S2, _, hit2 = cache2.get_or_compute("layer.weight", W)
            assert hit2
            torch.testing.assert_close(U1, U2)

    def test_cache_rejected_shape_change(self):
        """Test 8a: Cache rejected when shape changes."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cache = SVDCache(tmpdir, model_id="test-model")
            W1 = torch.randn(8, 8, dtype=torch.float32)
            cache.get_or_compute("layer.weight", W1)

            # Different shape → cache miss (recompute)
            W2 = torch.randn(8, 12, dtype=torch.float32)
            _, _, _, hit = cache.get_or_compute("layer.weight", W2)
            assert not hit

    def test_cache_rejected_model_change(self):
        """Test 8b: Cache rejected when model identifier changes."""
        with tempfile.TemporaryDirectory() as tmpdir:
            W = torch.randn(8, 8, dtype=torch.float32)

            cache1 = SVDCache(tmpdir, model_id="model-A")
            cache1.get_or_compute("layer.weight", W)

            cache2 = SVDCache(tmpdir, model_id="model-B")
            _, _, _, hit = cache2.get_or_compute("layer.weight", W)
            assert not hit

    def test_cache_rejected_fingerprint_change(self):
        """Test 8c: Cache rejected when weight fingerprint changes."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cache = SVDCache(tmpdir, model_id="test-model")
            W1 = torch.randn(8, 8, dtype=torch.float32)
            cache.get_or_compute("layer.weight", W1)

            # New cache instance (clears memory cache), different weight
            cache2 = SVDCache(tmpdir, model_id="test-model")
            W2 = torch.randn(8, 8, dtype=torch.float32)  # Different content
            _, _, _, hit = cache2.get_or_compute("layer.weight", W2)
            assert not hit

    def test_svd_validity(self):
        """Cached SVD factors reconstruct original weight."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cache = SVDCache(tmpdir, model_id="test-model")
            W = torch.randn(8, 12, dtype=torch.float32)
            U, S, Vh, _ = cache.get_or_compute("layer.weight", W)

            reconstructed = U @ torch.diag(S) @ Vh
            torch.testing.assert_close(
                reconstructed, W, atol=1e-5, rtol=1e-5
            )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

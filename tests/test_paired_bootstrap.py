"""
Tests for the paired bootstrap statistical comparison module.

Tests effect size calculations, bootstrap testing, and multiple
comparison correction. All tests are pure math — no GPU required.
"""

import numpy as np
import pytest

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "analysis"))

from paired_bootstrap import (
    cohens_d,
    cliffs_delta,
    paired_bootstrap_test,
    holm_bonferroni,
)


class TestCohensD:
    """Tests for Cohen's d effect size."""

    def test_identical_arrays(self):
        """Identical arrays should give d=0."""
        a = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        assert cohens_d(a, a) == 0.0

    def test_constant_arrays(self):
        """Constant arrays with same value should give d=0."""
        a = np.array([5.0, 5.0, 5.0, 5.0])
        b = np.array([5.0, 5.0, 5.0, 5.0])
        assert cohens_d(a, b) == 0.0

    def test_constant_arrays_different_values(self):
        """Constant arrays with different values — pooled_sd=0, should return 0."""
        a = np.array([5.0, 5.0, 5.0, 5.0])
        b = np.array([3.0, 3.0, 3.0, 3.0])
        assert cohens_d(a, b) == 0.0

    def test_large_effect(self):
        """Well-separated distributions should give large |d|."""
        a = np.array([10.0, 11.0, 10.5, 10.2, 10.8])
        b = np.array([1.0, 1.5, 0.8, 1.2, 1.1])
        d = cohens_d(a, b)
        assert d > 2.0  # Very large effect

    def test_sign_direction(self):
        """Positive d means A > B, negative means A < B."""
        a = np.array([5.0, 6.0, 7.0])
        b = np.array([1.0, 2.0, 3.0])
        assert cohens_d(a, b) > 0
        assert cohens_d(b, a) < 0

    def test_known_value(self):
        """Test against hand-computed Cohen's d."""
        # A = [4, 6], mean=5, sd=sqrt(2)
        # B = [1, 3], mean=2, sd=sqrt(2)
        # pooled_sd = sqrt((2+2)/2) = sqrt(2) ≈ 1.414
        # d = (5-2)/sqrt(2) ≈ 2.121
        a = np.array([4.0, 6.0])
        b = np.array([1.0, 3.0])
        d = cohens_d(a, b)
        expected = 3.0 / np.sqrt(2)
        assert abs(d - expected) < 0.001

    def test_small_effect(self):
        """Overlapping distributions should give small |d|."""
        rng = np.random.default_rng(42)
        a = rng.normal(5.0, 2.0, size=100)
        b = rng.normal(4.8, 2.0, size=100)
        d = cohens_d(a, b)
        assert abs(d) < 0.5  # Small effect


class TestCliffsDelta:
    """Tests for Cliff's delta non-parametric effect size."""

    def test_identical_arrays(self):
        """Identical arrays should give delta=0."""
        a = np.array([1.0, 2.0, 3.0])
        assert cliffs_delta(a, a) == 0.0

    def test_complete_dominance_positive(self):
        """All A > all B should give delta=+1."""
        a = np.array([10.0, 11.0, 12.0])
        b = np.array([1.0, 2.0, 3.0])
        assert cliffs_delta(a, b) == 1.0

    def test_complete_dominance_negative(self):
        """All B > all A should give delta=-1."""
        a = np.array([1.0, 2.0, 3.0])
        b = np.array([10.0, 11.0, 12.0])
        assert cliffs_delta(a, b) == -1.0

    def test_empty_arrays(self):
        """Empty arrays should return 0."""
        assert cliffs_delta(np.array([]), np.array([1.0, 2.0])) == 0.0
        assert cliffs_delta(np.array([1.0, 2.0]), np.array([])) == 0.0

    def test_range(self):
        """Result should always be in [-1, 1]."""
        rng = np.random.default_rng(42)
        for _ in range(10):
            a = rng.normal(0, 1, size=20)
            b = rng.normal(0, 1, size=20)
            d = cliffs_delta(a, b)
            assert -1.0 <= d <= 1.0

    def test_sign_direction(self):
        """Positive delta means A tends to be larger."""
        a = np.array([5.0, 6.0, 7.0, 8.0])
        b = np.array([1.0, 2.0, 3.0, 4.0])
        assert cliffs_delta(a, b) > 0
        assert cliffs_delta(b, a) < 0

    def test_known_value(self):
        """Test with a simple example.
        A = [3, 4], B = [1, 2]
        Comparisons: 3>1(+1), 3>2(+1), 4>1(+1), 4>2(+1) = 4
        Total comparisons = 2*2 = 4
        delta = 4/4 = 1.0
        """
        a = np.array([3.0, 4.0])
        b = np.array([1.0, 2.0])
        assert cliffs_delta(a, b) == 1.0


class TestPairedBootstrapTest:
    """Tests for the full paired bootstrap test."""

    def test_identical_scores(self):
        """Identical scores should give mean_diff=0 and high p-value."""
        scores = np.array([0.8, 0.9, 0.7, 0.85, 0.75])
        result = paired_bootstrap_test(scores, scores, seed=42)
        assert result["mean_diff"] == 0.0
        assert result["p_value"] == 1.0
        assert result["n_pairs"] == 5

    def test_clearly_different_scores(self):
        """Clearly separated scores should give low p-value."""
        a = np.array([0.9, 0.95, 0.92, 0.88, 0.91, 0.93, 0.89, 0.94, 0.90, 0.92])
        b = np.array([0.5, 0.55, 0.48, 0.52, 0.51, 0.49, 0.53, 0.50, 0.47, 0.54])
        result = paired_bootstrap_test(a, b, seed=42)
        assert result["mean_diff"] > 0
        assert result["p_value"] < 0.01
        assert result["cohens_d"] > 0
        assert result["cliffs_delta"] > 0

    def test_nan_handling(self):
        """NaN pairs should be removed before testing."""
        a = np.array([0.8, np.nan, 0.9, 0.7, np.nan])
        b = np.array([0.6, 0.7, np.nan, 0.5, 0.4])
        result = paired_bootstrap_test(a, b, seed=42)
        # Only pairs where BOTH are non-NaN: indices [0, 3] = 2 pairs
        assert result["n_pairs"] == 2

    def test_all_nan(self):
        """All-NaN should return NaN results."""
        a = np.array([np.nan, np.nan])
        b = np.array([np.nan, np.nan])
        result = paired_bootstrap_test(a, b, seed=42)
        assert result["n_pairs"] == 0
        assert np.isnan(result["mean_diff"])

    def test_mismatched_lengths_raises(self):
        """Different length arrays should raise AssertionError."""
        a = np.array([1.0, 2.0, 3.0])
        b = np.array([1.0, 2.0])
        with pytest.raises(AssertionError):
            paired_bootstrap_test(a, b)

    def test_result_keys(self):
        """Should return all expected keys."""
        a = np.array([0.8, 0.9, 0.7, 0.85, 0.75])
        b = np.array([0.6, 0.7, 0.5, 0.65, 0.55])
        result = paired_bootstrap_test(a, b, seed=42)
        expected_keys = {
            "mean_diff", "ci_lower", "ci_upper", "p_value",
            "n_pairs", "cohens_d", "cliffs_delta"
        }
        assert set(result.keys()) == expected_keys

    def test_ci_contains_mean(self):
        """CI should contain the observed mean difference."""
        a = np.array([0.8, 0.9, 0.7, 0.85, 0.75, 0.82, 0.88, 0.79])
        b = np.array([0.6, 0.7, 0.5, 0.65, 0.55, 0.62, 0.68, 0.59])
        result = paired_bootstrap_test(a, b, seed=42)
        assert result["ci_lower"] <= result["mean_diff"] <= result["ci_upper"]

    def test_determinism(self):
        """Same seed should give same results."""
        a = np.random.default_rng(1).normal(0.8, 0.1, size=30)
        b = np.random.default_rng(2).normal(0.7, 0.1, size=30)
        r1 = paired_bootstrap_test(a, b, seed=42)
        r2 = paired_bootstrap_test(a, b, seed=42)
        assert r1 == r2


class TestHolmBonferroni:
    """Tests for Holm-Bonferroni multiple comparison correction."""

    def test_single_pvalue(self):
        """Single p-value should be unchanged."""
        results = holm_bonferroni([0.03])
        assert results[0]["p_raw"] == 0.03
        assert results[0]["p_adjusted"] == 0.03
        assert results[0]["significant_raw"] is True
        assert results[0]["significant_adjusted"] is True

    def test_all_significant(self):
        """Very small p-values should remain significant after correction."""
        p_values = [0.001, 0.002, 0.003]
        results = holm_bonferroni(p_values)
        for r in results:
            assert r["significant_adjusted"] is True

    def test_none_significant(self):
        """Large p-values should not be significant."""
        p_values = [0.5, 0.6, 0.7, 0.8]
        results = holm_bonferroni(p_values)
        for r in results:
            assert r["significant_adjusted"] is False

    def test_correction_makes_pvalue_larger(self):
        """Adjusted p-values should be >= raw p-values."""
        p_values = [0.01, 0.03, 0.04, 0.06]
        results = holm_bonferroni(p_values)
        for r in results:
            assert r["p_adjusted"] >= r["p_raw"]

    def test_adjusted_capped_at_one(self):
        """Adjusted p-values should never exceed 1.0."""
        p_values = [0.4, 0.5, 0.6]
        results = holm_bonferroni(p_values)
        for r in results:
            assert r["p_adjusted"] <= 1.0

    def test_monotonicity(self):
        """After sorting by raw p-value, adjusted p-values should be non-decreasing."""
        p_values = [0.03, 0.01, 0.05, 0.02]
        results = holm_bonferroni(p_values)
        # Sort by raw p-value
        sorted_results = sorted(results, key=lambda r: r["p_raw"])
        adj_values = [r["p_adjusted"] for r in sorted_results]
        for i in range(len(adj_values) - 1):
            assert adj_values[i] <= adj_values[i + 1]

    def test_known_correction(self):
        """Test with known Holm-Bonferroni example.
        p-values: [0.01, 0.04, 0.03]
        Sorted: [0.01, 0.03, 0.04] with ranks 1, 2, 3
        Corrections: 0.01*3=0.03, 0.03*2=0.06, 0.04*1=0.04
        Monotonicity: [0.03, 0.06, 0.06]
        """
        p_values = [0.01, 0.04, 0.03]
        results = holm_bonferroni(p_values)
        # The smallest raw p-value (0.01) gets multiplied by 3
        assert abs(results[0]["p_adjusted"] - 0.03) < 1e-10
        # The middle (0.03) gets multiplied by 2 = 0.06
        assert abs(results[2]["p_adjusted"] - 0.06) < 1e-10
        # The largest (0.04) gets multiplied by 1 = 0.04, but
        # monotonicity forces it up to 0.06
        assert abs(results[1]["p_adjusted"] - 0.06) < 1e-10

    def test_preserves_original_order(self):
        """Results should be in the same order as input p-values."""
        p_values = [0.05, 0.01, 0.10]
        results = holm_bonferroni(p_values)
        assert results[0]["p_raw"] == 0.05
        assert results[1]["p_raw"] == 0.01
        assert results[2]["p_raw"] == 0.10

    def test_boundary_case(self):
        """p=0.05 with k=1 should be exactly significant."""
        results = holm_bonferroni([0.049])
        assert results[0]["significant_raw"] is True
        assert results[0]["significant_adjusted"] is True

        results = holm_bonferroni([0.051])
        assert results[0]["significant_raw"] is False
        assert results[0]["significant_adjusted"] is False

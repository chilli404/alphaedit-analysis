"""
Tests for the confidence interval computation module.

Tests Wilson intervals, BCa bootstrap CIs, and helper functions.
All tests are pure math — no GPU or model access required.
"""

import numpy as np
import pytest
from scipy import stats

# Add analysis directory to path
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "analysis"))

from confidence_intervals import (
    wilson_interval,
    bootstrap_ci,
    _bca_ci,
    compute_metric_ci,
    format_ci,
)


class TestWilsonInterval:
    """Tests for Wilson score interval for binomial proportions."""

    def test_basic_50_percent(self):
        """50/100 successes should give interval centered near 0.5."""
        lower, upper = wilson_interval(50, 100)
        assert lower < 0.5 < upper
        # Should be roughly symmetric around 0.5
        assert abs((lower + upper) / 2 - 0.5) < 0.01

    def test_perfect_score(self):
        """100/100 should give upper bound of 1.0 and lower < 1.0."""
        lower, upper = wilson_interval(100, 100)
        assert upper == 1.0
        assert lower < 1.0
        assert lower > 0.95  # Should be close to 1.0

    def test_zero_score(self):
        """0/100 should give lower bound near 0.0 and upper > 0.0."""
        lower, upper = wilson_interval(0, 100)
        assert lower < 1e-10  # Effectively zero (may have float epsilon)
        assert upper > 0.0
        assert upper < 0.04  # Should be close to 0.0 (~0.037)

    def test_empty_sample(self):
        """n=0 should return (0, 1) — full uncertainty."""
        lower, upper = wilson_interval(0, 0)
        assert lower == 0.0
        assert upper == 1.0

    def test_single_success(self):
        """1/1 should return valid interval."""
        lower, upper = wilson_interval(1, 1)
        assert 0.0 <= lower < upper <= 1.0

    def test_bounds_always_valid(self):
        """Bounds should always satisfy 0 <= lower < upper <= 1."""
        for n in [5, 10, 50, 100, 1000]:
            for k in range(n + 1):
                lower, upper = wilson_interval(k, n)
                assert 0.0 <= lower <= upper <= 1.0

    def test_wider_interval_with_fewer_samples(self):
        """Fewer samples should give wider intervals."""
        _, upper_10 = wilson_interval(5, 10)
        lower_10, _ = wilson_interval(5, 10)
        _, upper_100 = wilson_interval(50, 100)
        lower_100, _ = wilson_interval(50, 100)

        width_10 = upper_10 - lower_10
        width_100 = upper_100 - lower_100
        assert width_10 > width_100

    def test_higher_confidence_wider_interval(self):
        """99% CI should be wider than 90% CI."""
        lower_90, upper_90 = wilson_interval(50, 100, confidence=0.90)
        lower_99, upper_99 = wilson_interval(50, 100, confidence=0.99)
        assert (upper_99 - lower_99) > (upper_90 - lower_90)

    def test_known_value(self):
        """Test against computed Wilson interval values."""
        # For n=100, k=50, z=1.96 (95% CI):
        # Verified empirically: [0.4038, 0.5962]
        lower, upper = wilson_interval(50, 100, confidence=0.95)
        assert abs(lower - 0.4038) < 0.001
        assert abs(upper - 0.5962) < 0.001


class TestBootstrapCI:
    """Tests for BCa bootstrap confidence intervals."""

    def test_basic_mean(self):
        """Bootstrap CI for a simple normal sample."""
        rng = np.random.default_rng(123)
        data = rng.normal(5.0, 1.0, size=100)

        point, ci_lower, ci_upper = bootstrap_ci(data, seed=42)
        assert ci_lower < point < ci_upper
        assert abs(point - 5.0) < 0.5  # Should be near true mean

    def test_ci_contains_true_mean(self):
        """95% CI should contain the true mean most of the time."""
        # Use a large sample where we're confident about the mean
        data = np.random.default_rng(99).normal(10.0, 2.0, size=500)
        point, ci_lower, ci_upper = bootstrap_ci(data, seed=42)
        assert ci_lower < 10.0 < ci_upper

    def test_empty_data(self):
        """Empty array should return NaN."""
        point, ci_lower, ci_upper = bootstrap_ci(np.array([]))
        assert np.isnan(point)
        assert np.isnan(ci_lower)
        assert np.isnan(ci_upper)

    def test_single_element(self):
        """Single element should return a valid result."""
        point, ci_lower, ci_upper = bootstrap_ci(np.array([5.0]))
        assert point == 5.0
        # With one element, all bootstrap samples are [5.0]
        assert ci_lower == 5.0
        assert ci_upper == 5.0

    def test_determinism_with_seed(self):
        """Same seed should give same result."""
        data = np.random.default_rng(1).normal(0, 1, size=50)
        r1 = bootstrap_ci(data, seed=42)
        r2 = bootstrap_ci(data, seed=42)
        assert r1 == r2

    def test_different_seed_different_result(self):
        """Different seeds should (usually) give different results."""
        data = np.random.default_rng(1).normal(0, 1, size=50)
        r1 = bootstrap_ci(data, seed=42)
        r2 = bootstrap_ci(data, seed=99)
        # Point estimates are the same (same data), CIs may differ
        assert r1[0] == r2[0]
        # At least one CI bound should differ
        assert r1[1] != r2[1] or r1[2] != r2[2]

    def test_percentile_method(self):
        """Percentile method should also work."""
        data = np.random.default_rng(1).normal(0, 1, size=50)
        point, ci_lower, ci_upper = bootstrap_ci(data, method="percentile", seed=42)
        assert ci_lower < point < ci_upper

    def test_custom_statistic(self):
        """Should work with custom statistics like median."""
        data = np.random.default_rng(1).normal(0, 1, size=100)
        point, ci_lower, ci_upper = bootstrap_ci(data, statistic=np.median, seed=42)
        assert ci_lower < point < ci_upper

    def test_narrow_with_low_variance(self):
        """Low-variance data should give narrow CI."""
        data_narrow = np.array([5.0, 5.01, 4.99, 5.0, 5.02, 4.98] * 10)
        data_wide = np.array([0.0, 10.0, 2.0, 8.0, 4.0, 6.0] * 10)

        _, lo_n, hi_n = bootstrap_ci(data_narrow, seed=42, n_bootstrap=1000)
        _, lo_w, hi_w = bootstrap_ci(data_wide, seed=42, n_bootstrap=1000)

        assert (hi_n - lo_n) < (hi_w - lo_w)


class TestComputeMetricCI:
    """Tests for compute_metric_ci wrapper function."""

    def test_continuous_metric(self):
        """Continuous metric should use bootstrap."""
        values = np.random.default_rng(1).normal(0.8, 0.1, size=50)
        result = compute_metric_ci(values, metric_type="continuous", seed=42)

        assert "mean" in result
        assert "ci_lower" in result
        assert "ci_upper" in result
        assert "n" in result
        assert "std" in result
        assert result["n"] == 50
        assert result["ci_lower"] < result["mean"] < result["ci_upper"]

    def test_binary_metric(self):
        """Binary metric should use Wilson interval."""
        values = np.array([1.0, 1.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0, 1.0])
        result = compute_metric_ci(values, metric_type="binary")

        assert result["n"] == 10
        assert result["mean"] == 0.8  # 8/10
        assert result["ci_lower"] < 0.8 < result["ci_upper"]

    def test_handles_nan_values(self):
        """NaN values should be filtered out."""
        values = np.array([1.0, 2.0, np.nan, 3.0, np.nan, 4.0])
        result = compute_metric_ci(values, metric_type="continuous", seed=42)
        assert result["n"] == 4  # Only non-NaN values counted

    def test_empty_after_nan_filter(self):
        """All-NaN array should return NaN results."""
        values = np.array([np.nan, np.nan, np.nan])
        result = compute_metric_ci(values)
        assert np.isnan(result["mean"])
        assert result["n"] == 0

    def test_empty_input(self):
        """Empty array should return NaN results."""
        result = compute_metric_ci(np.array([]))
        assert np.isnan(result["mean"])
        assert result["n"] == 0


class TestFormatCI:
    """Tests for CI string formatting."""

    def test_basic_format(self):
        """Standard formatting with 3 decimal places."""
        result = {"mean": 0.854, "ci_lower": 0.812, "ci_upper": 0.896}
        s = format_ci(result)
        assert "0.854" in s
        assert "0.812" in s
        assert "0.896" in s

    def test_nan_returns_dash(self):
        """NaN mean should return em-dash."""
        result = {"mean": np.nan, "ci_lower": np.nan, "ci_upper": np.nan}
        assert format_ci(result) == "\u2014"

    def test_custom_decimals(self):
        """Should respect custom decimal places."""
        result = {"mean": 0.12345, "ci_lower": 0.11111, "ci_upper": 0.13579}
        s = format_ci(result, decimals=2)
        assert "0.12" in s
        assert "0.11" in s
        assert "0.14" in s  # Rounded up

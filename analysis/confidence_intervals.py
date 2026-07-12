#!/usr/bin/env python3
"""
Confidence interval computation for AlphaEdit reproducibility study.

Provides:
- Wilson intervals for binary success rates (efficacy, generalization, specificity)
- BCa (bias-corrected and accelerated) bootstrap CIs for continuous metrics
- Summary table generation with CI columns

References:
- Wilson, E.B. (1927). Probable inference, the law of succession, and statistical inference.
- Efron & Tibshirani (1993). An Introduction to the Bootstrap.
- DiCiccio & Efron (1996). Bootstrap confidence intervals.
"""

import numpy as np
from scipy import stats


N_BOOTSTRAP = 10000  # Consistent across all modules


def wilson_interval(successes: int, n: int, confidence: float = 0.95) -> tuple:
    """
    Wilson score interval for a binomial proportion.

    Preferred over normal approximation for proportions near 0 or 1,
    which is common in knowledge editing (efficacy often >95%).

    Args:
        successes: Number of successes
        n: Total trials
        confidence: Confidence level (default 0.95)

    Returns:
        (lower, upper) bounds of the confidence interval
    """
    if n == 0:
        return (0.0, 1.0)

    z = stats.norm.ppf(1 - (1 - confidence) / 2)
    p_hat = successes / n

    denominator = 1 + z**2 / n
    center = (p_hat + z**2 / (2 * n)) / denominator
    margin = (z / denominator) * np.sqrt(p_hat * (1 - p_hat) / n + z**2 / (4 * n**2))

    return (max(0.0, center - margin), min(1.0, center + margin))


def bootstrap_ci(
    data: np.ndarray,
    statistic=np.mean,
    n_bootstrap: int = N_BOOTSTRAP,
    confidence: float = 0.95,
    seed: int = 42,
    method: str = "bca",
) -> tuple:
    """
    Bootstrap confidence interval with BCa or percentile method.

    Args:
        data: Array of observations
        statistic: Function to compute (default: np.mean)
        n_bootstrap: Number of bootstrap resamples
        confidence: Confidence level
        seed: Random seed for reproducibility
        method: "bca" (default, recommended) or "percentile"

    Returns:
        (point_estimate, ci_lower, ci_upper)
    """
    rng = np.random.default_rng(seed)
    data = np.asarray(data, dtype=float)
    n = len(data)

    if n == 0:
        return (np.nan, np.nan, np.nan)

    point = float(statistic(data))

    boot_stats = np.array([
        statistic(rng.choice(data, size=n, replace=True))
        for _ in range(n_bootstrap)
    ])

    if method == "bca" and n >= 3:
        ci_lower, ci_upper = _bca_ci(data, boot_stats, point, confidence, statistic)
    else:
        alpha = 1 - confidence
        ci_lower = np.percentile(boot_stats, 100 * alpha / 2)
        ci_upper = np.percentile(boot_stats, 100 * (1 - alpha / 2))

    return (point, ci_lower, ci_upper)


def _bca_ci(
    data: np.ndarray,
    boot_stats: np.ndarray,
    observed: float,
    confidence: float,
    statistic=np.mean,
) -> tuple[float, float]:
    """
    Bias-corrected and accelerated (BCa) confidence interval.

    Better coverage than percentile method for skewed distributions.
    Falls back to percentile if numerical issues arise.
    """
    n = len(data)
    alpha = 1 - confidence

    # Bias correction factor (z0)
    prop_below = np.mean(boot_stats < observed)
    if prop_below == 0:
        prop_below = 1 / (2 * len(boot_stats))
    elif prop_below == 1:
        prop_below = 1 - 1 / (2 * len(boot_stats))
    z0 = stats.norm.ppf(prop_below)

    # Acceleration factor (a) via jackknife
    jackknife_stats = np.array([
        statistic(np.delete(data, i)) for i in range(n)
    ])
    jack_mean = jackknife_stats.mean()
    numerator = np.sum((jack_mean - jackknife_stats) ** 3)
    denominator = 6.0 * (np.sum((jack_mean - jackknife_stats) ** 2) ** 1.5)
    a = numerator / denominator if denominator != 0 else 0.0

    # Adjusted percentiles
    z_lo = stats.norm.ppf(alpha / 2)
    z_hi = stats.norm.ppf(1 - alpha / 2)

    denom_lo = 1 - a * (z0 + z_lo)
    denom_hi = 1 - a * (z0 + z_hi)

    if denom_lo == 0 or denom_hi == 0:
        ci_lower = np.percentile(boot_stats, 100 * alpha / 2)
        ci_upper = np.percentile(boot_stats, 100 * (1 - alpha / 2))
        return ci_lower, ci_upper

    alpha_lo = stats.norm.cdf(z0 + (z0 + z_lo) / denom_lo)
    alpha_hi = stats.norm.cdf(z0 + (z0 + z_hi) / denom_hi)

    alpha_lo = np.clip(alpha_lo, 0.001, 0.999)
    alpha_hi = np.clip(alpha_hi, 0.001, 0.999)

    ci_lower = np.percentile(boot_stats, 100 * alpha_lo)
    ci_upper = np.percentile(boot_stats, 100 * alpha_hi)

    return ci_lower, ci_upper


def compute_metric_ci(
    values: np.ndarray,
    metric_type: str = "continuous",
    confidence: float = 0.95,
    n_bootstrap: int = N_BOOTSTRAP,
    seed: int = 42,
) -> dict:
    """
    Compute appropriate CI for a metric.

    Args:
        values: Array of metric values
        metric_type: "binary" for success rates, "continuous" for probabilities/scores
        confidence: Confidence level
        n_bootstrap: Bootstrap resamples (for continuous)
        seed: Random seed

    Returns:
        Dict with keys: mean, ci_lower, ci_upper, n, std
    """
    values = np.asarray(values, dtype=float)
    values = values[~np.isnan(values)]
    n = len(values)

    if n == 0:
        return {"mean": np.nan, "ci_lower": np.nan, "ci_upper": np.nan, "n": 0, "std": np.nan}

    if metric_type == "binary":
        # Treat values as 0/1 indicators
        successes = int(np.sum(values >= 0.5))  # threshold at 0.5
        ci_lower, ci_upper = wilson_interval(successes, n, confidence)
        return {
            "mean": successes / n,
            "ci_lower": ci_lower,
            "ci_upper": ci_upper,
            "n": n,
            "std": np.std(values, ddof=1) if n > 1 else 0.0,
        }
    else:
        point, ci_lower, ci_upper = bootstrap_ci(
            values, n_bootstrap=n_bootstrap, confidence=confidence, seed=seed,
            method="bca",
        )
        return {
            "mean": point,
            "ci_lower": ci_lower,
            "ci_upper": ci_upper,
            "n": n,
            "std": np.std(values, ddof=1) if n > 1 else 0.0,
        }


def format_ci(result: dict, decimals: int = 3) -> str:
    """Format a CI result as 'mean [lower, upper]' for paper tables."""
    if np.isnan(result["mean"]):
        return "—"
    return (
        f"{result['mean']:.{decimals}f} "
        f"[{result['ci_lower']:.{decimals}f}, {result['ci_upper']:.{decimals}f}]"
    )

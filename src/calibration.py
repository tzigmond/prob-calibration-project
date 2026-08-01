"""Intervals + actual test values -> the headline numbers.

Coverage is gameable on its own (a wide-enough interval always covers), so it is
never reported alone - always alongside average width (the sharpness check). And
because the test set is small (~700 points -> ~7 expected violations at 99%),
every coverage number carries a binomial confidence interval and a Kupiec test,
so apparent differences aren't read as signal when they're noise.
"""
from __future__ import annotations

from typing import Callable

import numpy as np
import pandas as pd
from scipy import stats


def empirical_coverage(
    y_true: np.ndarray, lower: np.ndarray, upper: np.ndarray
) -> float:
    """Fraction of test points falling within [lower, upper]."""
    return float(np.mean((y_true >= lower) & (y_true <= upper)))


def average_width(lower: np.ndarray, upper: np.ndarray) -> float:
    """Mean interval width."""
    return float(np.mean(upper - lower))


def coverage_ci(n: int, covered: int, alpha: float = 0.05) -> tuple[float, float]:
    """Wilson score confidence interval on an empirical coverage estimate.

    ``covered`` of ``n`` test points fell inside the interval. Returns the
    (lo, hi) bounds at confidence 1-alpha. Wilson is used rather than the normal
    approximation because it stays inside [0, 1] and behaves well when the count
    of misses is tiny - exactly the high-confidence-level regime here.
    """
    if n == 0:
        return (float("nan"), float("nan"))
    z = stats.norm.ppf(1 - alpha / 2.0)
    p_hat = covered / n
    denom = 1 + z ** 2 / n
    center = (p_hat + z ** 2 / (2 * n)) / denom
    half = z * np.sqrt(p_hat * (1 - p_hat) / n + z ** 2 / (4 * n ** 2)) / denom
    return (float(center - half), float(center + half))


def kupiec_test(n: int, failures: int, level: float) -> float:
    """Kupiec proportion-of-failures (unconditional coverage) test.

    Likelihood-ratio test that the observed failure rate ``failures/n`` matches
    the expected miss rate ``1 - level``. Returns a p-value; small p means the
    interval's coverage is significantly off nominal. Turns "the table looks
    different" into a number.
    """
    if n == 0:
        return float("nan")
    p = 1 - level                       # expected failure probability
    x = failures
    p_hat = x / n

    def _binom_loglik(prob: float) -> float:
        # (n - x) * log(1 - prob) + x * log(prob), with 0 * log(0) := 0
        term_hi = 0.0 if x == 0 else x * np.log(prob)
        term_lo = 0.0 if (n - x) == 0 else (n - x) * np.log(1 - prob)
        return term_hi + term_lo

    lr = -2.0 * (_binom_loglik(p) - _binom_loglik(p_hat))
    lr = max(lr, 0.0)                    # guard tiny negative from rounding
    return float(stats.chi2.sf(lr, df=1))


IntervalFn = Callable[[float], tuple[np.ndarray, np.ndarray]]


def coverage_table(
    y_true: np.ndarray,
    models: dict[str, IntervalFn],
    levels: list[float],
) -> pd.DataFrame:
    """Assemble the centerpiece table.

    ``models`` maps a model name to an ``interval_fn(level) -> (lower, upper)``
    closure (each model captures its own predictions/scale/params). Rows are the
    models; columns are a (level, metric) MultiIndex where metric is one of
    {coverage, ci_lo, ci_hi, width, kupiec_p}. Experiments write the result to
    results/tables/.
    """
    n = len(y_true)
    records: dict[str, dict[tuple[str, str], float]] = {}
    for name, interval_fn in models.items():
        row: dict[tuple[str, str], float] = {}
        for level in levels:
            lower, upper = interval_fn(level)
            inside = (y_true >= lower) & (y_true <= upper)
            covered = int(np.sum(inside))
            cov = covered / n
            ci_lo, ci_hi = coverage_ci(n, covered)
            kp = kupiec_test(n, failures=n - covered, level=level)
            width = average_width(lower, upper)
            tag = f"{int(round(level * 100))}%"
            row[(tag, "coverage")] = cov
            row[(tag, "ci_lo")] = ci_lo
            row[(tag, "ci_hi")] = ci_hi
            row[(tag, "width")] = width
            row[(tag, "kupiec_p")] = kp
        records[name] = row

    df = pd.DataFrame.from_dict(records, orient="index")
    df.columns = pd.MultiIndex.from_tuples(df.columns, names=["level", "metric"])
    return df

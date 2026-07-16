"""Intervals + actual test values -> the headline numbers.

Coverage is gameable on its own (a wide-enough interval always covers), so it is
always reported alongside average width — the sharpness check that makes coverage
meaningful. And because the test set is small (~700 points -> ~7 expected
violations at 99%), every coverage number carries a binomial confidence interval
and a Kupiec test, so apparent differences aren't read as signal when they're
noise.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def empirical_coverage(
    y_true: np.ndarray, lower: np.ndarray, upper: np.ndarray
) -> float:
    """Fraction of test points falling within [lower, upper]."""
    # mean( (y_true >= lower) & (y_true <= upper) )
    raise NotImplementedError


def average_width(lower: np.ndarray, upper: np.ndarray) -> float:
    """Mean interval width."""
    # mean(upper - lower)
    raise NotImplementedError


def coverage_ci(
    n: int, covered: int, alpha: float = 0.05
) -> tuple[float, float]:
    """Binomial confidence interval on an empirical coverage estimate.

    ``covered`` of ``n`` test points fell inside the interval. Returns the
    (lo, hi) bounds at confidence 1-alpha (Wilson score interval — better than
    normal-approx in the tails where counts are small).
    """
    raise NotImplementedError


def kupiec_test(n: int, failures: int, level: float) -> float:
    """Kupiec unconditional-coverage (proportion-of-failures) test.

    Likelihood-ratio test that the observed failure rate ``failures/n`` matches
    the expected miss rate ``1 - level``. Returns a p-value; small p means the
    interval's coverage is significantly off nominal. Turns "the table looks
    different" into a number.
    """
    # LR_uc = -2 * log( ((1-p)^(n-x) p^x) / ((1-p̂)^(n-x) p̂^x) ), p = 1-level,
    #   x = failures, p̂ = x/n; p-value from chi2(df=1).
    raise NotImplementedError


def coverage_table(models: dict, levels: list[float]) -> pd.DataFrame:
    """Assemble the centerpiece table.

    Rows = the five models {Gaussian, EWMA-Gaussian, Laplace, Student-t,
    Empirical}; columns = the nominal ``levels`` (e.g. 0.90/0.95/0.99); each cell
    carries coverage, its binomial CI, average width, and the Kupiec p-value.
    ``models`` maps a model name to whatever is needed to build its intervals
    (preds, scale/params, or train residuals for the empirical model).
    Returns a DataFrame that experiments write to results/tables/.
    """
    raise NotImplementedError

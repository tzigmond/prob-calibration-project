"""Prediction intervals per model.

Given fitted predictions and the distribution's fitted scale (and nu for
Student-t), produce lower/upper bound arrays at a nominal ``level``. The
parametric builders consume the scale estimates produced in ``losses.py`` —
closing the loop from "the assumption that defined the loss" to "the interval
that assumption implies."

Five models total:
  1. gaussian_interval          (fixed σ)
  2. ewma_scaled_gaussian_interval  (time-varying σ_t via EWMA — the
     heteroskedasticity control that tests whether the fat tail is just
     volatility clustering in disguise)
  3. laplace_interval
  4. student_t_interval
  5. empirical_interval         (non-parametric historical simulation)

``level`` is the nominal central coverage, e.g. 0.95 -> the 2.5%/97.5% quantiles.
``scale`` may be a scalar (fixed-variance models) or a per-point array (EWMA).
"""
from __future__ import annotations

import numpy as np
from scipy import stats


def gaussian_interval(
    preds: np.ndarray, scale: float | np.ndarray, level: float
) -> tuple[np.ndarray, np.ndarray]:
    """preds ± z_quantile(level) * scale. ``scale`` broadcasts, so a per-point
    array yields the EWMA-scaled variant (see ewma_scaled_gaussian_interval)."""
    # z = stats.norm.ppf(0.5 + level/2); return preds - z*scale, preds + z*scale
    raise NotImplementedError


def ewma_volatility(series: np.ndarray, lam: float = 0.94) -> np.ndarray:
    """Trailing RiskMetrics EWMA volatility, one σ_t per point.

    σ²_t = lam * σ²_{t-1} + (1 - lam) * series_{t-1}²  (uses only past values, so
    no lookahead). ``series`` is returns or residuals; lam=0.94 is the daily
    RiskMetrics default. Returns an array of σ_t aligned to the prediction points.
    """
    raise NotImplementedError


def ewma_scaled_gaussian_interval(
    preds: np.ndarray, ewma_scale: np.ndarray, level: float
) -> tuple[np.ndarray, np.ndarray]:
    """Model 2: Gaussian interval with per-point σ_t from ``ewma_volatility``.

    Thin wrapper over ``gaussian_interval`` with an array scale — exists as a
    named function so experiments read cleanly and the roster is explicit.
    """
    # return gaussian_interval(preds, ewma_scale, level)
    raise NotImplementedError


def laplace_interval(
    preds: np.ndarray, scale: float, level: float
) -> tuple[np.ndarray, np.ndarray]:
    """preds ± Laplace-quantile(level) * scale."""
    # q = stats.laplace.ppf(0.5 + level/2, scale=scale)
    raise NotImplementedError


def student_t_interval(
    preds: np.ndarray, scale: float, nu: float, level: float
) -> tuple[np.ndarray, np.ndarray]:
    """preds ± t_quantile(level, nu) * scale."""
    # q = stats.t.ppf(0.5 + level/2, df=nu); return preds - q*scale, preds + q*scale
    raise NotImplementedError


def empirical_interval(
    preds: np.ndarray, train_residuals: np.ndarray, level: float
) -> tuple[np.ndarray, np.ndarray]:
    """Model 5: non-parametric historical simulation.

    Take the empirical (level)-central quantiles of the TRAIN residuals — no
    distributional assumption — and add them to each prediction. The baseline
    that answers "does the parametric assumption even matter, versus just using
    the right empirical quantiles?"
    """
    # lo_q, hi_q = np.quantile(train_residuals, [0.5 - level/2, 0.5 + level/2])
    # return preds + lo_q, preds + hi_q
    raise NotImplementedError

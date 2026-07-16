"""Prediction intervals per model.

Given fitted predictions and the distribution's fitted scale (and nu for
Student-t), produce lower/upper bound arrays at a nominal ``level``. The
parametric builders consume the scale estimates produced in ``losses.py`` —
closing the loop from "the assumption that defined the loss" to "the interval
that assumption implies."

Five models total:
  1. gaussian_interval              (fixed σ)
  2. ewma_scaled_gaussian_interval  (time-varying σ_t via EWMA — the
     heteroskedasticity control that tests whether the fat tail is just
     volatility clustering in disguise)
  3. laplace_interval
  4. student_t_interval
  5. empirical_interval             (non-parametric historical simulation)

``level`` is the nominal central coverage, e.g. 0.95 -> the 2.5%/97.5% quantiles.
``scale`` may be a scalar (fixed-variance models) or a per-point array (EWMA).
"""
from __future__ import annotations

import numpy as np
from scipy import stats


def _central_tail(level: float) -> float:
    """Upper-tail probability for a central interval: 0.95 -> 0.975."""
    return 0.5 + level / 2.0


def gaussian_interval(
    preds: np.ndarray, scale: float | np.ndarray, level: float
) -> tuple[np.ndarray, np.ndarray]:
    """preds ± z_quantile(level) * scale. ``scale`` broadcasts, so a per-point
    array yields the EWMA-scaled variant (see ewma_scaled_gaussian_interval)."""
    z = stats.norm.ppf(_central_tail(level))
    half = z * scale
    return preds - half, preds + half


def ewma_volatility(
    series: np.ndarray, lam: float = 0.94, seed_var: float | None = None
) -> np.ndarray:
    """Trailing RiskMetrics EWMA volatility, one σ_t per point.

    σ²_t = lam * σ²_{t-1} + (1 - lam) * series_{t-1}²  — each σ_t depends only on
    values strictly before t, so there is no lookahead. ``lam=0.94`` is the daily
    RiskMetrics default. ``series`` is returns or residuals.

    ``seed_var`` seeds σ²_0. Pass the *training* variance to avoid peeking at the
    test period; if omitted it defaults to the variance of ``series`` (a seed that
    is fully decayed away long before any sizeable test slice, but strictly a peek).
    """
    series = np.asarray(series, dtype=float)
    var = np.empty(len(series))
    if seed_var is not None:
        v = float(seed_var)
    else:
        v = float(np.var(series)) if len(series) else 0.0
    for t in range(len(series)):
        var[t] = v                      # forecast for point t (uses only past)
        v = lam * v + (1.0 - lam) * series[t] ** 2
    return np.sqrt(var)


def ewma_scaled_gaussian_interval(
    preds: np.ndarray, ewma_scale: np.ndarray, level: float
) -> tuple[np.ndarray, np.ndarray]:
    """Model 2: Gaussian interval with per-point σ_t from ``ewma_volatility``."""
    return gaussian_interval(preds, ewma_scale, level)


def laplace_interval(
    preds: np.ndarray, scale: float, level: float
) -> tuple[np.ndarray, np.ndarray]:
    """preds ± Laplace-quantile(level) * scale (scale = b, the Laplace scale)."""
    half = stats.laplace.ppf(_central_tail(level), scale=scale)
    return preds - half, preds + half


def student_t_interval(
    preds: np.ndarray, scale: float, nu: float, level: float
) -> tuple[np.ndarray, np.ndarray]:
    """preds ± t_quantile(level, nu) * scale."""
    q = stats.t.ppf(_central_tail(level), df=nu)
    half = q * scale
    return preds - half, preds + half


def empirical_interval(
    preds: np.ndarray, train_residuals: np.ndarray, level: float
) -> tuple[np.ndarray, np.ndarray]:
    """Model 5: non-parametric historical simulation.

    Take the empirical central-``level`` quantiles of the TRAIN residuals — no
    distributional assumption — and add them to each prediction. The baseline
    that answers "does the parametric assumption even matter, versus just using
    the right empirical quantiles?"
    """
    lo_q, hi_q = np.quantile(train_residuals, [0.5 - level / 2.0, 0.5 + level / 2.0])
    return preds + lo_q, preds + hi_q

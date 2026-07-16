"""The likelihood -> loss mapping.

Each error model exposes its negative log-likelihood (``loss``), its gradient
w.r.t. the regression weights (``gradient``), and the fitted scale needed later
for interval construction (``estimate_scale``). Keeping the scale estimator with
the distribution that defines it is deliberate: ``intervals.py`` reads it back
out per model.

Gaussian NLL (up to constants) == MSE. Laplace NLL == MAE. Student-t NLL is the
heavy-tailed one; its ``nu`` is FIXED at construction (from ``estimate_nu``) and
never optimized here.

Convention: for a linear model, residual r = X @ w - y (prediction minus target).
"""
from __future__ import annotations

import numpy as np
from scipy import stats


class Loss:
    """Common interface. All operate on a linear model: preds = X @ w."""

    def loss(self, X: np.ndarray, y: np.ndarray, w: np.ndarray) -> float:
        raise NotImplementedError

    def gradient(self, X: np.ndarray, y: np.ndarray, w: np.ndarray) -> np.ndarray:
        """Gradient of ``loss`` w.r.t. weights ``w``."""
        raise NotImplementedError

    def estimate_scale(self, residuals: np.ndarray) -> float:
        """Fitted scale/σ for this distribution, used by intervals.py."""
        raise NotImplementedError


class GaussianLoss(Loss):
    """Gaussian errors -> MSE. Gradient is the standard Xᵀ(Xw - y) form."""

    def loss(self, X, y, w):
        r = X @ w - y
        return float(np.mean(r ** 2))

    def gradient(self, X, y, w):
        r = X @ w - y
        return (2.0 / len(y)) * (X.T @ r)

    def estimate_scale(self, residuals):
        # σ = MLE standard deviation of residuals
        return float(np.std(residuals))


class LaplaceLoss(Loss):
    """Laplace errors -> MAE. Non-differentiable at zero; we take subgradient=0
    there (documented interview point)."""

    def loss(self, X, y, w):
        r = X @ w - y
        return float(np.mean(np.abs(r)))

    def gradient(self, X, y, w):
        r = X @ w - y
        # np.sign gives sign(0) == 0 — exactly the subgradient choice we want.
        return (1.0 / len(y)) * (X.T @ np.sign(r))

    def estimate_scale(self, residuals):
        # Laplace MLE scale b = mean(|residuals - median|)
        return float(np.mean(np.abs(residuals - np.median(residuals))))


class StudentTLoss(Loss):
    """Student-t errors -> heavy-tailed NLL, for a FIXED ``nu``.

    Gradient is w.r.t. weights only; nu comes from ``estimate_nu`` and is frozen.
    The per-residual weighting (nu+1) * r / (nu*scale**2 + r**2) is the
    redescending influence that makes this robust — and non-convex in w.
    """

    def __init__(self, nu: float, scale: float = 1.0):
        self.nu = nu
        self.scale = scale  # may be refreshed via estimate_scale before intervals

    def loss(self, X, y, w):
        r = X @ w - y
        # True scaled-t NLL (constants included via scipy for correctness).
        return float(-np.mean(stats.t.logpdf(r, df=self.nu, scale=self.scale)))

    def gradient(self, X, y, w):
        r = X @ w - y
        weight = (self.nu + 1.0) * r / (self.nu * self.scale ** 2 + r ** 2)
        return (1.0 / len(y)) * (X.T @ weight)

    def estimate_scale(self, residuals):
        # Fit scale with df fixed to self.nu and location pinned at 0.
        _, _, scale = stats.t.fit(residuals, f0=self.nu, floc=0.0)
        return float(scale)


def estimate_nu(residuals: np.ndarray) -> float:
    """Fit a Student-t to residuals and return the estimated degrees of freedom.

    Called once on OLS residuals before the Student-t model is trained; the
    returned nu is then baked into ``StudentTLoss(nu=...)`` and held fixed.
    Keep df only here — scale is re-estimated per model via ``estimate_scale``.
    """
    df, _, _ = stats.t.fit(residuals)
    return float(df)

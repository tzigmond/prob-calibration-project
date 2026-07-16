"""The likelihood -> loss mapping.

Each error model exposes its negative log-likelihood (``loss``), its gradient
w.r.t. the regression weights (``gradient``), and the fitted scale needed later
for interval construction (``estimate_scale``). Keeping the scale estimator with
the distribution that defines it is deliberate: ``intervals.py`` reads it back
out per model.

Gaussian NLL (up to constants) == MSE. Laplace NLL == MAE. Student-t NLL is the
heavy-tailed one; its ``nu`` is FIXED at construction (from ``estimate_nu``) and
never optimized here.
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
        # mean( (X@w - y)**2 )   (Gaussian NLL up to additive/scale constants)
        raise NotImplementedError

    def gradient(self, X, y, w):
        # (2/n) * X.T @ (X@w - y)
        raise NotImplementedError

    def estimate_scale(self, residuals):
        # σ = std of residuals (MLE up to bias)
        raise NotImplementedError


class LaplaceLoss(Loss):
    """Laplace errors -> MAE. Non-differentiable at zero; we take subgradient=0
    there (documented interview point — keep this an explicit, commented line)."""

    def loss(self, X, y, w):
        # mean( |X@w - y| )
        raise NotImplementedError

    def gradient(self, X, y, w):
        # (1/n) * X.T @ sign(X@w - y)   with sign(0) := 0  (subgradient choice)
        raise NotImplementedError

    def estimate_scale(self, residuals):
        # Laplace MLE scale b = mean(|residuals - median|)
        raise NotImplementedError


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
        # sum of -log t_pdf((X@w - y)/scale; nu) / scale, up to constants
        raise NotImplementedError

    def gradient(self, X, y, w):
        # r = X@w - y
        # weight = (nu + 1) * r / (nu * scale**2 + r**2)
        # grad = (1/n) * X.T @ weight
        raise NotImplementedError

    def estimate_scale(self, residuals):
        # scale from scipy.stats.t.fit with df fixed to self.nu (keep scale only)
        raise NotImplementedError


def estimate_nu(residuals: np.ndarray) -> float:
    """Fit a Student-t to residuals and return the estimated degrees of freedom.

    Called once on OLS residuals before the Student-t model is trained; the
    returned nu is then baked into ``StudentTLoss(nu=...)`` and held fixed.
    Keep df only here — scale is re-estimated per model via ``estimate_scale``.
    """
    # df, loc, scale = scipy.stats.t.fit(residuals); return df
    raise NotImplementedError

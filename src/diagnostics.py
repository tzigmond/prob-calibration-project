"""Residual-distribution evidence.

Confirms the distributional assumption actually fits — not merely that intervals
happened to cover. Well-calibrated intervals are necessary but not sufficient
evidence for a distribution; these diagnostics close that gap. All plotting
functions save to a caller-supplied path.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def _standardize(residuals: np.ndarray) -> np.ndarray:
    residuals = np.asarray(residuals, dtype=float)
    return (residuals - np.mean(residuals)) / np.std(residuals)


def plot_residual_hist(residuals: np.ndarray, fitted_dists: dict, save_path: str) -> None:
    """Histogram of standardized residuals with fitted densities overlaid.

    ``fitted_dists`` maps a label to a frozen scipy distribution (already scaled
    to standardized residuals, e.g. ``stats.norm(0, 1)``, ``stats.t(nu)``).
    """
    import matplotlib.pyplot as plt

    z = _standardize(residuals)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(z, bins=80, density=True, alpha=0.4, color="0.5", label="standardized residuals")
    grid = np.linspace(z.min(), z.max(), 500)
    for label, dist in fitted_dists.items():
        ax.plot(grid, dist.pdf(grid), lw=1.8, label=label)
    ax.set_xlabel("standardized residual")
    ax.set_ylabel("density")
    ax.set_title("Residual distribution vs. fitted densities")
    ax.legend()
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


def qq_plot(residuals: np.ndarray, dist, save_path: str) -> None:
    """Q-Q plot of standardized residuals against an assumed distribution ``dist``
    (a frozen scipy distribution, e.g. ``stats.t(nu)``)."""
    import matplotlib.pyplot as plt

    z = np.sort(_standardize(residuals))
    n = len(z)
    probs = (np.arange(1, n + 1) - 0.5) / n
    theo = dist.ppf(probs)

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(theo, z, s=8, alpha=0.5)
    lo = min(theo[np.isfinite(theo)].min(), z.min())
    hi = max(theo[np.isfinite(theo)].max(), z.max())
    ax.plot([lo, hi], [lo, hi], color="crimson", lw=1.2, label="y = x")
    ax.set_xlabel("theoretical quantile")
    ax.set_ylabel("empirical quantile")
    ax.set_title("Q-Q plot")
    ax.legend()
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


def tail_probabilities(residuals: np.ndarray, fitted_dists: dict) -> pd.DataFrame:
    """Empirical vs. modeled P(|z| > 2), P(|z| > 3), P(|z| > 4).

    ``fitted_dists`` maps a label to a frozen scipy distribution over standardized
    residuals. Returns a small DataFrame: rows = thresholds, columns = empirical
    plus one per model.
    """
    z = _standardize(residuals)
    thresholds = [2, 3, 4]
    data: dict[str, list[float]] = {
        "empirical": [float(np.mean(np.abs(z) > k)) for k in thresholds]
    }
    for label, dist in fitted_dists.items():
        data[label] = [float(dist.sf(k) + dist.cdf(-k)) for k in thresholds]
    return pd.DataFrame(data, index=[f"P(|z|>{k})" for k in thresholds])

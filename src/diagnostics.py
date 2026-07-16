"""Residual-distribution evidence.

Confirms the distributional assumption actually fits — not merely that intervals
happened to cover. All plotting functions save to results/figures/.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def plot_residual_hist(residuals: np.ndarray, fitted_dists: dict, save_path: str) -> None:
    """Histogram of standardized residuals with fitted Gaussian/Laplace/Student-t
    densities overlaid. ``fitted_dists`` maps a label to a frozen scipy dist."""
    raise NotImplementedError


def qq_plot(residuals: np.ndarray, dist, save_path: str) -> None:
    """Q-Q plot of residuals against an assumed distribution ``dist``."""
    raise NotImplementedError


def tail_probabilities(residuals: np.ndarray, fitted_dists: dict) -> pd.DataFrame:
    """Empirical vs. modeled P(|z| > 2), P(|z| > 3), P(|z| > 4) as a small table."""
    raise NotImplementedError

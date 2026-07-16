"""Shared calibration-study pipeline for the real-data experiments (02, 03).

Private helper (underscore-prefixed) so experiments 02 and 03 don't duplicate the
fit -> intervals -> coverage-table -> diagnostics wiring. This is orchestration,
not library code, which is why it lives under experiments/ rather than src/.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
from scipy import stats

from src import optimizers as O
from src.losses import GaussianLoss, LaplaceLoss, StudentTLoss, estimate_nu
from src.train import fit
from src import intervals as I
from src import calibration as C
from src import diagnostics as D

LEVELS = [0.90, 0.95, 0.99]


def run_study(
    X: np.ndarray,
    y: np.ndarray,
    dataset_name: str,
    figures_dir: Path,
    train_frac: float = 0.8,
    epochs: int = 3000,
    lr: float = 0.01,
):
    """Fit the five models, build the coverage table, and write diagnostics.

    Returns (coverage_table_df, tail_prob_df, meta_dict).
    """
    figures_dir = Path(figures_dir)
    figures_dir.mkdir(parents=True, exist_ok=True)
    split = int(len(y) * train_frac)
    Xtr, ytr, Xte, yte = X[:split], y[:split], X[split:], y[split:]

    # --- Point model: least-squares mean (shared by Gaussian/EWMA/Empirical) ---
    gauss = GaussianLoss()
    w_g = fit(Xtr, ytr, gauss, O.Adam(lr=lr), epochs=epochs).weights
    preds_te = Xte @ w_g
    resid_tr = ytr - Xtr @ w_g
    sigma = gauss.estimate_scale(resid_tr)

    # EWMA σ_t over the full residual series; take the test slice (no lookahead).
    # Seed σ²_0 from the TRAIN variance only, so the test period is never peeked at.
    resid_full = y - X @ w_g
    seed_var = float(np.var(resid_full[:split]))
    sigma_t = I.ewma_volatility(resid_full, seed_var=seed_var)[split:]

    # --- Laplace model (its own weights + scale) ---
    lap = LaplaceLoss()
    w_l = fit(Xtr, ytr, lap, O.Adam(lr=lr), epochs=epochs).weights
    preds_l = Xte @ w_l
    b = lap.estimate_scale(ytr - Xtr @ w_l)

    # --- Student-t model: estimate ν on OLS residuals, hold fixed ---
    nu = estimate_nu(resid_tr)
    tloss = StudentTLoss(nu=nu)
    tloss.scale = tloss.estimate_scale(resid_tr)
    w_t = fit(Xtr, ytr, tloss, O.Adam(lr=lr), epochs=epochs).weights
    preds_t = Xte @ w_t
    scale_t = tloss.estimate_scale(ytr - Xtr @ w_t)

    models = {
        "Gaussian":      lambda lv: I.gaussian_interval(preds_te, sigma, lv),
        "EWMA-Gaussian": lambda lv: I.ewma_scaled_gaussian_interval(preds_te, sigma_t, lv),
        "Laplace":       lambda lv: I.laplace_interval(preds_l, b, lv),
        "Student-t":     lambda lv: I.student_t_interval(preds_t, scale_t, nu, lv),
        "Empirical":     lambda lv: I.empirical_interval(preds_te, resid_tr, lv),
    }
    table = C.coverage_table(yte, models, LEVELS)

    # --- Diagnostics on standardized Gaussian residuals (unit-variance overlays) ---
    unit_t_scale = np.sqrt((nu - 2) / nu) if nu > 2 else 1.0
    fitted = {
        "Normal": stats.norm(0, 1),
        "Laplace": stats.laplace(scale=1 / np.sqrt(2)),
        f"Student-t(nu={nu:.1f})": stats.t(df=nu, scale=unit_t_scale),
    }
    slug = dataset_name.lower().replace(" ", "_")
    D.plot_residual_hist(resid_tr, fitted, figures_dir / f"{slug}_residual_hist.png")
    D.qq_plot(resid_tr, stats.t(df=nu, scale=unit_t_scale),
              figures_dir / f"{slug}_qq_studentt.png",
              title=f"{dataset_name}: Q-Q vs Student-t(nu={nu:.1f})")
    D.qq_plot(resid_tr, stats.norm(0, 1), figures_dir / f"{slug}_qq_normal.png",
              title=f"{dataset_name}: Q-Q vs Normal")
    D.plot_calibration(table, dataset_name, figures_dir / f"{slug}_calibration.png")
    tails = D.tail_probabilities(resid_tr, fitted)

    meta = {
        "dataset": dataset_name,
        "n_train": split,
        "n_test": len(yte),
        "nu": nu,
        "excess_kurtosis": float(stats.kurtosis(resid_tr, fisher=True)),
    }
    return table, tails, meta

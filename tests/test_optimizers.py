"""Optimizers recover the closed-form Gaussian solution, and match scikit-learn."""
import numpy as np
import pytest
from sklearn.linear_model import LinearRegression

from src import optimizers as O
from src.losses import GaussianLoss
from src.train import fit


@pytest.fixture
def gaussian_problem():
    rng = np.random.default_rng(0)
    n = 3000
    X = np.column_stack([np.ones(n), rng.standard_normal((n, 3))])
    w_true = np.array([0.5, -1.2, 2.0, 0.7])
    y = X @ w_true + rng.normal(0, 0.5, n)
    w_ols = np.linalg.lstsq(X, y, rcond=None)[0]
    return X, y, w_ols


@pytest.mark.parametrize("opt_factory,batch,tol", [
    (lambda: O.BatchGD(lr=0.1), None, 1e-2),
    (lambda: O.SGD(lr=0.05), 256, 5e-2),
    (lambda: O.Momentum(lr=0.05), 256, 5e-2),
    (lambda: O.Adam(lr=0.05), 256, 8e-2),
])
def test_optimizer_recovers_ols(gaussian_problem, opt_factory, batch, tol):
    X, y, w_ols = gaussian_problem
    w = fit(X, y, GaussianLoss(), opt_factory(), epochs=600, batch_size=batch, seed=0).weights
    assert np.max(np.abs(w - w_ols)) < tol


def test_batchgd_matches_sklearn(gaussian_problem):
    X, y, _ = gaussian_problem
    w = fit(X, y, GaussianLoss(), O.BatchGD(lr=0.1), epochs=2000, seed=0).weights
    # X already carries an intercept column, so disable sklearn's own intercept.
    skl = LinearRegression(fit_intercept=False).fit(X, y).coef_
    assert np.allclose(w, skl, atol=1e-2)


def test_loss_history_decreases(gaussian_problem):
    X, y, _ = gaussian_problem
    # Start away from the optimum (default init is OLS, i.e. already minimal).
    w0 = np.zeros(X.shape[1])
    hist = fit(X, y, GaussianLoss(), O.BatchGD(lr=0.1), epochs=200, w_init=w0, seed=0).loss_history
    assert hist[-1] < hist[0]

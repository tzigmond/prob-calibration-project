"""Loss values and analytic gradients are correct (finite-difference checked)."""
import numpy as np
import pytest
from scipy import stats

from src.losses import GaussianLoss, LaplaceLoss, StudentTLoss, estimate_nu


def _numeric_grad(loss, X, y, w, eps=1e-6):
    g = np.zeros_like(w)
    for i in range(len(w)):
        wp, wm = w.copy(), w.copy()
        wp[i] += eps
        wm[i] -= eps
        g[i] = (loss.loss(X, y, wp) - loss.loss(X, y, wm)) / (2 * eps)
    return g


@pytest.fixture
def problem():
    rng = np.random.default_rng(7)
    n = 400
    X = np.column_stack([np.ones(n), rng.standard_normal((n, 3))])
    y = rng.standard_normal(n)
    w = rng.standard_normal(4) * 0.3       # away from any residual==0 kink
    return X, y, w


@pytest.mark.parametrize("loss", [GaussianLoss(), LaplaceLoss(), StudentTLoss(nu=5.0, scale=1.0)])
def test_gradient_matches_numeric(problem, loss):
    X, y, w = problem
    analytic = loss.gradient(X, y, w)
    numeric = _numeric_grad(loss, X, y, w)
    assert np.allclose(analytic, numeric, atol=1e-5, rtol=1e-4)


def test_gaussian_loss_is_mse(problem):
    X, y, w = problem
    r = X @ w - y
    assert GaussianLoss().loss(X, y, w) == pytest.approx(np.mean(r ** 2))


def test_laplace_loss_is_mae(problem):
    X, y, w = problem
    r = X @ w - y
    assert LaplaceLoss().loss(X, y, w) == pytest.approx(np.mean(np.abs(r)))


def test_estimate_nu_recovers_true():
    r = stats.t.rvs(df=4.0, scale=0.5, size=20000, random_state=0)
    assert abs(estimate_nu(r) - 4.0) < 1.0


def test_estimate_scale_gaussian_is_std():
    rng = np.random.default_rng(0)
    resid = rng.normal(0, 2.0, 10000)
    assert GaussianLoss().estimate_scale(resid) == pytest.approx(np.std(resid))

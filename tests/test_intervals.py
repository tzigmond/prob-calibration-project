"""Interval builders hit nominal coverage and respect the no-lookahead rule."""
import numpy as np
import pytest
from scipy import stats

from src import intervals as I


def test_gaussian_interval_nominal_coverage():
    rng = np.random.default_rng(0)
    y = rng.normal(0, 1.0, 200000)
    lo, hi = I.gaussian_interval(np.zeros_like(y), 1.0, 0.95)
    cov = np.mean((y >= lo) & (y <= hi))
    assert cov == pytest.approx(0.95, abs=0.01)


def test_student_t_interval_matches_ppf():
    lo, hi = I.student_t_interval(np.array([0.0]), scale=2.0, nu=5.0, level=0.90)
    expected = stats.t.ppf(0.95, df=5.0) * 2.0
    assert hi[0] == pytest.approx(expected)
    assert lo[0] == pytest.approx(-expected)


def test_empirical_interval_uses_residual_quantiles():
    resid = np.linspace(-1, 1, 1001)          # symmetric, known quantiles
    preds = np.array([10.0])
    lo, hi = I.empirical_interval(preds, resid, 0.90)
    q = np.quantile(resid, [0.05, 0.95])
    assert lo[0] == pytest.approx(10.0 + q[0])
    assert hi[0] == pytest.approx(10.0 + q[1])


def test_ewma_volatility_no_lookahead():
    rng = np.random.default_rng(3)
    series = rng.normal(0, 1.0, 500)
    base = I.ewma_volatility(series, seed_var=1.0)
    # Perturb a FUTURE value; σ_t at/earlier than it must be unchanged.
    t = 200
    perturbed = series.copy()
    perturbed[t] += 100.0
    after = I.ewma_volatility(perturbed, seed_var=1.0)
    assert np.allclose(base[: t + 1], after[: t + 1])   # includes σ_t itself
    assert not np.isclose(base[t + 1], after[t + 1])     # next point does change


def test_ewma_seed_var_controls_first_point():
    series = np.zeros(5)
    out = I.ewma_volatility(series, seed_var=4.0)
    assert out[0] == pytest.approx(2.0)                  # sqrt(seed_var)

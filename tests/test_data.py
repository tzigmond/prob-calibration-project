"""Feature construction, splitting, and log returns (no network)."""
import numpy as np
import pandas as pd
import pytest

from src import data as D


def test_compute_log_returns_matches_definition():
    prices = pd.Series([100.0, 110.0, 99.0], index=pd.date_range("2020-01-01", periods=3))
    r = D.compute_log_returns(prices)
    assert len(r) == 2
    assert r.iloc[0] == pytest.approx(np.log(110.0 / 100.0))
    assert r.iloc[1] == pytest.approx(np.log(99.0 / 110.0))


def test_build_ar_features_shape_and_alignment():
    returns = pd.Series(np.arange(1, 11, dtype=float),
                        index=pd.date_range("2020-01-01", periods=10))
    X, y = D.build_ar_features(returns, lags=3)
    # 10 rows minus 3 dropped for the lag warmup.
    assert X.shape == (7, 5)   # intercept + 3 lags + |r_lag1|
    assert y.shape == (7,)
    assert np.all(X[:, 0] == 1.0)                       # intercept column
    # First usable target is r_4 = 4.0; its lag1 is 3.0, lag2 2.0, lag3 1.0.
    assert y[0] == pytest.approx(4.0)
    assert X[0, 1] == pytest.approx(3.0)
    assert X[0, 2] == pytest.approx(2.0)
    assert X[0, 3] == pytest.approx(1.0)
    assert X[0, 4] == pytest.approx(3.0)                # |r_lag1|


def test_chronological_split_preserves_order_and_sizes():
    X = np.arange(100).reshape(100, 1).astype(float)
    y = np.arange(100).astype(float)
    Xtr, ytr, Xte, yte = D.chronological_split(X, y, train_frac=0.8)
    assert len(ytr) == 80 and len(yte) == 20
    assert ytr[0] == 0 and ytr[-1] == 79          # train is the earlier block
    assert yte[0] == 80 and yte[-1] == 99         # test is the later block


def test_build_market_model_aligns_and_adds_intercept():
    idx = pd.date_range("2020-01-01", periods=5)
    stock = pd.Series([0.1, 0.2, np.nan, 0.4, 0.5], index=idx)
    market = pd.Series([0.05, 0.06, 0.07, np.nan, 0.09], index=idx)
    X, y = D.build_market_model(stock, market)
    # Rows with a NaN in either series are dropped -> 3 usable dates.
    assert len(y) == 3
    assert X.shape == (3, 2)
    assert np.all(X[:, 0] == 1.0)

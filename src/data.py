"""Everything between raw internet data and a clean (X, y) split chronologically.

Nothing downstream should ever touch yfinance directly — this module is the sole
boundary. Pulls are cached to results/*.csv so repeated runs work offline and
survive yfinance outages.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def fetch_prices(ticker: str, start: str, end: str) -> pd.DataFrame:
    """Pull raw OHLCV from yfinance, caching to results/<ticker>.csv.

    On a cache hit, read the CSV instead of re-hitting the API. Sanitize on the
    way in: drop non-positive prices and duplicate dates so one bad tick can't
    blow up a log return.
    """
    raise NotImplementedError


def compute_log_returns(prices: pd.Series) -> pd.Series:
    """Daily log returns: log(p_t) - log(p_{t-1})."""
    raise NotImplementedError


def build_ar_features(returns: pd.Series, lags: int = 3) -> tuple[np.ndarray, np.ndarray]:
    """Design matrix (r_{t-1}, r_{t-2}, r_{t-3}, |r_{t-1}|), target r_t.

    Drops the leading rows where the lags don't yet exist. Returns (X, y).
    """
    raise NotImplementedError


def build_market_model(
    stock_returns: pd.Series, market_returns: pd.Series
) -> tuple[np.ndarray, np.ndarray]:
    """AAPL-on-SPY variant: target = stock return, predictor = contemporaneous
    market return. Residual = idiosyncratic return. Used only by experiment 03."""
    raise NotImplementedError


def chronological_split(
    X: np.ndarray, y: np.ndarray, train_frac: float = 0.8
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Split by time order, NO shuffling — the single enforcement point for the
    no-lookahead rule. Returns (X_train, y_train, X_test, y_test)."""
    raise NotImplementedError

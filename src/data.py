"""Everything between raw internet data and a clean (X, y) split chronologically.

Nothing downstream should ever touch yfinance directly — this module is the sole
boundary. Pulls are cached to results/*.csv so repeated runs work offline and
survive yfinance outages.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

_CACHE_DIR = Path(__file__).resolve().parents[1] / "results"


def fetch_prices(ticker: str, start: str, end: str, use_cache: bool = True) -> pd.DataFrame:
    """Pull raw OHLCV from yfinance, caching to results/<ticker>.csv.

    On a cache hit, read the CSV instead of re-hitting the API. Sanitize on the
    way in: drop non-positive closes and duplicate dates, and sort by date, so a
    single bad tick can't blow up a log return.
    """
    _CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache = _CACHE_DIR / f"{ticker}.csv"

    if use_cache and cache.exists():
        df = pd.read_csv(cache, index_col=0, parse_dates=True)
    else:
        import yfinance as yf

        df = yf.download(ticker, start=start, end=end, auto_adjust=False, progress=False)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        df.to_csv(cache)

    df = df[df["Close"] > 0]
    df = df[~df.index.duplicated(keep="first")]
    return df.sort_index()


def compute_log_returns(prices: pd.Series) -> pd.Series:
    """Daily log returns: log(p_t) - log(p_{t-1})."""
    return np.log(prices).diff().dropna()


def build_ar_features(returns: pd.Series, lags: int = 3) -> tuple[np.ndarray, np.ndarray]:
    """Design matrix and target for the AR(lags) + lagged-|r| model.

    Columns: [intercept, r_{t-1}, ..., r_{t-lags}, |r_{t-1}|]; target r_t. Leading
    rows without a full lag history are dropped. Returns (X, y) as float arrays.
    """
    df = pd.DataFrame({"r": returns})
    for k in range(1, lags + 1):
        df[f"r_lag{k}"] = df["r"].shift(k)
    df["abs_lag1"] = df["r"].shift(1).abs()
    df = df.dropna()

    y = df["r"].to_numpy(dtype=float)
    feats = df[[f"r_lag{k}" for k in range(1, lags + 1)] + ["abs_lag1"]].to_numpy(dtype=float)
    X = np.column_stack([np.ones(len(feats)), feats])
    return X, y


def build_market_model(
    stock_returns: pd.Series, market_returns: pd.Series
) -> tuple[np.ndarray, np.ndarray]:
    """AAPL-on-SPY variant: target = stock return, predictor = contemporaneous
    market return (plus intercept). Residual = idiosyncratic return. Used by
    experiment 03. Returns (X, y) aligned on their shared dates."""
    joined = pd.concat([stock_returns.rename("stock"), market_returns.rename("mkt")],
                       axis=1).dropna()
    y = joined["stock"].to_numpy(dtype=float)
    X = np.column_stack([np.ones(len(joined)), joined["mkt"].to_numpy(dtype=float)])
    return X, y


def chronological_split(
    X: np.ndarray, y: np.ndarray, train_frac: float = 0.8
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Split by time order, NO shuffling — the single enforcement point for the
    no-lookahead rule. Returns (X_train, y_train, X_test, y_test)."""
    split = int(len(y) * train_frac)
    return X[:split], y[:split], X[split:], y[split:]

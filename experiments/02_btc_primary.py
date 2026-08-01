"""Experiment 02 - the core BTC result.

Pull BTC-USD, build AR(3)+|r| features on daily log returns, split chronologically,
fit the five models, and produce the centerpiece coverage/width table (with
binomial CIs and Kupiec p-values) plus residual diagnostics.

The question this answers: do heavy-tailed and volatility-aware models produce
better-calibrated intervals than a fixed-variance Gaussian - and does any
Student-t advantage survive the EWMA-Gaussian and empirical baselines?
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
try:
    sys.stdout.reconfigure(encoding="utf-8")  # avoid cp1252 console encode errors
except Exception:
    pass

from src import data as D
from _pipeline import run_study

ROOT = Path(__file__).resolve().parents[1]
TABLES = ROOT / "results" / "tables"
FIGURES = ROOT / "results" / "figures"

# Pinned window for reproducible reported results; --refresh re-pulls this window.
START = "2015-01-01"
END = "2026-07-01"


def main(refresh: bool = False):
    prices = D.fetch_prices("BTC-USD", start=START, end=END, use_cache=not refresh)
    returns = D.compute_log_returns(prices["Close"])
    X, y = D.build_ar_features(returns, lags=3)

    table, tails, meta = run_study(X, y, dataset_name="BTC", figures_dir=FIGURES)

    TABLES.mkdir(parents=True, exist_ok=True)
    table.to_csv(TABLES / "btc_coverage.csv")
    tails.to_csv(TABLES / "btc_tail_probabilities.csv")

    print(f"BTC daily log returns - train {meta['n_train']}, test {meta['n_test']}")
    print(f"Estimated nu = {meta['nu']:.2f}   residual excess kurtosis = {meta['excess_kurtosis']:.2f}\n")
    print("Coverage / width table (nominal 90/95/99%):")
    print(table.round(3).to_string())
    print("\nStandardized-residual tail probabilities:")
    print(tails.round(4).to_string())
    print(f"\nTables written to {TABLES}\nFigures written to {FIGURES}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="BTC interval-calibration experiment")
    ap.add_argument("--refresh", action="store_true",
                    help="re-pull the latest data from Yahoo instead of using the cache")
    main(refresh=ap.parse_args().refresh)

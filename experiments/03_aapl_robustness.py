"""Experiment 03 — AAPL-on-SPY robustness check.

Re-run the same calibration pipeline on a different, more canonical setup: a
market-model regression of AAPL returns on SPY returns, where the residual is the
idiosyncratic return. If the BTC conclusion is real, it should not depend on the
quirks of one asset.
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
START = "2010-01-01"
END = "2026-07-01"


def main(refresh: bool = False):
    cache = not refresh
    aapl = D.compute_log_returns(D.fetch_prices("AAPL", START, END, use_cache=cache)["Close"])
    spy = D.compute_log_returns(D.fetch_prices("SPY", START, END, use_cache=cache)["Close"])
    X, y = D.build_market_model(aapl, spy)

    table, tails, meta = run_study(X, y, dataset_name="AAPL", figures_dir=FIGURES)

    TABLES.mkdir(parents=True, exist_ok=True)
    table.to_csv(TABLES / "aapl_coverage.csv")
    tails.to_csv(TABLES / "aapl_tail_probabilities.csv")

    print(f"AAPL-on-SPY idiosyncratic returns — train {meta['n_train']}, test {meta['n_test']}")
    print(f"Estimated nu = {meta['nu']:.2f}   residual excess kurtosis = {meta['excess_kurtosis']:.2f}\n")
    print("Coverage / width table (nominal 90/95/99%):")
    print(table.round(3).to_string())
    print("\nStandardized-residual tail probabilities:")
    print(tails.round(4).to_string())
    print(f"\nTables written to {TABLES}\nFigures written to {FIGURES}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="AAPL-on-SPY interval-calibration experiment")
    ap.add_argument("--refresh", action="store_true",
                    help="re-pull the latest data from Yahoo instead of using the cache")
    main(refresh=ap.parse_args().refresh)

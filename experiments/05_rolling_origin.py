"""Experiment 05 — rolling-origin (walk-forward) robustness.

The single-split result in exp 02 answers "is the calibration good on one held-out
window?". This asks the harder question a reviewer will: "is it robust to *where*
we split?" We walk an expanding training window forward across BTC history,
re-fit all five models at each step, and evaluate coverage on the next block. If
the headline conclusion is real it should hold across folds, not just one.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

from src import data as D
from src import calibration as C
from src.plotstyle import use_house_style, PALETTE
from _pipeline import fit_models, LEVELS

ROOT = Path(__file__).resolve().parents[1]
TABLES = ROOT / "results" / "tables"
FIGURES = ROOT / "results" / "figures"

START = "2015-01-01"
END = "2026-07-01"
N_FOLDS = 8
MIN_TRAIN_FRAC = 0.4
MODELS = ["Gaussian", "EWMA-Gaussian", "Laplace", "Student-t", "Empirical"]


def main(refresh: bool = False):
    prices = D.fetch_prices("BTC-USD", start=START, end=END, use_cache=not refresh)
    returns = D.compute_log_returns(prices["Close"])
    X, y = D.build_ar_features(returns, lags=3)
    n = len(y)
    min_train = int(MIN_TRAIN_FRAC * n)
    test_size = (n - min_train) // N_FOLDS

    cov = {m: {lv: [] for lv in LEVELS} for m in MODELS}
    rejects = {m: {lv: 0 for lv in LEVELS} for m in MODELS}

    print(f"Walk-forward on BTC: {N_FOLDS} expanding folds, test block ~{test_size} days each\n")
    for i in range(N_FOLDS):
        tr_end = min_train + i * test_size
        te_end = tr_end + test_size
        models, yte, _ = fit_models(X[:te_end], y[:te_end], tr_end, epochs=2000)
        for name, interval_fn in models.items():
            for lv in LEVELS:
                lo, hi = interval_fn(lv)
                covered = int(np.sum((yte >= lo) & (yte <= hi)))
                cov[name][lv].append(covered / len(yte))
                if C.kupiec_test(len(yte), len(yte) - covered, lv) < 0.05:
                    rejects[name][lv] += 1
        print(f"  fold {i + 1}/{N_FOLDS}: train {tr_end}, test {test_size}")

    # --- Summary: mean±std coverage and Kupiec rejection count across folds ---
    rows = {}
    for name in MODELS:
        row = {}
        for lv in LEVELS:
            arr = np.array(cov[name][lv])
            tag = f"{int(lv * 100)}%"
            row[(tag, "mean_cov")] = arr.mean()
            row[(tag, "std_cov")] = arr.std()
            row[(tag, "rejects")] = f"{rejects[name][lv]}/{N_FOLDS}"
        rows[name] = row
    summary = pd.DataFrame.from_dict(rows, orient="index")
    summary.columns = pd.MultiIndex.from_tuples(summary.columns, names=["level", "metric"])

    TABLES.mkdir(parents=True, exist_ok=True)
    summary.to_csv(TABLES / "btc_rolling_coverage.csv")
    print("\nRolling coverage summary (mean/std over folds, Kupiec rejections at 5%):")
    print(summary.to_string())

    # --- Plot: coverage across folds at 95% and 99% ---
    use_house_style()
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    folds = np.arange(1, N_FOLDS + 1)
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.6))
    for ax, lv in zip(axes, [0.95, 0.99]):
        for name in MODELS:
            ax.plot(folds, cov[name][lv], marker="o", markersize=5, lw=1.8,
                    color=PALETTE[name], label=name)
        ax.axhline(lv, ls="--", color="#999999", lw=1.3, label="nominal")
        ax.set_xlabel("rolling fold (expanding train window →)")
        ax.set_ylabel("empirical coverage")
        ax.set_title(f"BTC: {int(lv * 100)}% coverage across folds")
        ax.set_xticks(folds)
    axes[1].legend(title="model", bbox_to_anchor=(1.02, 1), loc="upper left")
    fig.suptitle("Rolling-origin calibration robustness — BTC", fontsize=15, fontweight="bold")
    fig.tight_layout()
    FIGURES.mkdir(parents=True, exist_ok=True)
    out = FIGURES / "btc_rolling_coverage.png"
    fig.savefig(out)
    plt.close(fig)
    print(f"\nRolling-coverage plot written to {out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="BTC rolling-origin robustness experiment")
    ap.add_argument("--refresh", action="store_true",
                    help="re-pull the latest data from Yahoo instead of using the cache")
    main(refresh=ap.parse_args().refresh)

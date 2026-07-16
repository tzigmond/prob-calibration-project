"""Experiment 04 — optimizer convergence sanity check.

Runs all four from-scratch optimizers against the Gaussian loss on BTC data and
confirms they converge to consistent coefficients (a convex problem, so they
should agree). Also fits the Student-t loss to illustrate the caveat: its NLL is
non-convex in the weights, so different optimizers can settle at slightly
different points from the same OLS start.

This is a sanity check, not a second experiment — it confirms the headline result
isn't an artifact of choosing Adam.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
try:
    sys.stdout.reconfigure(encoding="utf-8")  # avoid cp1252 console encode errors
except Exception:
    pass

from src import data as D
from src import optimizers as O
from src.losses import GaussianLoss, StudentTLoss, estimate_nu
from src.train import fit

ROOT = Path(__file__).resolve().parents[1]
FIGURES = ROOT / "results" / "figures"


def main():
    prices = D.fetch_prices("BTC-USD", start="2015-01-01", end="2025-01-01")
    returns = D.compute_log_returns(prices["Close"])
    X, y = D.build_ar_features(returns, lags=3)
    split = int(len(y) * 0.8)
    Xtr, ytr = X[:split], y[:split]

    optimizers = {
        "BatchGD":  O.BatchGD(lr=0.05),
        "SGD":      O.SGD(lr=0.02),
        "Momentum": O.Momentum(lr=0.02),
        "Adam":     O.Adam(lr=0.01),
    }

    print("Gaussian loss (convex) — optimizers should converge to the same weights:")
    histories = {}
    gauss_weights = {}
    for name, opt in optimizers.items():
        bs = None if name == "BatchGD" else 256
        res = fit(Xtr, ytr, GaussianLoss(), opt, epochs=2000, batch_size=bs)
        histories[name] = res.loss_history
        gauss_weights[name] = res.weights
        print(f"  {name:10s} final loss={res.loss_history[-1]:.3e}  weights={np.round(res.weights, 5)}")

    ref = gauss_weights["BatchGD"]
    spread = max(float(np.max(np.abs(w - ref))) for w in gauss_weights.values())
    print(f"  -> max weight spread across optimizers: {spread:.2e} "
          f"({'consistent' if spread < 1e-2 else 'check learning rates'})")

    print("\nStudent-t loss (non-convex) — expect small disagreement from shared OLS init:")
    nu = estimate_nu(ytr - Xtr @ np.linalg.lstsq(Xtr, ytr, rcond=None)[0])
    t_weights = {}
    for name, opt in optimizers.items():
        opt.reset()
        loss = StudentTLoss(nu=nu)
        loss.scale = loss.estimate_scale(ytr - Xtr @ np.linalg.lstsq(Xtr, ytr, rcond=None)[0])
        bs = None if name == "BatchGD" else 256
        res = fit(Xtr, ytr, loss, opt, epochs=2000, batch_size=bs)
        t_weights[name] = res.weights
        print(f"  {name:10s} weights={np.round(res.weights, 5)}")
    ref_t = t_weights["BatchGD"]
    spread_t = max(float(np.max(np.abs(w - ref_t))) for w in t_weights.values())
    print(f"  -> max weight spread (non-convex, expected larger): {spread_t:.2e}")

    # Convergence-trajectory plot for the Gaussian loss.
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    FIGURES.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 5))
    for name, hist in histories.items():
        ax.plot(hist, label=name, lw=1.4)
    ax.set_yscale("log")
    ax.set_xlabel("epoch")
    ax.set_ylabel("Gaussian loss (MSE)")
    ax.set_title("Optimizer convergence — Gaussian loss on BTC")
    ax.legend()
    fig.tight_layout()
    out = FIGURES / "optimizer_convergence.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"\nConvergence plot written to {out}")


if __name__ == "__main__":
    main()

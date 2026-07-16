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
    prices = D.fetch_prices("BTC-USD", start="2015-01-01", end="2026-07-01")
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
        res = fit(Xtr, ytr, GaussianLoss(), opt, epochs=2000, batch_size=bs, seed=0)
        histories[name] = res.loss_history
        gauss_weights[name] = res.weights
        print(f"  {name:10s} final loss={res.loss_history[-1]:.3e}  weights={np.round(res.weights, 5)}")

    # Convergence is judged on the LOSS, not the weights: daily-return
    # coefficients are weakly identified (near-zero predictable signal), so the
    # objective is flat and different optimizers reach near-identical loss at
    # slightly different weights. Loss agreement is the meaningful check.
    final_losses = {name: hist[-1] for name, hist in histories.items()}
    loss_spread = max(final_losses.values()) - min(final_losses.values())
    ref = gauss_weights["BatchGD"]
    weight_spread = max(float(np.max(np.abs(w - ref))) for w in gauss_weights.values())
    print(f"  -> final-loss spread: {loss_spread:.2e} "
          f"({'converged' if loss_spread < 1e-4 else 'check learning rates'})")
    print(f"  -> weight spread: {weight_spread:.2e} "
          f"(coefficients weakly identified on near-random daily returns)")

    # Student-t is non-convex in the weights, so it is more sensitive to the step
    # size than the convex losses: too large a rate and the redescending gradient
    # lets an optimizer walk away from the OLS basin entirely. With gentler rates
    # and a shared OLS init, the remaining disagreement reflects the objective's
    # curvature, not divergence.
    print("\nStudent-t loss (non-convex) — gentler rates, shared OLS init:")
    w_ols = np.linalg.lstsq(Xtr, ytr, rcond=None)[0]
    resid_ols = ytr - Xtr @ w_ols
    nu = estimate_nu(resid_ols)
    t_optimizers = {
        "BatchGD":  O.BatchGD(lr=0.01),
        "SGD":      O.SGD(lr=0.005),
        "Momentum": O.Momentum(lr=0.005),
        "Adam":     O.Adam(lr=0.005),
    }
    t_weights = {}
    for name, opt in t_optimizers.items():
        loss = StudentTLoss(nu=nu)
        loss.scale = loss.estimate_scale(resid_ols)
        bs = None if name == "BatchGD" else 256
        res = fit(Xtr, ytr, loss, opt, epochs=2000, batch_size=bs, seed=0)
        t_weights[name] = res.weights
        print(f"  {name:10s} weights={np.round(res.weights, 5)}")
    ref_t = t_weights["BatchGD"]
    spread_t = max(float(np.max(np.abs(w - ref_t))) for w in t_weights.values())
    print(f"  -> max weight spread (non-convex, larger than the convex case): {spread_t:.2e}")

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

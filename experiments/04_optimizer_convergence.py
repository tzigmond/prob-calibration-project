"""Experiment 04 - optimizer convergence sanity check.

Runs all four from-scratch optimizers against the Gaussian loss on BTC data and
confirms they converge to consistent coefficients (a convex problem, so they
should agree). Also fits the Student-t loss to illustrate the caveat: its NLL is
non-convex in the weights, so different optimizers can settle at slightly
different points from the same OLS start.

This is a sanity check, not a second experiment - it confirms the headline result
isn't an artifact of choosing Adam. The convergence *trajectory* is plotted on a
synthetic well-conditioned problem, because daily returns are too weakly
predictable for the loss to show a meaningful descent.
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
from src.plotstyle import use_house_style

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

    print("Gaussian loss (convex) - optimizers should converge to the same weights:")
    # Initialize AWAY from the optimum (default init is OLS, i.e. already minimal)
    # so the convergence trace shows an actual descent to compare.
    w0 = np.zeros(Xtr.shape[1])
    histories = {}
    gauss_weights = {}
    for name, opt in optimizers.items():
        bs = None if name == "BatchGD" else 256
        res = fit(Xtr, ytr, GaussianLoss(), opt, epochs=2000, batch_size=bs, w_init=w0, seed=0)
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
    print("\nStudent-t loss (non-convex) - gentler rates, shared OLS init:")
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

    # Convergence-trajectory plot on a WELL-CONDITIONED SYNTHETIC problem.
    # Daily returns carry almost no predictable signal, so the Gaussian loss is
    # nearly flat in the weights (predicting zero is almost as good as the
    # optimum) - a descent curve is only meaningful where the optimum sits far
    # from the origin. This isolates optimizer *speed*, which the BTC checks
    # above (agreement at the optimum) cannot show.
    print("\nConvergence trajectory plotted on a synthetic well-conditioned problem\n"
          "(BTC returns are too weakly predictable to show a meaningful descent).")
    rng = np.random.default_rng(0)
    ns, ds = 4000, 5
    Xs = np.column_stack([np.ones(ns), rng.standard_normal((ns, ds))])
    ws = rng.standard_normal(ds + 1)
    ys = Xs @ ws + rng.normal(0, 0.5, ns)
    w0s = np.zeros(ds + 1)
    demo_opts = {
        "BatchGD":  O.BatchGD(lr=0.05),
        "SGD":      O.SGD(lr=0.02),
        "Momentum": O.Momentum(lr=0.02),
        "Adam":     O.Adam(lr=0.05),
    }
    demo_hist = {}
    for name, opt in demo_opts.items():
        bs = None if name == "BatchGD" else 256
        demo_hist[name] = fit(Xs, ys, GaussianLoss(), opt, epochs=200,
                              batch_size=bs, w_init=w0s, seed=0).loss_history

    import matplotlib
    matplotlib.use("Agg")
    use_house_style()
    import matplotlib.pyplot as plt

    colors = {"BatchGD": "#4C72B0", "SGD": "#DD8452", "Momentum": "#55A868", "Adam": "#C44E52"}
    FIGURES.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots()
    for name, hist in demo_hist.items():
        ax.plot(hist, label=name, color=colors.get(name), lw=2.0)
    ax.set_yscale("log")
    ax.set_xlabel("epoch")
    ax.set_ylabel("Gaussian loss (MSE)")
    ax.set_title("Optimizer convergence - synthetic well-conditioned problem")
    ax.legend(title="optimizer")
    out = FIGURES / "optimizer_convergence.png"
    fig.savefig(out)
    plt.close(fig)
    print(f"Convergence plot written to {out}")


if __name__ == "__main__":
    main()

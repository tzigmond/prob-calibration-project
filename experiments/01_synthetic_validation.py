"""Experiment 01 — synthetic validation (the gate before real data).

Generate data from known Gaussian, Laplace, and Student-t(ν) distributions with
known true coefficients, then confirm:
  * every optimizer recovers the true weights for the (convex) Gaussian loss,
  * the Laplace and Student-t losses recover their weights under Adam,
  * nu estimated separately from OLS residuals matches the true nu.

If this fails, the math is wrong — nothing downstream is trustworthy.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
try:
    sys.stdout.reconfigure(encoding="utf-8")  # avoid cp1252 console encode errors
except Exception:
    pass

from src import optimizers as O
from src.losses import GaussianLoss, LaplaceLoss, StudentTLoss, estimate_nu
from src.train import fit

TOL = 0.05  # max abs coefficient error we accept as "recovered"


def make_design(rng, n=4000, d=3):
    X = np.column_stack([np.ones(n), rng.standard_normal((n, d))])
    w_true = np.array([0.5, -1.2, 2.0, 0.7])
    return X, w_true


def check(name, w_true, w_hat):
    err = float(np.max(np.abs(w_hat - w_true)))
    status = "PASS" if err < TOL else "FAIL"
    print(f"  [{status}] {name:22s} max|err|={err:.4f}  weights={np.round(w_hat, 3)}")
    return err < TOL


def main():
    rng = np.random.default_rng(0)
    X, w_true = make_design(rng)
    n = len(X)
    ok = True

    print("Gaussian errors — every optimizer should recover w_true:")
    y = X @ w_true + rng.normal(0, 0.5, n)
    for opt in [O.BatchGD(lr=0.1), O.SGD(lr=0.05), O.Momentum(lr=0.05), O.Adam(lr=0.05)]:
        bs = None if isinstance(opt, O.BatchGD) else 256
        w = fit(X, y, GaussianLoss(), opt, epochs=500, batch_size=bs, seed=0).weights
        ok &= check(type(opt).__name__, w_true, w)

    print("\nLaplace errors — Adam under MAE:")
    y = X @ w_true + rng.laplace(0, 0.4, n)
    w = fit(X, y, LaplaceLoss(), O.Adam(lr=0.02), epochs=1000, seed=0).weights
    ok &= check("Adam", w_true, w)

    print("\nStudent-t errors — estimate nu separately, hold fixed, recover weights:")
    nu_true = 4.0
    y = X @ w_true + stats.t.rvs(df=nu_true, scale=0.4, size=n, random_state=1)
    w_ols = np.linalg.lstsq(X, y, rcond=None)[0]
    nu_hat = estimate_nu(y - X @ w_ols)
    tloss = StudentTLoss(nu=nu_hat)
    tloss.scale = tloss.estimate_scale(y - X @ w_ols)
    w = fit(X, y, tloss, O.Adam(lr=0.02), epochs=1000, seed=0).weights
    ok &= check("Adam", w_true, w)
    nu_ok = abs(nu_hat - nu_true) < 1.5
    print(f"  [{'PASS' if nu_ok else 'FAIL'}] nu recovery          "
          f"nu_true={nu_true}  nu_hat={nu_hat:.3f}  scale_hat={tloss.scale:.3f}")
    ok &= nu_ok

    print("\n" + ("ALL CHECKS PASSED" if ok else "SOME CHECKS FAILED"))
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()

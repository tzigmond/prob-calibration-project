# From Likelihood to Loss

Gradient descent under competing probabilistic assumptions.

A model's loss function is a consequence of an assumed error distribution:
**Gaussian ⇒ MSE**, **Laplace ⇒ MAE**, **Student-t ⇒ a heavy-tailed NLL**.
This project builds all three from scratch, optimizes them with hand-written
gradient descent, and empirically tests whether getting the distributional
assumption right or wrong changes the **calibration** of predictive confidence
intervals on real financial data.

## Core question

Does the error-distribution choice actually change interval calibration — and
does any apparent Student-t advantage survive once we account for volatility
clustering (a Gaussian scale mixture is heavy-tailed by construction) and a
non-parametric empirical-quantile baseline?

## Model roster (5)

1. **Gaussian, fixed σ** — the MSE baseline.
2. **Gaussian, EWMA-scaled** — a minimal heteroskedasticity control (RiskMetrics
   λ≈0.94). Tests whether the fat tail is just volatility clustering in disguise.
3. **Laplace** — MAE / double-exponential errors.
4. **Student-t** — heavy-tailed NLL; ν estimated separately and held fixed.
5. **Empirical** — historical-simulation intervals from residual quantiles. The
   "does the parametric assumption even matter?" baseline.

## Success criterion

A clean coverage/width table at 90/95/99% nominal levels, **with a binomial
confidence interval and a Kupiec test on every coverage number**. Succeeds
scientifically even if Student-t loses its edge to vol-scaling — as long as that
is explained.

## Layout

- `src/` — the library (optimizers, losses, intervals, calibration, diagnostics, data)
- `experiments/` — runnable scripts producing the tables and figures
- `notebooks/` — exploratory work
- `results/` — generated figures and tables
- `report/` — the final writeup

## Setup (Windows-native)

```powershell
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

## Explicit non-goals

- Not claiming tradeable alpha — this is a calibration/statistics study.
- Not full time-varying variance (no GARCH); model 2 is a single deliberate
  EWMA step. GARCH is a noted future extension.
- Not claiming joint ν estimation is impossible — a harder, separate extension.

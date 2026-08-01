# experiments/ - runnable scripts

Each script is an entry point that wires together `src/` modules, runs one piece
of the study, and writes figures to `results/figures/` and tables to
`results/tables/`. Run from the project root with the venv active.

## 01_synthetic_validation.py
The math checkpoint, run before any real data. Generates synthetic data from
known Gaussian, Laplace, and Student-t(ν) distributions with ground-truth
parameters, then confirms:
- every optimizer recovers the true coefficients,
- separately-estimated ν ≈ true ν,
- the EWMA-scaled and empirical-quantile interval builders hit their nominal
  coverage on data with known properties.

## 02_btc_primary.py
The core result. Pulls BTC-USD, builds features, fits all five models on the
chronological 80% train split, and produces the centerpiece **coverage / width
table** (5 models × 90/95/99%, with binomial CIs and Kupiec p-values) plus the
residual diagnostics. This is the load-bearing experiment.

## 03_aapl_robustness.py
Generalization check: the full Week-2 pipeline re-run on AAPL-on-SPY (market-model
regression, residual = idiosyncratic return). Does the BTC conclusion hold on a
different, more canonical dataset?

## 04_optimizer_convergence.py
Sanity check, not a second experiment (≤2 pages in the writeup). Confirms Batch
GD / SGD / Momentum / Adam agree for the convex losses (Gaussian, Laplace) and
documents the *expected* divergence / local minima for the non-convex Student-t
NLL from a shared OLS initialization.

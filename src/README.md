# src/ — the library

Reusable, tested building blocks. No script here pulls data or prints a table on
import; the `experiments/` scripts orchestrate these modules.

## optimizers.py
Gradient-descent optimizers, all from scratch in raw NumPy:
**Batch GD, SGD, Momentum, Adam**. Each takes a loss+gradient callable and
parameters, returns fitted weights plus a convergence trace. All runs initialize
from the OLS solution so the (non-convex) Student-t fits are comparable across
optimizers. Unit-tested against scikit-learn's closed-form / SGD solutions on the
Gaussian case as a correctness check.

## losses.py
Negative log-likelihoods and their gradients w.r.t. regression weights:
- **Gaussian** → MSE.
- **Laplace** → MAE (subgradient at zero).
- **Student-t** → heavy-tailed NLL; gradient w.r.t. weights only, with ν held
  fixed (ν comes from `data.py` / scipy, not optimized here).

## train.py
The generic training loop — the one place that couples a loss to an optimizer,
so `losses.py` and `optimizers.py` can stay ignorant of each other. `fit()`
repeatedly asks `loss.gradient()` for gradients, hands them to `optimizer.step()`,
records the loss trajectory, and returns fitted weights + convergence history.
Initializes from the OLS solution so the non-convex Student-t fits are comparable
across optimizers.

## intervals.py
Builds predictive intervals from a fitted model, mapping
`(prediction, scale, params) → (lo, hi)` for each of the five models:
- z-quantile (Gaussian fixed-σ)
- EWMA-scaled Gaussian (trailing-volatility standardization, then scale back up)
- Laplace quantile
- Student-t quantile
- empirical residual quantiles (historical simulation)

## calibration.py
The load-bearing evaluation module:
- chronological 80/20 split (no lookahead)
- empirical **coverage** and average **interval width** at 90/95/99%
- a **binomial confidence interval** on each coverage estimate
- the **Kupiec** unconditional-coverage test (p-value per model/level)
- assembles the centerpiece coverage/width table

## diagnostics.py
Distributional goodness-of-fit checks: standardized-residual histograms with
fitted densities overlaid, Q-Q plots, and tail-probability comparisons
P(|z|>2), P(|z|>3), P(|z|>4). Confirms the assumption fits, not just that
intervals happened to cover.

## data.py
Data acquisition and feature construction:
- pull BTC-USD (and AAPL/SPY) via yfinance
- **sanitize** ticks (drop non-positive prices, dedupe dates)
- daily log returns
- AR(3) lagged-return + lagged-|r| feature matrix
- separate ν estimation via `scipy.stats.t.fit` on OLS residuals (keep df only;
  re-estimate scale from own residuals)

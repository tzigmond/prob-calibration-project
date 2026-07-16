# From Likelihood to Loss

**Gradient descent under competing probabilistic assumptions — and whether the
assumption you pick actually changes the calibration of your predictions.**

Every regression loss is a maximum-likelihood estimator in disguise. Minimizing
mean squared error is exactly maximum likelihood under Gaussian errors;
minimizing mean absolute error is maximum likelihood under Laplace errors; a
heavy-tailed Student-*t* likelihood yields a robust, redescending loss that
belongs to neither. This project makes that correspondence concrete — deriving
each loss from its error distribution, implementing the losses and their
gradient-descent optimizers from scratch, and then asking an empirical question
the derivation alone can't answer:

> **On real financial return data, does choosing the "right" error distribution
> produce better-calibrated prediction intervals — or does an apparent advantage
> just reflect volatility clustering that a simpler model captures too?**

---

## The core correspondence

For a linear predictor $\hat{y} = X w$ with i.i.d. errors $\varepsilon = y - \hat{y}$,
the negative log-likelihood is the loss. Three choices of error density give
three familiar losses:

| Error distribution | Density (kernel) | Negative log-likelihood ⇒ loss | Gradient character |
|---|---|---|---|
| **Gaussian** $\mathcal{N}(0,\sigma^2)$ | $\exp(-\varepsilon^2 / 2\sigma^2)$ | $\sum \varepsilon^2$ → **MSE** | linear in residual |
| **Laplace** $\text{Lap}(0,b)$ | $\exp(-\lvert\varepsilon\rvert / b)$ | $\sum \lvert\varepsilon\rvert$ → **MAE** | sign of residual (subgradient at 0) |
| **Student-*t*** $t_\nu(0,\sigma)$ | $\left(1 + \frac{\varepsilon^2}{\nu\sigma^2}\right)^{-(\nu+1)/2}$ | heavy-tailed NLL | $\propto \dfrac{(\nu+1)\,\varepsilon}{\nu\sigma^2 + \varepsilon^2}$ (redescending) |

The Gaussian penalizes large residuals quadratically, so a few outliers dominate
the fit. The Laplace penalizes linearly, giving the median-like robustness of
MAE. The Student-*t* goes further — its influence function *redescends*, so
extreme residuals are actively down-weighted. That robustness is also why the
Student-*t* NLL is **non-convex in the weights**, a fact this project confronts
rather than hides (see *Optimizer convergence*).

Once a distribution defines the loss, it also defines the **prediction
interval**: the same likelihood that gives you MSE gives you $\hat{y} \pm z_{1-\alpha/2}\,\sigma$.
Getting the distribution wrong therefore mis-sizes your intervals — and interval
calibration is exactly what this project measures.

---

## Research question and hypothesis

Bitcoin daily log returns are famously heavy-tailed (excess kurtosis ≈ 12). A
fixed-variance Gaussian model will therefore produce intervals that are too
narrow in the tails and under-cover at high confidence levels.

**Falsifiable hypothesis.** A Student-*t* error model will produce
better-calibrated intervals than a fixed-variance Gaussian on BTC returns.

**The real test.** A Gaussian *scale mixture* — a Gaussian whose variance
changes over time — is itself heavy-tailed. Bitcoin has exactly that structure
(volatility clustering). So the interesting question isn't whether Student-*t*
beats a fixed Gaussian (it almost has to), but whether its advantage **survives**
once volatility clustering is accounted for, and whether it beats a
**non-parametric** baseline that assumes no distribution at all. If the fat tail
is really just time-varying variance in disguise, a volatility-scaled Gaussian
should erase most of the Student-*t*'s edge. That is the result this project is
built to detect — and it succeeds scientifically whether the hypothesis holds or
fails, provided the outcome is explained.

---

## The five models

| # | Model | Interval basis | What it isolates |
|---|---|---|---|
| 1 | **Gaussian, fixed σ** | $z$-quantile × σ | The naive MSE baseline |
| 2 | **Gaussian, EWMA-scaled σ_t** | $z$-quantile × σ_t | Volatility clustering (RiskMetrics EWMA, λ = 0.94) — the heteroskedasticity control |
| 3 | **Laplace** | Laplace quantile × b | Linear-penalty / median-regression errors |
| 4 | **Student-*t*** | $t_\nu$ quantile × σ | Genuinely heavy per-observation tails |
| 5 | **Empirical** | quantiles of training residuals | A distribution-free historical-simulation baseline |

Models 2 and 5 are the ones that make the study honest: without a
volatility-scaled Gaussian and a non-parametric baseline, "Student-*t* wins" is
nearly a foregone conclusion rather than a finding.

---

## Methodology

- **Data.** BTC-USD daily log returns via `yfinance` (thousands of observations,
  no weekend gaps). Pulls are cached and sanitized (non-positive prices and
  duplicate dates dropped) so one bad tick can't corrupt a log return.
- **Features.** A deliberately minimal AR(3) design matrix — lagged returns
  $r_{t-1}, r_{t-2}, r_{t-3}$ plus $\lvert r_{t-1}\rvert$ as a volatility-clustering
  proxy. Daily returns are near-unpredictable, so the mean model is intentionally
  weak: the story is the error distribution, not the predictors.
- **Estimation.** ν for the Student-*t* is estimated **once** on OLS residuals
  (`scipy.stats.t.fit`, keeping degrees of freedom) and then **held fixed** during
  regression. Joint estimation of ν and the weights is non-convex, involves
  digamma/gamma gradients, and is poorly identified in finite samples — a
  deliberate scope decision, with joint estimation noted as a future extension.
- **Split.** Chronological 80/20 — never random — so no test-period information
  leaks backward into training.
- **Optimizers.** Adam for the headline results; Batch GD, SGD, and Momentum
  serve as a convergence sanity check (all four implemented from scratch, all
  initialized from the OLS solution so the non-convex Student-*t* fits are
  comparable).

## How calibration is measured

Coverage alone is gameable — a wide-enough interval always covers — so it is
never reported alone.

- **Empirical coverage** at 90 / 95 / 99% nominal levels.
- **Average interval width** — the sharpness check that makes coverage meaningful.
- **Binomial (Wilson) confidence intervals** on every coverage estimate. With a
  ~700-point test set, a 99% interval expects only ~7 violations, so tail
  differences are noisy and must be reported with uncertainty, not as point
  values.
- **Kupiec proportion-of-failures test** — a likelihood-ratio test turning
  "these coverages look different" into a p-value.

## Diagnostics

Calibration can be right for the wrong reasons, so the distributional assumption
is checked directly: standardized-residual histograms with fitted densities
overlaid, Q-Q plots, and empirical-vs-modeled tail probabilities
$P(\lvert z\rvert > 2, 3, 4)$. Well-calibrated intervals are necessary but not
sufficient evidence that a distribution fits — the diagnostics close that gap.

---

## Repository layout

```
prob-calibration-project/
├── src/                  # Importable library — no script logic
│   ├── data.py           # yfinance pull (cached/sanitized), log returns, AR features, chrono split
│   ├── optimizers.py     # Batch GD, SGD, Momentum, Adam — shared .step(params, grads)
│   ├── losses.py         # Gaussian/Laplace/Student-t NLL, gradients, scale + ν estimation
│   ├── train.py          # Generic fit loop — the one place a loss meets an optimizer
│   ├── intervals.py      # Five interval builders (incl. EWMA-scaled and empirical)
│   ├── calibration.py    # Coverage, width, binomial CI, Kupiec test, centerpiece table
│   └── diagnostics.py    # Residual histograms, Q-Q plots, tail probabilities
├── experiments/          # Runnable orchestration scripts, each reads top-to-bottom as a narrative
│   ├── 01_synthetic_validation.py   # Recover known coefficients — the gate before real data
│   ├── 02_btc_primary.py            # The core result: coverage/width table + diagnostics
│   ├── 03_aapl_robustness.py        # Generalization check on AAPL-on-SPY
│   └── 04_optimizer_convergence.py  # Sanity check across all four optimizers
├── notebooks/            # Exploration only
├── results/              # Generated tables and figures (build output)
└── report/               # Final writeup
```

**Design principle.** `src/` is a library; `experiments/` is the application.
Nothing in `src/` imports from `experiments/`, and the `src/` modules stay
decoupled from each other — optimizers and losses only ever exchange gradient
vectors, and `train.fit()` is the sole place that couples the two. This is what
lets any optimizer run against any loss, and lets each module be unit-tested in
isolation.

---

## Setup

Windows-native — no WSL or Linux toolchain required; every dependency ships a
prebuilt wheel.

```powershell
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

Run an experiment from the project root, e.g.:

```powershell
python experiments/02_btc_primary.py
```

Tables land in `results/tables/`, figures in `results/figures/`.

---

## Scope and non-goals

Stated up front so they read as deliberate, not missed:

- **Not a trading strategy.** This is a calibration/statistics study, not an
  alpha signal.
- **Variance model held fixed** for four of the five models. Model 2's single
  EWMA step is the one deliberate move toward time-varying variance; a full
  GARCH treatment is the natural next extension, not part of this scope.
- **ν estimated separately**, not jointly with the weights — a scope and
  identifiability decision, not an oversight.

## Roadmap

- [ ] `src/` library implementations (stubs in place)
- [ ] Synthetic-recovery validation (experiment 01)
- [ ] BTC primary result — coverage/width table + diagnostics (experiment 02)
- [ ] AAPL-on-SPY robustness check (experiment 03)
- [ ] Optimizer-convergence sanity check (experiment 04)
- [ ] Final writeup

# From Likelihood to Loss

**Do competing error-distribution assumptions change the calibration of
prediction intervals on financial returns — and does a heavy-tailed model earn
its keep once volatility clustering is accounted for?**

---

## 1. Derivation: likelihood → loss

For a linear predictor $\hat{y} = Xw$ with i.i.d. errors $\varepsilon = y - \hat{y}$,
maximum likelihood minimizes the negative log-likelihood, which *is* the loss.

**Gaussian.** $p(\varepsilon) \propto \exp(-\varepsilon^2/2\sigma^2)$ gives
$-\log p = \tfrac{1}{2\sigma^2}\varepsilon^2 + \text{const}$. Summed over the
sample this is **MSE**, with gradient $\frac{2}{n}X^\top(Xw-y)$ — linear in the
residual, so large errors dominate quadratically.

**Laplace.** $p(\varepsilon) \propto \exp(-|\varepsilon|/b)$ gives
$-\log p = \tfrac{1}{b}|\varepsilon| + \text{const}$ — **MAE**. The gradient is
$\frac{1}{n}X^\top \operatorname{sign}(Xw-y)$; non-differentiable at zero, where
we take the subgradient $\operatorname{sign}(0)=0$. Linear penalty ⇒ median-like
robustness.

**Student-*t*.** $p(\varepsilon) \propto \left(1+\frac{\varepsilon^2}{\nu\sigma^2}\right)^{-(\nu+1)/2}$
gives a heavy-tailed NLL whose gradient carries the per-residual weight
$\frac{(\nu+1)\varepsilon}{\nu\sigma^2+\varepsilon^2}$. This weight *redescends* —
it rises then falls toward zero as $|\varepsilon|\to\infty$ — so extreme residuals
are actively discounted. That robustness is also why the Student-*t* NLL is
**non-convex in the weights**. As $\nu\to\infty$ the *t* converges to the
Gaussian and the loss back to MSE.

Each distribution also defines its **interval**: the same likelihood that yields
MSE yields $\hat{y}\pm z_{1-\alpha/2}\,\sigma$. Getting the distribution wrong
therefore mis-sizes the interval — which is what we measure.

---

## 2. Synthetic validation

Before touching real data, we generated data from each distribution with known
coefficients $w^\star=[0.5,-1.2,2.0,0.7]$ and confirmed recovery
(`experiments/01`):

| Error law | Optimizer(s) | max │ŵ − w★│ | ν recovery |
|---|---|---|---|
| Gaussian | BatchGD / SGD / Momentum / Adam | ≤ 0.048 | — |
| Laplace | Adam | 0.006 | — |
| Student-*t* | Adam (ν fixed) | 0.013 | ν̂ = 4.52 vs ν = 4.0 |

All optimizers recover the Gaussian coefficients; the separately-estimated ν
matches truth. The math and the machinery are sound.

---

## 3. Primary result: BTC-USD daily log returns

Data: BTC-USD 2015–2025, daily log returns, AR(3)+|r| features, chronological
80/20 split (train 2919, test 730). Estimated **ν = 2.13**, residual **excess
kurtosis = 11.1** — the documented heavy tail is present.

**Coverage (nominal) / average width, held-out test set:**

| Model | 90% cov | 95% cov | 99% cov | width @95% | Kupiec @90% |
|---|---|---|---|---|---|
| Gaussian (fixed σ) | 0.968 | 0.984 | 0.999 | 0.152 | reject |
| **EWMA-Gaussian** | **0.885** | 0.919 | 0.958 | **0.096** | **p = 0.18 (ok)** |
| Laplace | 0.952 | 0.981 | 1.000 | 0.150 | reject |
| Student-*t* | 0.952 | 0.988 | 1.000 | 0.162 | reject |
| Empirical | 0.962 | 0.986 | 1.000 | 0.164 | reject |

Two findings stand out:

1. **The fixed-variance models over-cover and are wide.** The chronological test
   window (2023–2024) is calmer than the training era (2017–2021), so a single
   fixed σ estimated on the training period produces intervals that are too fat
   for the test period. This is the opposite of the naive "heavy tails ⇒
   under-coverage" intuition, and it is a direct consequence of holding the
   variance model fixed across a volatility regime shift.

2. **The volatility-scaled Gaussian dominates.** EWMA-Gaussian is the only model
   whose 90% coverage is not rejected by the Kupiec test, and its intervals are
   ~40% narrower than the fixed Gaussian's at every level. Adapting the *scale*
   to local volatility matters far more than adapting the *shape* of the error
   law.

**Tail diagnostics** confirm the mechanism. Standardized-residual tail masses:

| | empirical | Normal | Laplace | *t*(ν=2.1) |
|---|---|---|---|---|
| P(│z│>2) | 0.057 | 0.046 | 0.059 | 0.012 |
| P(│z│>3) | 0.016 | 0.003 | 0.014 | 0.005 |
| P(│z│>4) | 0.006 | 0.000 | 0.004 | 0.003 |

The *unconditional* residual distribution looks like a *t* with ν ≈ 2 (variance
barely finite). But a *t*(2) fitted to the marginal is exactly what a
**Gaussian scale mixture** — a Gaussian whose variance changes over time —
produces. The fact that a single EWMA volatility step recovers calibration and
sharpness is strong evidence that BTC's fat marginal tail is **substantially
volatility clustering, not a genuine per-observation heavy tail**.

---

## 4. Robustness: AAPL-on-SPY

Market-model regression of AAPL returns on SPY, 2010–2025 (train 3018, test 755).
Estimated **ν = 3.62**, excess kurtosis **7.79**.

| Model | 90% cov (Kupiec) | 95% cov (Kupiec) | width @95% |
|---|---|---|---|
| Gaussian (fixed σ) | 0.956 (reject) | 0.970 (reject) | 0.052 |
| **EWMA-Gaussian** | **0.905 (p=0.67)** | **0.938 (p=0.14)** | **0.041** |
| Laplace | 0.954 (reject) | 0.977 (reject) | 0.056 |
| Student-*t* | 0.936 (reject) | 0.970 (reject) | 0.052 |
| Empirical | 0.942 (reject) | 0.968 (reject) | 0.052 |

The conclusion generalizes: **EWMA-Gaussian is again the best-calibrated and
sharpest model**, non-rejected at both 90% and 95%. Here ν = 3.6 is less extreme
and the *t*(3.6) marginal tail masses match the empirical ones closely — yet
volatility scaling still wins on calibration, because the value is in the
*conditional* variance, not the marginal shape.

---

## 5. Optimizer convergence (sanity check)

On the convex Gaussian loss, all four from-scratch optimizers converge to the
same weights (max spread 9.7 × 10⁻³) — the headline result is not an Adam
artifact. On the Student-*t* loss the spread balloons to 1.6: from the shared OLS
initialization the optimizers settle at different points, and Momentum in
particular wanders. This is the **expected** signature of the non-convex,
redescending *t* NLL, not a bug — and a reason to treat Student-*t* point
estimates with more caution than the interval story requires.

---

## 6. Conclusion

The falsifiable hypothesis — that a Student-*t* error model yields
better-calibrated intervals than a fixed-variance Gaussian — is only *weakly*
supported, and for the wrong reason. The decisive factor on both datasets is not
the *shape* of the error distribution but whether the model adapts its *scale* to
local volatility. A one-parameter EWMA volatility step beats every fixed-variance
model — Gaussian, Laplace, Student-*t*, and non-parametric alike — on both
calibration and sharpness. Much of what looks like a heavy per-observation tail
in crypto and equity returns is volatility clustering in disguise.

---

## 7. Limitations and extensions

- **Regime shift under the chronological split.** Holding σ fixed across a
  train→test volatility change is what makes the fixed-variance models
  over-cover. This is a faithful consequence of the fixed-variance scope, not an
  error — but it means the fixed-Gaussian baseline is unflattering for a reason
  worth stating plainly.
- **ν estimated on unconditional residuals.** For BTC this yields ν ≈ 2.1, where
  the variance is barely finite, precisely because the unconditional residuals
  conflate tail weight with volatility clustering. Estimating ν *conditionally*
  (after volatility scaling) would likely give a larger, better-identified ν.
- **Fixed variance for four of five models.** Only the EWMA-Gaussian adapts its
  scale. A full **GARCH** conditional-variance model — combined with a
  conditional Student-*t* — is the natural next step and would properly separate
  volatility dynamics from residual tail shape.
- **ν held fixed, estimated separately.** Joint estimation of ν and the weights
  is non-convex and poorly identified in finite samples; a deliberate scope
  choice, noted as an extension rather than a limitation of the finding.
- **Not a trading strategy.** This is a calibration study; nothing here claims
  predictive edge on the *direction* of returns.

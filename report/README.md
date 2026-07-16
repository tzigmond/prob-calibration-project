# report/ — the writeup

The final narrative deliverable that ties the derivation, experiments, and
diagnostics together.

## final_report.md
Structured as:
1. **Derivation** — likelihood → loss for Gaussian (MSE), Laplace (MAE), and
   Student-t (NLL).
2. **Synthetic validation** — proof the math and optimizers are correct.
3. **BTC coverage/width table + residual diagnostics** — the centerpiece, all
   five models, with coverage CIs and Kupiec tests.
4. **AAPL robustness check** — does the conclusion generalize?
5. **Optimizer convergence sanity check** — brief.
6. **Limitations** — fixed vs EWMA variance model, separately-estimated ν, why
   the chronological split, and what would change under a GARCH-style extension.

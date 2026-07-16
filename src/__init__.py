"""prob-calibration-project library.

Reusable, importable building blocks. Nothing here imports from `experiments/`.
The two designed-for-sharing interfaces are:
  - optimizers expose `.step(params, grads) -> params`
  - losses expose `.loss()`, `.gradient()`, `.estimate_scale()`
`train.fit()` is the only place that couples a loss to an optimizer.
"""

"""The generic training loop — the seam where a loss meets an optimizer.

This is the ONLY module that knows about both interfaces at once, which is what
lets ``losses.py`` and ``optimizers.py`` stay ignorant of each other. Every
experiment calls ``fit`` rather than re-writing the epoch loop inline.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .losses import Loss
from .optimizers import Optimizer


@dataclass
class FitResult:
    weights: np.ndarray
    loss_history: list[float]


def _ols_init(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Least-squares solution — the shared starting point for all optimizers so
    the non-convex Student-t fits are comparable."""
    w, *_ = np.linalg.lstsq(X, y, rcond=None)
    return w


def fit(
    X: np.ndarray,
    y: np.ndarray,
    loss: Loss,
    optimizer: Optimizer,
    epochs: int = 1000,
    batch_size: int | None = None,
    w_init: np.ndarray | None = None,
    tol: float | None = None,
) -> FitResult:
    """Couple ``loss`` to ``optimizer`` and run gradient descent.

    Loop: each epoch (and mini-batch, if ``batch_size`` is set), ask
    ``loss.gradient`` for gradients, hand them to ``optimizer.step``, record
    ``loss.loss``. Returns fitted weights + the loss trajectory.

    ``w_init`` defaults to the OLS solution so non-convex Student-t fits are
    comparable across optimizers. ``tol`` optionally early-stops on the absolute
    change in full-dataset loss between epochs.
    """
    w = _ols_init(X, y) if w_init is None else np.array(w_init, dtype=float)
    optimizer.reset()

    n = len(y)
    loss_history: list[float] = []
    prev_loss: float | None = None

    for _ in range(epochs):
        if batch_size is None:
            grads = loss.gradient(X, y, w)
            w = optimizer.step(w, grads)
        else:
            order = np.random.permutation(n)
            for start in range(0, n, batch_size):
                idx = order[start:start + batch_size]
                grads = loss.gradient(X[idx], y[idx], w)
                w = optimizer.step(w, grads)

        current = loss.loss(X, y, w)
        loss_history.append(current)

        if tol is not None and prev_loss is not None and abs(prev_loss - current) < tol:
            break
        prev_loss = current

    return FitResult(weights=w, loss_history=loss_history)

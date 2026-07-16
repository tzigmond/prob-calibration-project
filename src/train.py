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

    Loop: for each epoch (and mini-batch, if ``batch_size`` is set), ask
    ``loss.gradient`` for gradients, hand them to ``optimizer.step``, record
    ``loss.loss``. Returns fitted weights + the loss trajectory.

    ``w_init`` defaults to the OLS solution upstream so that the non-convex
    Student-t fits are comparable across optimizers (shared starting point).
    ``tol`` optionally enables early stopping on loss-change.
    """
    raise NotImplementedError

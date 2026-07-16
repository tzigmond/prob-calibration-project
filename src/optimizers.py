"""From-scratch gradient-descent optimizers.

All four expose the same interface — ``step(params, grads) -> params`` — so the
training loop and experiments can swap them freely. Optimizers know NOTHING about
the loss; they only ever exchange parameter and gradient vectors. This is what
keeps the 4-optimizers x 3-losses grid clean.

Stateful optimizers (Momentum, Adam) carry accumulators between steps, which is
why these are classes, not functions. Call ``reset()`` before reusing an
instance for a fresh fit.
"""
from __future__ import annotations

import numpy as np


class Optimizer:
    """Common interface. Subclasses implement ``step``."""

    def step(self, params: np.ndarray, grads: np.ndarray) -> np.ndarray:
        """Return updated params given current params and their gradient."""
        raise NotImplementedError

    def reset(self) -> None:
        """Clear internal state (velocity, moments, timestep) for a fresh run."""
        # Stateless optimizers override with a no-op; stateful ones clear here.
        pass


class BatchGD(Optimizer):
    """Full-dataset gradient step, fixed learning rate. Stateless."""

    def __init__(self, lr: float = 1e-2):
        self.lr = lr

    def step(self, params, grads):
        # params <- params - lr * grads
        raise NotImplementedError


class SGD(Optimizer):
    """Stochastic / mini-batch step. The stochasticity comes from the batching
    in ``train.fit``; the update rule here is identical to BatchGD. Stateless."""

    def __init__(self, lr: float = 1e-2):
        self.lr = lr

    def step(self, params, grads):
        # params <- params - lr * grads   (grads computed on a mini-batch upstream)
        raise NotImplementedError


class Momentum(Optimizer):
    """SGD with a velocity term. Stateful: carries a running velocity vector."""

    def __init__(self, lr: float = 1e-2, beta: float = 0.9):
        self.lr = lr
        self.beta = beta
        self.velocity: np.ndarray | None = None  # lazily shaped to params

    def step(self, params, grads):
        # v <- beta * v + grads ; params <- params - lr * v
        raise NotImplementedError

    def reset(self):
        self.velocity = None


class Adam(Optimizer):
    """Adam: first-moment ``m``, second-moment ``v``, timestep ``t``, with bias
    correction. Stateful."""

    def __init__(self, lr: float = 1e-3, beta1: float = 0.9,
                 beta2: float = 0.999, eps: float = 1e-8):
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.m: np.ndarray | None = None
        self.v: np.ndarray | None = None
        self.t: int = 0

    def step(self, params, grads):
        # t += 1
        # m <- beta1*m + (1-beta1)*grads
        # v <- beta2*v + (1-beta2)*grads**2
        # m_hat = m/(1-beta1**t) ; v_hat = v/(1-beta2**t)
        # params <- params - lr * m_hat / (sqrt(v_hat) + eps)
        raise NotImplementedError

    def reset(self):
        self.m = None
        self.v = None
        self.t = 0

"""The training loop is reproducible when seeded, and honors w_init."""
import numpy as np

from src import optimizers as O
from src.losses import GaussianLoss
from src.train import fit


def _problem():
    rng = np.random.default_rng(1)
    n = 1500
    X = np.column_stack([np.ones(n), rng.standard_normal((n, 2))])
    y = X @ np.array([1.0, -0.5, 0.3]) + rng.normal(0, 0.4, n)
    return X, y


def test_seeded_minibatch_runs_are_identical():
    X, y = _problem()
    a = fit(X, y, GaussianLoss(), O.SGD(lr=0.05), epochs=100, batch_size=128, seed=42).weights
    b = fit(X, y, GaussianLoss(), O.SGD(lr=0.05), epochs=100, batch_size=128, seed=42).weights
    assert np.array_equal(a, b)


def test_different_seeds_differ():
    X, y = _problem()
    a = fit(X, y, GaussianLoss(), O.SGD(lr=0.05), epochs=50, batch_size=128, seed=1).weights
    b = fit(X, y, GaussianLoss(), O.SGD(lr=0.05), epochs=50, batch_size=128, seed=2).weights
    assert not np.array_equal(a, b)


def test_w_init_respected():
    X, y = _problem()
    # Zero epochs: fit should return exactly the provided init.
    w0 = np.array([9.0, 9.0, 9.0])
    res = fit(X, y, GaussianLoss(), O.BatchGD(lr=0.1), epochs=0, w_init=w0)
    assert np.array_equal(res.weights, w0)

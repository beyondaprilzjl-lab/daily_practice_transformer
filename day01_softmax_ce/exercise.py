"""Day 01 exercise: stable softmax, log-softmax, and cross entropy."""

from __future__ import annotations

import numpy as np


def stable_softmax(logits: np.ndarray, axis: int = -1) -> np.ndarray:
    """Compute a numerically stable softmax along ``axis``."""
    # TODO:
    # shifted = ...
    # exp_x = ...
    # return ...
    raise NotImplementedError("Implement stable_softmax")


def stable_log_softmax(logits: np.ndarray, axis: int = -1) -> np.ndarray:
    """Compute a numerically stable log-softmax along ``axis``.

    Do not implement this as ``np.log(stable_softmax(...))``.
    """
    # TODO:
    # shifted = ...
    # return shifted - log(sum(exp(shifted)))
    raise NotImplementedError("Implement stable_log_softmax")


def cross_entropy(
    logits: np.ndarray,
    targets: np.ndarray,
    reduction: str = "mean",
) -> np.ndarray | float:
    """Compute cross entropy from ``[N, C]`` logits and ``[N]`` targets."""
    # TODO:
    # log_p = ...
    # losses = ...
    # return the requested reduction
    raise NotImplementedError("Implement cross_entropy")


if __name__ == "__main__":
    logits = np.array([[2.0, 1.0, 0.0]])
    targets = np.array([0])

    print("softmax:", stable_softmax(logits))
    print("log_softmax:", stable_log_softmax(logits))
    print("cross_entropy:", cross_entropy(logits, targets))

"""Day 01 exercise: stable softmax, log-softmax, and cross entropy."""

from __future__ import annotations

from typing import Literal

import numpy as np


def stable_softmax(logits: np.ndarray, axis: int = -1) -> np.ndarray:
    """Compute a numerically stable softmax along ``axis``.

    Args:
        logits: Floating-point input array.
        axis: Dimension whose values are normalized.

    Returns:
        An array with the same shape as ``logits``.
    """
    # TODO:
    # 1. Validate the input and axis.
    # 2. Subtract the maximum value along ``axis``.
    # 3. Exponentiate, sum, and normalize without changing the shape.
    raise NotImplementedError("Implement stable_softmax")


def stable_log_softmax(logits: np.ndarray, axis: int = -1) -> np.ndarray:
    """Compute a numerically stable log-softmax along ``axis``.

    Do not implement this as ``np.log(stable_softmax(...))``.
    """
    # TODO:
    # 1. Reuse the same maximum-shift idea as stable_softmax.
    # 2. Compute shifted - log(sum(exp(shifted))).
    raise NotImplementedError("Implement stable_log_softmax")


def cross_entropy(
    logits: np.ndarray,
    targets: np.ndarray,
    reduction: Literal["none", "mean", "sum"] = "mean",
) -> np.ndarray | float:
    """Compute multiclass cross entropy from logits.

    Args:
        logits: A floating-point array shaped ``[N, C]``.
        targets: Integer class indices shaped ``[N]``.
        reduction: ``none``, ``mean``, or ``sum``.

    Returns:
        Per-sample losses for ``none``; otherwise a scalar.

    Raises:
        ValueError: If shapes, target values, or reduction are invalid.
    """
    # TODO:
    # 1. Validate logits, targets, and reduction.
    # 2. Compute log probabilities with stable_log_softmax.
    # 3. Select the target class for every sample using vectorized indexing.
    # 4. Apply none, mean, or sum reduction.
    raise NotImplementedError("Implement cross_entropy")

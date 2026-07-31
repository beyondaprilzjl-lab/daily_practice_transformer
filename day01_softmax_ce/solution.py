"""Day 01 solution: stable softmax, log-softmax, and cross entropy."""

from __future__ import annotations

from typing import Literal

import numpy as np


Reduction = Literal["none", "mean", "sum"]


def _validate_logits(logits: np.ndarray, axis: int) -> tuple[np.ndarray, int]:
    """Return validated logits and a normalized non-negative axis."""
    array = np.asarray(logits)

    if array.ndim == 0:
        raise ValueError("logits must have at least one dimension")
    if not isinstance(axis, (int, np.integer)):
        raise TypeError("axis must be an integer")
    if not -array.ndim <= axis < array.ndim:
        raise ValueError(
            f"axis {axis} is out of bounds for an array with {array.ndim} dimensions"
        )
    if not np.issubdtype(array.dtype, np.floating):
        raise TypeError("logits must use a floating-point dtype")

    normalized_axis = int(axis) % array.ndim
    if array.shape[normalized_axis] == 0:
        raise ValueError("the normalized axis must not be empty")
    if not np.isfinite(array).all():
        raise ValueError("logits must contain only finite values")

    return array, normalized_axis


def stable_softmax(logits: np.ndarray, axis: int = -1) -> np.ndarray:
    """Compute a numerically stable softmax along ``axis``."""
    array, normalized_axis = _validate_logits(logits, axis)

    maximum = np.max(array, axis=normalized_axis, keepdims=True)
    shifted = array - maximum
    exponentials = np.exp(shifted)
    normalizer = np.sum(exponentials, axis=normalized_axis, keepdims=True)
    return exponentials / normalizer


def stable_log_softmax(logits: np.ndarray, axis: int = -1) -> np.ndarray:
    """Compute log-softmax directly with the log-sum-exp trick."""
    array, normalized_axis = _validate_logits(logits, axis)

    maximum = np.max(array, axis=normalized_axis, keepdims=True)
    shifted = array - maximum
    log_normalizer = np.log(
        np.sum(np.exp(shifted), axis=normalized_axis, keepdims=True)
    )
    return shifted - log_normalizer


def cross_entropy(
    logits: np.ndarray,
    targets: np.ndarray,
    reduction: Reduction = "mean",
) -> np.ndarray | float:
    """Compute multiclass cross entropy directly from logits.

    Args:
        logits: Floating-point scores shaped ``[N, C]``.
        targets: Integer class indices shaped ``[N]``.
        reduction: ``none``, ``mean``, or ``sum``.

    Returns:
        Per-sample losses for ``none``; otherwise a Python float.
    """
    array = np.asarray(logits)
    target_array = np.asarray(targets)

    if array.ndim != 2:
        raise ValueError("logits must have shape [N, C]")
    if target_array.ndim != 1:
        raise ValueError("targets must have shape [N]")
    if array.shape[0] == 0:
        raise ValueError("the batch dimension must not be empty")
    if array.shape[1] == 0:
        raise ValueError("the class dimension must not be empty")
    if target_array.shape[0] != array.shape[0]:
        raise ValueError("targets length must match the logits batch size")
    if not np.issubdtype(target_array.dtype, np.integer):
        raise TypeError("targets must use an integer dtype")
    if reduction not in {"none", "mean", "sum"}:
        raise ValueError("reduction must be 'none', 'mean', or 'sum'")

    num_classes = array.shape[1]
    if np.any(target_array < 0) or np.any(target_array >= num_classes):
        raise ValueError("targets contain a class index outside [0, C)")

    log_probabilities = stable_log_softmax(array, axis=1)
    row_indices = np.arange(array.shape[0])
    losses = -log_probabilities[row_indices, target_array]

    if reduction == "none":
        return losses
    if reduction == "sum":
        return float(np.sum(losses))
    return float(np.mean(losses))

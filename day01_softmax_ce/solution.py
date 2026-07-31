"""Day 01 interview solution: softmax, log-softmax, and cross entropy."""

from __future__ import annotations

import numpy as np


def stable_softmax(logits: np.ndarray, axis: int = -1) -> np.ndarray:
    """Compute a numerically stable softmax along ``axis``."""
    shifted = logits - np.max(logits, axis=axis, keepdims=True)
    exp_x = np.exp(shifted)
    return exp_x / np.sum(exp_x, axis=axis, keepdims=True)


def stable_log_softmax(logits: np.ndarray, axis: int = -1) -> np.ndarray:
    """Compute log-softmax directly with the log-sum-exp trick."""
    shifted = logits - np.max(logits, axis=axis, keepdims=True)
    return shifted - np.log(
        np.sum(np.exp(shifted), axis=axis, keepdims=True)
    )


def cross_entropy(
    logits: np.ndarray,
    targets: np.ndarray,
    reduction: str = "mean",
) -> np.ndarray | float:
    """Compute multiclass cross entropy from ``[N, C]`` logits."""
    log_p = stable_log_softmax(logits, axis=1)
    losses = -log_p[np.arange(len(targets)), targets]

    if reduction == "none":
        return losses
    if reduction == "sum":
        return float(np.sum(losses))
    if reduction == "mean":
        return float(np.mean(losses))
    raise ValueError("reduction must be 'none', 'mean', or 'sum'")


if __name__ == "__main__":
    logits = np.array([[2.0, 1.0, 0.0]])
    targets = np.array([0])

    print("softmax:", stable_softmax(logits))
    print("log_softmax:", stable_log_softmax(logits))
    print("cross_entropy:", cross_entropy(logits, targets))

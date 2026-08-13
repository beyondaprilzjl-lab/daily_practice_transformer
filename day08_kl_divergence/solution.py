"""Day 08 interview solution: KL divergence."""

from __future__ import annotations

import numpy as np


def kl_divergence(
    p: np.ndarray,
    q: np.ndarray,
    reduction: str = "mean",
    eps: float = 1e-12,
) -> np.ndarray | float:
    """Compute KL(p || q) over the last dimension."""
    p_safe = np.clip(p, eps, 1.0)
    q_safe = np.clip(q, eps, 1.0)

    losses = np.sum(
        p * (np.log(p_safe) - np.log(q_safe)),
        axis=-1,
    )

    if reduction == "none":
        return losses
    if reduction == "sum":
        return float(np.sum(losses))
    if reduction == "mean":
        return float(np.mean(losses))
    raise ValueError("reduction must be 'none', 'mean', or 'sum'")


if __name__ == "__main__":
    p = np.array(
        [[0.5, 0.5], [0.8, 0.2]],
        dtype=np.float64,
    )
    q = np.array(
        [[0.5, 0.5], [0.6, 0.4]],
        dtype=np.float64,
    )

    losses = kl_divergence(p, q, reduction="none")

    print("KL per sample:", losses)
    print("same distribution KL:", losses[0])
    print("mean KL:", kl_divergence(p, q))
    print("KL(p || q):", kl_divergence(p[1], q[1]))
    print("KL(q || p):", kl_divergence(q[1], p[1]))

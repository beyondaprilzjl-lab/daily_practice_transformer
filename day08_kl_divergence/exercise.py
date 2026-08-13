"""Day 08 exercise: KL divergence."""

from __future__ import annotations

import numpy as np


def kl_divergence(
    p: np.ndarray,
    q: np.ndarray,
    reduction: str = "mean",
    eps: float = 1e-12,
) -> np.ndarray | float:
    """Compute KL(p || q) over the last dimension."""
    # TODO:
    # 1. clip p and q before taking logarithms
    # 2. compute sum(p * (log(p) - log(q))) over axis=-1
    # 3. apply none, mean, or sum reduction
    raise NotImplementedError("Implement kl_divergence")


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

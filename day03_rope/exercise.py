"""Day 03 exercise: rotary position embedding."""

from __future__ import annotations

import torch
from torch import Tensor


def rotate_half(x: Tensor) -> Tensor:
    """Rotate every adjacent feature pair: [x1, x2] -> [-x2, x1]."""
    # TODO:
    # x1 = even-indexed features
    # x2 = odd-indexed features
    # interleave -x2 and x1
    raise NotImplementedError("Implement rotate_half")


def rope_cache(
    seq_len: int,
    dim: int,
    base: float = 10000.0,
    device: torch.device | None = None,
) -> tuple[Tensor, Tensor]:
    """Build cos and sin shaped [S, Dh]."""
    # TODO:
    # freq = ...
    # pos = ...
    # theta = outer(pos, freq)
    # repeat each angle for its feature pair
    raise NotImplementedError("Implement rope_cache")


def apply_rope(x: Tensor, cos: Tensor, sin: Tensor) -> Tensor:
    """Apply RoPE to x shaped [B, H, S, Dh]."""
    # TODO: x * cos + rotate_half(x) * sin
    raise NotImplementedError("Implement apply_rope")


if __name__ == "__main__":
    q = torch.tensor(
        [[[[1.0, 0.0, 1.0, 0.0], [1.0, 0.0, 1.0, 0.0]]]]
    )
    cos, sin = rope_cache(q.size(-2), q.size(-1), device=q.device)
    q_rot = apply_rope(q, cos, sin)

    print("position 0:", q_rot[0, 0, 0])
    print("position 1:", q_rot[0, 0, 1])
    print("norm preserved:", torch.allclose(q.norm(dim=-1), q_rot.norm(dim=-1)))

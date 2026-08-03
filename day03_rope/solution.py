"""Day 03 interview solution: rotary position embedding."""

from __future__ import annotations

import torch
from torch import Tensor


def rotate_half(x: Tensor) -> Tensor:
    """Rotate every adjacent feature pair: [x1, x2] -> [-x2, x1]."""
    x1 = x[..., 0::2]
    x2 = x[..., 1::2]
    return torch.stack((-x2, x1), dim=-1).flatten(-2)


def rope_cache(
    seq_len: int,
    dim: int,
    base: float = 10000.0,
    device: torch.device | None = None,
) -> tuple[Tensor, Tensor]:
    """Build cos and sin shaped [S, Dh]."""
    if dim % 2 != 0:
        raise ValueError("dim must be even")

    idx = torch.arange(0, dim, 2, device=device, dtype=torch.float32)
    freq = base ** (-idx / dim)
    pos = torch.arange(seq_len, device=device, dtype=torch.float32)
    theta = torch.outer(pos, freq).repeat_interleave(2, dim=-1)
    return theta.cos(), theta.sin()


def apply_rope(x: Tensor, cos: Tensor, sin: Tensor) -> Tensor:
    """Apply RoPE to x shaped [B, H, S, Dh]."""
    cos = cos.to(x)
    sin = sin.to(x)
    return x * cos + rotate_half(x) * sin


if __name__ == "__main__":
    q = torch.tensor(
        [[[[1.0, 0.0, 1.0, 0.0], [1.0, 0.0, 1.0, 0.0]]]]
    )
    cos, sin = rope_cache(q.size(-2), q.size(-1), device=q.device)
    q_rot = apply_rope(q, cos, sin)

    print("position 0:", q_rot[0, 0, 0])
    print("position 1:", q_rot[0, 0, 1])
    print("norm preserved:", torch.allclose(q.norm(dim=-1), q_rot.norm(dim=-1)))

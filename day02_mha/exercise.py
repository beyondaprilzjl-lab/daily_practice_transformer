"""Day 02 exercise: multi-head self-attention."""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn


def split_heads(x: Tensor, h: int) -> Tensor:
    """Convert [B, S, D] to [B, H, S, Dh]."""
    # TODO: split D into H and Dh, then move H before S
    raise NotImplementedError("Implement split_heads")


def merge_heads(x: Tensor) -> Tensor:
    """Convert [B, H, S, Dh] to [B, S, D]."""
    # TODO: move H after S, then combine H and Dh
    raise NotImplementedError("Implement merge_heads")


def attention(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    mask: Tensor | None = None,
) -> tuple[Tensor, Tensor]:
    """Compute scaled dot-product attention."""
    # TODO:
    # scores = ...
    # apply mask before softmax
    # attn = ...
    # return attended values and attention weights
    raise NotImplementedError("Implement attention")


class MHA(nn.Module):
    """Multi-head self-attention."""

    def __init__(self, d_model: int, n_heads: int, bias: bool = True) -> None:
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError("d_model must be divisible by n_heads")

        self.h = n_heads
        self.q_proj = nn.Linear(d_model, d_model, bias=bias)
        self.k_proj = nn.Linear(d_model, d_model, bias=bias)
        self.v_proj = nn.Linear(d_model, d_model, bias=bias)
        self.out_proj = nn.Linear(d_model, d_model, bias=bias)

    def forward(
        self,
        x: Tensor,
        mask: Tensor | None = None,
    ) -> tuple[Tensor, Tensor]:
        """Apply self-attention to x shaped [B, S, D]."""
        # TODO:
        # q, k, v = project x and split heads
        # out, attn = attention(...)
        # merge heads and apply out_proj
        raise NotImplementedError("Implement MHA.forward")


if __name__ == "__main__":
    mha = MHA(d_model=4, n_heads=2, bias=False)

    with torch.no_grad():
        mha.q_proj.weight.zero_()
        mha.k_proj.weight.zero_()
        mha.v_proj.weight.copy_(torch.eye(4))
        mha.out_proj.weight.copy_(torch.eye(4))

    x = torch.tensor(
        [[[1.0, 10.0, 100.0, 1000.0], [3.0, 30.0, 300.0, 3000.0]]]
    )
    mask = torch.tril(torch.ones(2, 2, dtype=torch.bool))

    out, attn = mha(x, mask)
    print("output:", out)
    print("head 0 attention:", attn[0, 0])

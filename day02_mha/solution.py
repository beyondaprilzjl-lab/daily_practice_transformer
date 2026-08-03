"""Day 02 interview solution: multi-head self-attention."""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn


def split_heads(x: Tensor, h: int) -> Tensor:
    """Convert [B, S, D] to [B, H, S, Dh]."""
    b, s, d = x.shape
    dh = d // h
    return x.reshape(b, s, h, dh).transpose(1, 2)


def merge_heads(x: Tensor) -> Tensor:
    """Convert [B, H, S, Dh] to [B, S, D]."""
    b, h, s, dh = x.shape
    return x.transpose(1, 2).contiguous().view(b, s, h * dh)


def attention(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    mask: Tensor | None = None,
) -> tuple[Tensor, Tensor]:
    """Compute scaled dot-product attention."""
    scores = q @ k.transpose(-2, -1) / math.sqrt(q.size(-1))

    if mask is not None:
        scores = scores.masked_fill(~mask, float("-inf"))

    attn = torch.softmax(scores, dim=-1)
    return attn @ v, attn


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
        q = split_heads(self.q_proj(x), self.h)
        k = split_heads(self.k_proj(x), self.h)
        v = split_heads(self.v_proj(x), self.h)

        out, attn = attention(q, k, v, mask)
        out = self.out_proj(merge_heads(out))
        return out, attn


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

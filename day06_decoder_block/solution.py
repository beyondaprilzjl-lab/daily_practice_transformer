"""Day 06 interview solution: Pre-Norm Transformer Decoder Block."""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn


class RMSNorm(nn.Module):
    """RMSNorm over the last dimension."""

    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: Tensor) -> Tensor:
        rms = torch.sqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return x / rms * self.weight


def split_heads(x: Tensor, h: int) -> Tensor:
    """Convert [B, S, D] to [B, H, S, Dh]."""
    b, s, d = x.shape
    dh = d // h
    return x.view(b, s, h, dh).transpose(1, 2)


def merge_heads(x: Tensor) -> Tensor:
    """Convert [B, H, S, Dh] to [B, S, D]."""
    b, h, s, dh = x.shape
    return x.transpose(1, 2).contiguous().view(b, s, h * dh)


class MHA(nn.Module):
    """Multi-head causal self-attention."""

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        bias: bool = False,
    ) -> None:
        super().__init__()
        self.h = n_heads
        self.q = nn.Linear(d_model, d_model, bias=bias)
        self.k = nn.Linear(d_model, d_model, bias=bias)
        self.v = nn.Linear(d_model, d_model, bias=bias)
        self.out = nn.Linear(d_model, d_model, bias=bias)

    def forward(
        self,
        x: Tensor,
        mask: Tensor | None = None,
    ) -> tuple[Tensor, Tensor]:
        q = split_heads(self.q(x), self.h)
        k = split_heads(self.k(x), self.h)
        v = split_heads(self.v(x), self.h)

        scores = q @ k.transpose(-2, -1) / math.sqrt(q.size(-1))
        if mask is not None:
            scores = scores.masked_fill(~mask, float("-inf"))

        attn = torch.softmax(scores, dim=-1)
        out = merge_heads(attn @ v)
        return self.out(out), attn


def silu(x: Tensor) -> Tensor:
    """Apply the SiLU activation."""
    return x * torch.sigmoid(x)


class SwiGLU(nn.Module):
    """SwiGLU feed-forward network."""

    def __init__(
        self,
        d_model: int,
        d_ff: int,
        bias: bool = False,
    ) -> None:
        super().__init__()
        self.gate = nn.Linear(d_model, d_ff, bias=bias)
        self.up = nn.Linear(d_model, d_ff, bias=bias)
        self.down = nn.Linear(d_ff, d_model, bias=bias)

    def forward(self, x: Tensor) -> Tensor:
        return self.down(silu(self.gate(x)) * self.up(x))


class DecoderBlock(nn.Module):
    """Pre-Norm Transformer Decoder Block."""

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        d_ff: int,
    ) -> None:
        super().__init__()
        self.norm1 = RMSNorm(d_model)
        self.attn = MHA(d_model, n_heads)
        self.norm2 = RMSNorm(d_model)
        self.ffn = SwiGLU(d_model, d_ff)

    def forward(
        self,
        x: Tensor,
        mask: Tensor | None = None,
    ) -> tuple[Tensor, Tensor]:
        """Apply attention and FFN with two residual connections."""
        a, attn = self.attn(self.norm1(x), mask)
        x = x + a
        x = x + self.ffn(self.norm2(x))
        return x, attn


if __name__ == "__main__":
    torch.manual_seed(0)

    x = torch.randn(2, 4, 8, requires_grad=True)
    mask = torch.tril(torch.ones(4, 4, dtype=torch.bool))
    block = DecoderBlock(d_model=8, n_heads=2, d_ff=16)

    out, attn = block(x, mask)
    out.mean().backward()

    print("input shape:", x.shape)
    print("output shape:", out.shape)
    print("attention shape:", attn.shape)
    print("future masked:", bool((attn[0, 0, 0, 1:] == 0).all()))
    print("input has grad:", x.grad is not None)

"""Day 07 exercise: KV Cache for autoregressive attention."""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn


def split_heads(x: Tensor, h: int) -> Tensor:
    """Convert [B, S, D] to [B, H, S, Dh]."""
    b, s, d = x.shape
    dh = d // h
    return x.view(b, s, h, dh).transpose(1, 2)


def merge_heads(x: Tensor) -> Tensor:
    """Convert [B, H, S, Dh] to [B, S, D]."""
    b, h, s, dh = x.shape
    return x.transpose(1, 2).contiguous().view(b, s, h * dh)


class CachedMHA(nn.Module):
    """Multi-head attention with a KV Cache."""

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        bias: bool = False,
    ) -> None:
        super().__init__()
        # TODO: create h, q, k, v, and out
        raise NotImplementedError("Implement CachedMHA.__init__")

    def forward(
        self,
        x: Tensor,
        cache: tuple[Tensor, Tensor] | None = None,
    ) -> tuple[Tensor, tuple[Tensor, Tensor]]:
        """Run prefill or one-token cached decode."""
        # TODO:
        # 1. project and split q, k, v
        # 2. use a causal mask during prefill
        # 3. append old k and v during decode
        # 4. compute attention
        # 5. return output and the updated (k, v) cache
        raise NotImplementedError("Implement CachedMHA.forward")


if __name__ == "__main__":
    torch.manual_seed(0)

    x = torch.randn(1, 4, 8)
    mha = CachedMHA(d_model=8, n_heads=2)

    with torch.no_grad():
        full, _ = mha(x)

        _, cache = mha(x[:, :3])
        print("prefill cache shape:", cache[0].shape)

        step, cache = mha(x[:, 3:4], cache)
        print("decode cache shape:", cache[0].shape)

    print("cached output shape:", step.shape)
    print("matches full output:", torch.allclose(step, full[:, 3:4]))

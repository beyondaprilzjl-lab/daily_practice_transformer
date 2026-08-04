"""Day 02 upgrade exercise: self and cross attention."""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn


def split_heads(x: Tensor, h: int) -> Tensor:
    """Convert [B, T, D] to [B, H, T, Dh]."""
    # TODO: reshape D into H and Dh, then move H before T
    raise NotImplementedError("Implement split_heads")


def merge_heads(x: Tensor) -> Tensor:
    """Convert [B, H, T, Dh] to [B, T, D]."""
    # TODO: move H after T, then combine H and Dh
    raise NotImplementedError("Implement merge_heads")


def stable_softmax(x: Tensor, dim: int = -1) -> Tensor:
    """Compute Softmax after subtracting the maximum value."""
    # TODO: shift x, compute exp, then normalize
    raise NotImplementedError("Implement stable_softmax")


def make_mask(
    pad: Tensor | None,
    tq: int,
    tk: int,
    causal: bool,
    device: torch.device,
) -> Tensor | None:
    """Build a mask broadcastable to [B, H, Tq, Tk]."""
    # TODO:
    # pad mask:    [B, Tk] -> [B, 1, 1, Tk]
    # causal mask: [Tq, Tk] -> [1, 1, Tq, Tk]
    # combine them with logical AND
    raise NotImplementedError("Implement make_mask")


def attention(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    mask: Tensor | None = None,
) -> tuple[Tensor, Tensor]:
    """Compute scaled dot-product attention."""
    # TODO:
    # scores = QK^T / sqrt(Dh)
    # apply mask before stable Softmax
    # return attn @ V and attn
    raise NotImplementedError("Implement attention")


class MHA(nn.Module):
    """Multi-head self or cross attention."""

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
        q_x: Tensor,
        kv_x: Tensor | None = None,
        pad: Tensor | None = None,
        causal: bool = False,
    ) -> tuple[Tensor, Tensor]:
        """Apply self attention when kv_x is None, otherwise cross attention."""
        # TODO:
        # use q_x as kv_x for self attention
        # project Q from q_x and K/V from kv_x
        # build mask, calculate attention, merge heads and project output
        raise NotImplementedError("Implement MHA.forward")


if __name__ == "__main__":
    torch.manual_seed(0)
    mha = MHA(d_model=4, n_heads=2)

    x = torch.randn(2, 3, 4)
    pad = torch.tensor(
        [[True, True, False], [True, True, True]]
    )
    self_out, self_attn = mha(x, pad=pad, causal=True)

    memory = torch.randn(2, 4, 4)
    memory_pad = torch.tensor(
        [[True, True, False, False], [True, True, True, False]]
    )
    cross_out, cross_attn = mha(x, kv_x=memory, pad=memory_pad)

    print("self output shape:", tuple(self_out.shape))
    print("self attention shape:", tuple(self_attn.shape))
    print("self head 0:", self_attn[0, 0])
    print("cross output shape:", tuple(cross_out.shape))
    print("cross attention shape:", tuple(cross_attn.shape))
    print("cross padded probabilities:", cross_attn[0, 0, :, 2:])

"""Day 02 upgrade solution: self and cross attention."""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn


def split_heads(x: Tensor, h: int) -> Tensor:
    """Convert [B, T, D] to [B, H, T, Dh]."""
    b, t, d = x.shape
    dh = d // h
    return x.reshape(b, t, h, dh).transpose(1, 2)


def merge_heads(x: Tensor) -> Tensor:
    """Convert [B, H, T, Dh] to [B, T, D]."""
    b, h, t, dh = x.shape
    return x.transpose(1, 2).contiguous().view(b, t, h * dh)


def stable_softmax(x: Tensor, dim: int = -1) -> Tensor:
    """Compute Softmax after subtracting the maximum value."""
    x = x - x.max(dim=dim, keepdim=True).values
    exp_x = torch.exp(x)
    return exp_x / exp_x.sum(dim=dim, keepdim=True)


def make_mask(
    pad: Tensor | None,
    tq: int,
    tk: int,
    causal: bool,
    device: torch.device,
) -> Tensor | None:
    """Build a mask broadcastable to [B, H, Tq, Tk]."""
    mask = pad[:, None, None, :].to(device) if pad is not None else None

    if causal:
        c_mask = torch.tril(
            torch.ones(tq, tk, dtype=torch.bool, device=device)
        )[None, None, :, :]
        mask = c_mask if mask is None else mask & c_mask

    return mask


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

    attn = stable_softmax(scores, dim=-1)
    return attn @ v, attn


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
        kv_x = q_x if kv_x is None else kv_x

        q = split_heads(self.q_proj(q_x), self.h)
        k = split_heads(self.k_proj(kv_x), self.h)
        v = split_heads(self.v_proj(kv_x), self.h)

        mask = make_mask(
            pad,
            q.size(-2),
            k.size(-2),
            causal,
            q.device,
        )
        out, attn = attention(q, k, v, mask)
        out = self.out_proj(merge_heads(out))
        return out, attn


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

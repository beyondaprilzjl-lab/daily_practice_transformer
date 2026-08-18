"""Day 09 interview solution: LayerNorm."""

import torch
from torch import Tensor, nn


def layer_norm(
    x: Tensor,
    weight: Tensor,
    bias: Tensor,
    eps: float = 1e-5,
) -> Tensor:
    """Apply LayerNorm over the last dimension."""
    mean = x.mean(dim=-1, keepdim=True)
    centered = x - mean
    var = centered.pow(2).mean(dim=-1, keepdim=True)
    return centered / torch.sqrt(var + eps) * weight + bias


class LayerNorm(nn.Module):
    """LayerNorm with a learnable scale and bias."""

    def __init__(self, dim: int, eps: float = 1e-5) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))
        self.bias = nn.Parameter(torch.zeros(dim))

    def forward(self, x: Tensor) -> Tensor:
        """Normalize x over its last dimension."""
        return layer_norm(x, self.weight, self.bias, self.eps)


if __name__ == "__main__":
    x = torch.tensor(
        [[[1.0, 2.0, 3.0, 4.0], [2.0, 2.0, 2.0, 2.0]]]
    )
    norm = LayerNorm(dim=4)
    ref = nn.LayerNorm(4, eps=norm.eps)

    with torch.no_grad():
        ref.weight.copy_(norm.weight)
        ref.bias.copy_(norm.bias)

    out = norm(x)
    expected = ref(x)

    print("output:", out)
    print("output mean:", out.mean(dim=-1))
    print("output variance:", out.var(dim=-1, unbiased=False))
    print("matches PyTorch:", torch.allclose(out, expected))

"""Day 09 exercise: LayerNorm."""

import torch
from torch import Tensor, nn


def layer_norm(
    x: Tensor,
    weight: Tensor,
    bias: Tensor,
    eps: float = 1e-5,
) -> Tensor:
    """Apply LayerNorm over the last dimension."""
    # TODO:
    # 1. Compute the mean over the last dimension.
    # 2. Center x and compute the variance.
    # 3. Normalize, scale by weight, and add bias.
    raise NotImplementedError("Implement layer_norm")


class LayerNorm(nn.Module):
    """LayerNorm with a learnable scale and bias."""

    def __init__(self, dim: int, eps: float = 1e-5) -> None:
        super().__init__()
        # TODO: save eps, then create weight as ones and bias as zeros
        raise NotImplementedError("Implement LayerNorm.__init__")

    def forward(self, x: Tensor) -> Tensor:
        """Normalize x over its last dimension."""
        # TODO: call layer_norm
        raise NotImplementedError("Implement LayerNorm.forward")


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

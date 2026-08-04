"""Day 04 interview solution: RMSNorm."""

import torch
from torch import Tensor, nn


def rms_norm(
    x: Tensor,
    weight: Tensor,
    eps: float = 1e-6,
) -> Tensor:
    """Apply RMSNorm over the last dimension."""
    rms = torch.sqrt(x.pow(2).mean(dim=-1, keepdim=True) + eps)
    return x / rms * weight


class RMSNorm(nn.Module):
    """RMSNorm with a learnable scale."""

    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: Tensor) -> Tensor:
        """Normalize x over its last dimension."""
        return rms_norm(x, self.weight, self.eps)


if __name__ == "__main__":
    x = torch.tensor(
        [[[1.0, 2.0, 3.0, 4.0], [2.0, 2.0, 2.0, 2.0]]]
    )
    norm = RMSNorm(dim=4)
    out = norm(x)

    print("output:", out)
    print("output rms:", torch.sqrt(out.pow(2).mean(dim=-1)))

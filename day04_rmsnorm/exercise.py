"""Day 04 exercise: RMSNorm."""

import torch
from torch import Tensor, nn


def rms_norm(
    x: Tensor,
    weight: Tensor,
    eps: float = 1e-6,
) -> Tensor:
    """Apply RMSNorm over the last dimension."""
    # TODO:
    # rms = sqrt(mean(x^2) + eps)
    # return normalized x times weight
    raise NotImplementedError("Implement rms_norm")


class RMSNorm(nn.Module):
    """RMSNorm with a learnable scale."""

    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        # TODO: save eps and create a learnable weight initialized to ones
        raise NotImplementedError("Implement RMSNorm.__init__")

    def forward(self, x: Tensor) -> Tensor:
        """Normalize x over its last dimension."""
        # TODO: call rms_norm
        raise NotImplementedError("Implement RMSNorm.forward")


if __name__ == "__main__":
    x = torch.tensor(
        [[[1.0, 2.0, 3.0, 4.0], [2.0, 2.0, 2.0, 2.0]]]
    )
    norm = RMSNorm(dim=4)
    out = norm(x)

    print("output:", out)
    print("output rms:", torch.sqrt(out.pow(2).mean(dim=-1)))

"""Day 05 exercise: SwiGLU feed-forward network."""

import torch
from torch import Tensor, nn


def silu(x: Tensor) -> Tensor:
    """Apply the SiLU activation."""
    # TODO: SiLU(x) = x * sigmoid(x)
    raise NotImplementedError("Implement silu")


class SwiGLU(nn.Module):
    """SwiGLU feed-forward network."""

    def __init__(
        self,
        d_model: int,
        d_ff: int,
        bias: bool = False,
    ) -> None:
        super().__init__()
        # TODO: create gate, up, and down linear layers
        raise NotImplementedError("Implement SwiGLU.__init__")

    def forward(self, x: Tensor) -> Tensor:
        """Apply the SwiGLU feed-forward network."""
        # TODO:
        # g = silu(gate(x))
        # u = up(x)
        # return down(g * u)
        raise NotImplementedError("Implement SwiGLU.forward")


if __name__ == "__main__":
    x = torch.tensor([[[1.0, 2.0], [3.0, 4.0]]])
    ffn = SwiGLU(d_model=2, d_ff=2, bias=False)

    with torch.no_grad():
        ffn.gate.weight.copy_(torch.eye(2))
        ffn.up.weight.copy_(torch.eye(2))
        ffn.down.weight.copy_(torch.eye(2))

    out = ffn(x)
    expected = silu(x) * x

    print("input shape:", x.shape)
    print("output shape:", out.shape)
    print("output:", out)
    print("matches expected:", torch.allclose(out, expected))

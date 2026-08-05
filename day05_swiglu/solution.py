"""Day 05 interview solution: SwiGLU feed-forward network."""

import torch
from torch import Tensor, nn


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
        """Apply the SwiGLU feed-forward network."""
        g = silu(self.gate(x))
        u = self.up(x)
        return self.down(g * u)


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

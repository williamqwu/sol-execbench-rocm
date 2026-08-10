import torch


@torch.no_grad()
def run(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
    n, _, h, w = x.shape
    out = torch.empty((n, 64, (h + 1) // 2, (w + 1) // 2), device=x.device, dtype=x.dtype)
    return torch.ops.aten.miopen_convolution.out(
        x, weight, bias, [1, 1], [2, 2], [1, 1], 1, False, False, out=out
    )

import torch


@torch.compile
def _impl(x, weight, bias, eps):
    B, C, H, W = x.shape
    z = x.view(B, 32, C // 32, H, W)
    mean = z.mean(dim=(2, 3, 4), keepdim=True)
    var = z.var(dim=(2, 3, 4), keepdim=True, unbiased=False)
    z = (z - mean) / torch.sqrt(var + eps)
    return z.view(B, C, H, W) * weight.view(1, C, 1, 1) + bias.view(1, C, 1, 1)


@torch.no_grad()
def run(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor, eps: float) -> torch.Tensor:
    return _impl(x, weight, bias, eps)

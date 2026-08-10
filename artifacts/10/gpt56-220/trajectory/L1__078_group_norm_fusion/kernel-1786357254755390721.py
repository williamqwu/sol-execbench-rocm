import torch


@torch.compile(mode="max-autotune-no-cudagraphs", fullgraph=True)
def _impl(x, weight, bias, eps):
    B, C, H, W = x.shape
    z = x.view(B, 32, C // 32, H, W)
    var, mean = torch.var_mean(z, dim=(2, 3, 4), correction=0, keepdim=True)
    z = (z - mean) / torch.sqrt(var + eps)
    return z.view(B, C, H, W) * weight.view(1, C, 1, 1) + bias.view(1, C, 1, 1)


@torch.no_grad()
def run(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor, eps: float) -> torch.Tensor:
    return _impl(x, weight, bias, eps)

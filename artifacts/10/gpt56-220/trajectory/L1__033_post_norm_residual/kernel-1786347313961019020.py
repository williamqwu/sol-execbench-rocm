import torch


@torch.compile(fullgraph=True, dynamic=False)
def _compiled(
    sublayer_output: torch.Tensor,
    residual: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
) -> torch.Tensor:
    x = sublayer_output.float()
    variance = x.square().mean(dim=-1, keepdim=True)
    normalized = x * torch.rsqrt(variance + eps)
    normalized = (normalized * weight.float()).to(sublayer_output.dtype)
    return residual + normalized


@torch.no_grad()
def run(
    sublayer_output: torch.Tensor,
    residual: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
) -> torch.Tensor:
    return _compiled(sublayer_output, residual, weight, eps)

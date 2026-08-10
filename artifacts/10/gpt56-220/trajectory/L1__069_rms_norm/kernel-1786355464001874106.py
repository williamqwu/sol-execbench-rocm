import torch


@torch.compile
def _compiled(hidden_states: torch.Tensor, residual: torch.Tensor,
              weight: torch.Tensor, eps: float) -> torch.Tensor:
    x = residual + hidden_states
    x_fp32 = x.float()
    variance = (x_fp32 * x_fp32).mean(-1, keepdim=True)
    return weight * (x_fp32 * torch.rsqrt(variance + eps)).bfloat16()


@torch.no_grad()
def run(hidden_states: torch.Tensor, residual: torch.Tensor,
        weight: torch.Tensor, eps: float) -> torch.Tensor:
    return _compiled(hidden_states, residual, weight, eps)

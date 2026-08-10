import torch


@torch.no_grad()
def run(grad_output: torch.Tensor, x: torch.Tensor, sigmoid_x: torch.Tensor) -> torch.Tensor:
    return torch.ops.aten.silu_backward(grad_output, x)

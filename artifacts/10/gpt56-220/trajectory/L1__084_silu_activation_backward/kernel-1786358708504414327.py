import torch


@torch.compile
@torch.no_grad()
def run(grad_output: torch.Tensor, x: torch.Tensor, sigmoid_x: torch.Tensor) -> torch.Tensor:
    return grad_output * sigmoid_x * (1.0 + x * (1.0 - sigmoid_x))

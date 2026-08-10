import torch


@torch.compile(fullgraph=True)
def _gelu_formula(x: torch.Tensor) -> torch.Tensor:
    sigmoid_arg = x * (1.5957691216057308 + 0.07135481627260025 * x * x)
    return x * torch.sigmoid(sigmoid_arg)


@torch.no_grad()
def run(x: torch.Tensor) -> torch.Tensor:
    return _gelu_formula(x)

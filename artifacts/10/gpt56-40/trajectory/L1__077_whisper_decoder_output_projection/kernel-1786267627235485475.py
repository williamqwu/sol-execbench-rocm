import torch


@torch.compile(fullgraph=True, dynamic=True, mode="max-autotune")
def _project(hidden_states: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    return torch.matmul(hidden_states, weight.t())


@torch.no_grad()
def run(hidden_states: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    return _project(hidden_states, weight)

import torch


@torch.compile
def _compiled(query: torch.Tensor, key: torch.Tensor, scaling: float) -> torch.Tensor:
    return (torch.matmul(query, key.transpose(2, 3)) * scaling).to(query.dtype)

@torch.no_grad()
def run(query: torch.Tensor, key: torch.Tensor, scaling: float) -> torch.Tensor:
    return _compiled(query, key, scaling)

import torch


@torch.compile
def _project(hidden_states: torch.Tensor, v_proj_weight: torch.Tensor) -> torch.Tensor:
    weight_by_head = v_proj_weight.view(8, 128, 5120)
    return torch.einsum("bsh,ndh->bnsd", hidden_states, weight_by_head)

@torch.no_grad()
def run(hidden_states: torch.Tensor, v_proj_weight: torch.Tensor) -> torch.Tensor:
    return _project(hidden_states, v_proj_weight)

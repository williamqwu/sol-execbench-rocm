import torch


@torch.no_grad()
def run(hidden_states: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    """LM head projection: [B, S, H] @ [H, V] -> [B, S, V]."""
    return torch.matmul(hidden_states, weight.t())

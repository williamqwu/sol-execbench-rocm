import torch


def _project(hidden_states: torch.Tensor, v_proj_weight: torch.Tensor) -> torch.Tensor:
    batch_size, seq_len, _ = hidden_states.shape
    value_proj = torch.mm(hidden_states.view(-1, 5120), v_proj_weight.t())
    return value_proj.view(batch_size, seq_len, 8, 128).transpose(1, 2)

@torch.no_grad()
def run(hidden_states: torch.Tensor, v_proj_weight: torch.Tensor) -> torch.Tensor:
    return _project(hidden_states, v_proj_weight)

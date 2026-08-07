import torch

_compiled = None

def _fn(hidden_states, v_proj_weight):
    batch_size, seq_len, hidden_size = hidden_states.shape
    value_proj = torch.nn.functional.linear(hidden_states, v_proj_weight)
    value_states = value_proj.view(batch_size, seq_len, 8, 128)
    value_states = value_states.transpose(1, 2).contiguous()
    return value_states

@torch.no_grad()
def run(hidden_states: torch.Tensor, v_proj_weight: torch.Tensor) -> torch.Tensor:
    global _compiled
    if _compiled is None:
        _compiled = torch.compile(_fn, mode="max-autotune", dynamic=True)
    return _compiled(hidden_states, v_proj_weight)

import torch

@torch.no_grad()
def run(hidden_states: torch.Tensor, v_proj_weight: torch.Tensor) -> torch.Tensor:
    batch_size, seq_len, hidden_size = hidden_states.shape
    num_kv_heads = 8
    head_dim = 128
    # Use 2D matmul with contiguous-transposed weight, then reshape+transpose.
    wt = v_proj_weight.t().contiguous()
    h2 = hidden_states.reshape(-1, hidden_size)
    value_proj = h2 @ wt
    value_states = value_proj.view(batch_size, seq_len, num_kv_heads, head_dim)
    value_states = value_states.transpose(1, 2).contiguous()
    return value_states

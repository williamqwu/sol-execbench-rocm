import torch

@torch.no_grad()
def run(hidden_states: torch.Tensor, v_proj_weight: torch.Tensor) -> torch.Tensor:
    batch_size, seq_len, hidden_size = hidden_states.shape
    num_kv_heads = 8
    head_dim = 128
    # Reshape weight to per-head form [num_kv_heads, head_dim, hidden_size].
    w = v_proj_weight.view(num_kv_heads, head_dim, hidden_size)
    if batch_size == 1 and seq_len >= 2048:
        # Batched GEMM broadcasting the single batch over heads writes
        # directly to [1, num_kv_heads, seq_len, head_dim], fusing the transpose.
        wt = w.transpose(1, 2).unsqueeze(0)  # [1, hidden_size, head_dim]
        value_states = torch.matmul(hidden_states.unsqueeze(1), wt)
        return value_states
    # einsum fuses projection + reshape + transpose into one contraction
    # writing directly to [batch, num_kv_heads, seq_len, head_dim].
    value_states = torch.einsum('bsk,hdk->bhsd', hidden_states, w)
    return value_states

import torch

@torch.no_grad()
def run(hidden_states: torch.Tensor, v_proj_weight: torch.Tensor) -> torch.Tensor:
    batch_size, seq_len, hidden_size = hidden_states.shape
    num_kv_heads = 8
    head_dim = 128
    # einsum fuses projection + reshape + transpose into one contraction
    # that writes directly to [batch, num_kv_heads, seq_len, head_dim].
    w = v_proj_weight.view(num_kv_heads, head_dim, hidden_size)
    value_states = torch.einsum('bsk,hdk->bhsd', hidden_states, w)
    return value_states

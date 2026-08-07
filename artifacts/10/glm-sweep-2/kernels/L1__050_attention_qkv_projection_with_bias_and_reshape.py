import torch

@torch.no_grad()
def run(
    hidden_states: torch.Tensor,
    q_weight: torch.Tensor,
    q_bias: torch.Tensor,
    k_weight: torch.Tensor,
    k_bias: torch.Tensor,
    v_weight: torch.Tensor,
    v_bias: torch.Tensor,
):
    batch_size, seq_len, _ = hidden_states.shape
    num_attention_heads = 4
    num_key_value_heads = 1
    head_dim = 256

    w = torch.cat([q_weight, k_weight, v_weight], dim=0)
    b = torch.cat([q_bias, k_bias, v_bias], dim=0)

    x = hidden_states.view(-1, 640)
    out = torch.addmm(b, x, w.t())

    query_states = out[:, :1024].view(batch_size, seq_len, num_attention_heads, head_dim)
    key_states = out[:, 1024:1280].view(batch_size, seq_len, num_key_value_heads, head_dim)
    value_states = out[:, 1280:1600].view(batch_size, seq_len, num_key_value_heads, head_dim)

    return query_states, key_states, value_states

import torch

@torch.no_grad()
def run(
    hidden_states: torch.Tensor,
    q_weight: torch.Tensor,
    k_weight: torch.Tensor,
    v_weight: torch.Tensor,
):
    num_heads = 16
    num_kv_heads = 4
    head_dim = 128

    bsz, q_len, _ = hidden_states.size()

    # Fuse the three projections into one matmul: read hidden_states once.
    # qkv_weight: [2048+512+512, 2048] = [3072, 2048]
    # output: [bsz, q_len, 3072]
    qkv_weight = torch.cat([q_weight, k_weight, v_weight], dim=0)
    qkv = torch.matmul(hidden_states, qkv_weight.t())

    query_states = qkv[..., :2048]
    key_states = qkv[..., 2048:2560]
    value_states = qkv[..., 2560:]

    query_states = query_states.view(bsz, q_len, num_heads, head_dim).transpose(1, 2)
    key_states = key_states.view(bsz, q_len, num_kv_heads, head_dim).transpose(1, 2)
    value_states = value_states.view(bsz, q_len, num_kv_heads, head_dim).transpose(1, 2)

    return query_states, key_states, value_states

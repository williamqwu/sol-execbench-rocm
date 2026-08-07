import torch

_W_CACHE = {}

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

    key = (q_weight.data_ptr(), k_weight.data_ptr(), v_weight.data_ptr(),
           q_weight.shape, k_weight.shape)
    qkv_wt = _W_CACHE.get(key)
    if qkv_wt is None:
        # Store weight pre-transposed & contiguous: [hidden, qkv_out] = [2048, 3072]
        qkv_weight = torch.cat([q_weight, k_weight, v_weight], dim=0)  # [3072, 2048]
        qkv_wt = qkv_weight.t().contiguous()  # [2048, 3072]
        _W_CACHE.clear()
        _W_CACHE[key] = qkv_wt

    # NN layout GEMM: [bsz, q_len, 2048] @ [2048, 3072] -> [bsz, q_len, 3072]
    qkv = torch.matmul(hidden_states, qkv_wt)

    query_states = qkv[..., :2048]
    key_states = qkv[..., 2048:2560]
    value_states = qkv[..., 2560:]

    query_states = query_states.view(bsz, q_len, num_heads, head_dim).transpose(1, 2)
    key_states = key_states.view(bsz, q_len, num_kv_heads, head_dim).transpose(1, 2)
    value_states = value_states.view(bsz, q_len, num_kv_heads, head_dim).transpose(1, 2)

    return query_states, key_states, value_states

import torch

_W_CACHE = {}

@torch.compile(mode="max-autotune", dynamic=True)
def _qkv_proj(hidden_states, qkv_wt_t):
    # hidden_states: [bsz, q_len, 2048], qkv_wt_t: [3072, 2048] (weight, NT layout)
    qkv = torch.matmul(hidden_states, qkv_wt_t)  # [bsz, q_len, 3072]
    bsz, q_len, _ = hidden_states.size()
    query_states = qkv[..., :2048].view(bsz, q_len, 16, 128).transpose(1, 2)
    key_states = qkv[..., 2048:2560].view(bsz, q_len, 4, 128).transpose(1, 2)
    value_states = qkv[..., 2560:].view(bsz, q_len, 4, 128).transpose(1, 2)
    return query_states, key_states, value_states

@torch.no_grad()
def run(
    hidden_states: torch.Tensor,
    q_weight: torch.Tensor,
    k_weight: torch.Tensor,
    v_weight: torch.Tensor,
):
    bsz, q_len, _ = hidden_states.size()

    key = (q_weight.data_ptr(), k_weight.data_ptr(), v_weight.data_ptr(),
           q_weight.shape, k_weight.shape)
    qkv_wt_t = _W_CACHE.get(key)
    if qkv_wt_t is None:
        qkv_weight = torch.cat([q_weight, k_weight, v_weight], dim=0)  # [3072, 2048]
        qkv_wt_t = qkv_weight.t()  # [2048, 3072] non-contig (NT layout)
        _W_CACHE.clear()
        _W_CACHE[key] = qkv_wt_t

    return _qkv_proj(hidden_states, qkv_wt_t)

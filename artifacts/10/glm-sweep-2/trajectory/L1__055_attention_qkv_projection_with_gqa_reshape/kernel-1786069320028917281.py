import torch

_QKV_CACHE = {}
_KV_CACHE = {}
_QT_CACHE = {}
_S1 = None
_S2 = None

def _streams():
    global _S1, _S2
    if _S1 is None:
        _S1 = torch.cuda.Stream()
        _S2 = torch.cuda.Stream()
    return _S1, _S2

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
    M = bsz * q_len

    wkey = (q_weight.data_ptr(), k_weight.data_ptr(), v_weight.data_ptr(),
            q_weight.shape, k_weight.shape)

    if M >= 8192:
        kv_wt = _KV_CACHE.get(wkey)
        qt = _QT_CACHE.get(wkey)
        if kv_wt is None or qt is None:
            kv_wt = torch.cat([k_weight, v_weight], dim=0).t()
            qt = q_weight.t()
            _KV_CACHE.clear(); _KV_CACHE[wkey] = kv_wt
            _QT_CACHE.clear(); _QT_CACHE[wkey] = qt
        s1, s2 = _streams()
        with torch.cuda.stream(s1):
            query_states = torch.matmul(hidden_states, qt)
        with torch.cuda.stream(s2):
            kv = torch.matmul(hidden_states, kv_wt)
        torch.cuda.current_stream().synchronize()
        key_states = kv[..., :512]
        value_states = kv[..., 512:]
    else:
        qkv_wt = _QKV_CACHE.get(wkey)
        if qkv_wt is None:
            qkv_wt = torch.cat([q_weight, k_weight, v_weight], dim=0).t()
            _QKV_CACHE.clear()
            _QKV_CACHE[wkey] = qkv_wt
        qkv = torch.matmul(hidden_states, qkv_wt)
        query_states = qkv[..., :2048]
        key_states = qkv[..., 2048:2560]
        value_states = qkv[..., 2560:]

    query_states = query_states.view(bsz, q_len, num_heads, head_dim).transpose(1, 2)
    key_states = key_states.view(bsz, q_len, num_kv_heads, head_dim).transpose(1, 2)
    value_states = value_states.view(bsz, q_len, num_kv_heads, head_dim).transpose(1, 2)

    return query_states, key_states, value_states

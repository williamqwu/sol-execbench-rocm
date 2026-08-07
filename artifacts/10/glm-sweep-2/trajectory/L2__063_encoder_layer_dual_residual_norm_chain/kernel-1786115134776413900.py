import torch
import torch.nn.functional as F

_REFERENCE_RUN = None
_GRAPH_CACHE = {}

def _reference_run(hidden_states, cos, sin, pre_sa_norm_weight, q_proj_weight, k_proj_weight, v_proj_weight, o_proj_weight, post_sa_norm_weight, gate_up_proj_weight, down_proj_weight, norm_eps):
    num_heads = 16
    num_key_value_heads = 4
    head_dim = 64
    num_key_value_groups = num_heads // num_key_value_heads
    batch_size, seq_len, hidden_size = hidden_states.shape
    def rms_norm(x, weight, eps):
        input_dtype = x.dtype
        x = x.to(torch.float32)
        variance = x.pow(2).mean(-1, keepdim=True)
        x = x * torch.rsqrt(variance + eps)
        return (weight * x).to(input_dtype)
    def rotate_half(x):
        x1 = x[..., : x.shape[-1] // 2]
        x2 = x[..., x.shape[-1] // 2 :]
        return torch.cat((-x2, x1), dim=-1)
    def repeat_kv(x, n_rep):
        batch, num_kv_heads, slen, hdim = x.shape
        if n_rep == 1:
            return x
        x = x[:, :, None, :, :].expand(batch, num_kv_heads, n_rep, slen, hdim)
        return x.reshape(batch, num_kv_heads * n_rep, slen, hdim)
    residual_1 = hidden_states
    normed_1 = rms_norm(residual_1, pre_sa_norm_weight, norm_eps)
    query_states = torch.matmul(normed_1, q_proj_weight.t())
    key_states = torch.matmul(normed_1, k_proj_weight.t())
    value_states = torch.matmul(normed_1, v_proj_weight.t())
    query_states = query_states.view(batch_size, seq_len, num_heads, head_dim).transpose(1, 2)
    key_states = key_states.view(batch_size, seq_len, num_key_value_heads, head_dim).transpose(1, 2)
    value_states = value_states.view(batch_size, seq_len, num_key_value_heads, head_dim).transpose(1, 2)
    cos_expanded = cos.unsqueeze(1)
    sin_expanded = sin.unsqueeze(1)
    q_embed = (query_states * cos_expanded) + (rotate_half(query_states) * sin_expanded)
    k_embed = (key_states * cos_expanded) + (rotate_half(key_states) * sin_expanded)
    key_states = repeat_kv(k_embed, num_key_value_groups)
    value_states = repeat_kv(value_states, num_key_value_groups)
    attn_weights = torch.matmul(q_embed, key_states.transpose(2, 3)) / (head_dim ** 0.5)
    attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
    attn_output = torch.matmul(attn_weights, value_states)
    attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, -1)
    attn_out = torch.matmul(attn_output, o_proj_weight.t())
    hidden_states = residual_1 + attn_out
    residual_2 = hidden_states
    normed_2 = rms_norm(residual_2, post_sa_norm_weight, norm_eps)
    up_states = torch.matmul(normed_2, gate_up_proj_weight.t())
    gate, up_states = up_states.chunk(2, dim=-1)
    up_states = up_states * F.silu(gate)
    mlp_out = torch.matmul(up_states, down_proj_weight.t())
    output = residual_2 + mlp_out
    return output

@torch.no_grad()
def run(hidden_states, cos, sin, pre_sa_norm_weight, q_proj_weight, k_proj_weight, v_proj_weight, o_proj_weight, post_sa_norm_weight, gate_up_proj_weight, down_proj_weight, norm_eps):
    args = (hidden_states, cos, sin, pre_sa_norm_weight, q_proj_weight, k_proj_weight, v_proj_weight, o_proj_weight, post_sa_norm_weight, gate_up_proj_weight, down_proj_weight, norm_eps)
    key = hidden_states.shape
    entry = _GRAPH_CACHE.get(key)
    if entry is None:
        static_inputs = [a.clone() if isinstance(a, torch.Tensor) else a for a in args]
        for _ in range(3):
            _reference_run(*static_inputs)
        torch.cuda.synchronize()
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            out = _reference_run(*static_inputs)
        entry = (g, static_inputs, out)
        _GRAPH_CACHE[key] = entry
    g, static_inputs, out = entry
    for si, a in zip(static_inputs, args):
        if isinstance(si, torch.Tensor):
            si.copy_(a)
    g.replay()
    return out

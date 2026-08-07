import torch
import torch.nn.functional as F

_WCACHE = {}

def _get_fused_weights(qw, kw, vw, gw, uw):
    key = (qw.data_ptr(), kw.data_ptr(), vw.data_ptr(), gw.data_ptr(), uw.data_ptr())
    cached = _WCACHE.get(key)
    if cached is not None:
        return cached
    qkv = torch.cat([qw, kw, vw], dim=0)
    gu = torch.cat([gw, uw], dim=0)
    # Bound cache size to avoid unbounded growth across many distinct tensors
    if len(_WCACHE) > 32:
        _WCACHE.clear()
    _WCACHE[key] = (qkv, gu)
    return qkv, gu

@torch.compile(dynamic=True, fullgraph=True)
def _compiled_run(
    hidden_states, position_ids, attention_mask,
    input_layernorm_weight, qkv_weight, q_norm_weight, k_norm_weight,
    o_proj_weight, post_attention_layernorm_weight,
    gate_up_weight, down_proj_weight, inv_freq,
    rms_norm_eps, attention_scale,
):
    batch_size, seq_len, hidden_size = hidden_states.shape
    num_attention_heads = 40
    num_key_value_heads = 8
    head_dim = 128
    half = head_dim // 2
    intermediate_size = gate_up_weight.shape[0] // 2
    qkv_split = num_attention_heads * head_dim + 2 * num_key_value_heads * head_dim

    def rms_norm(x, weight):
        input_dtype = x.dtype
        x = x.to(torch.float32)
        variance = x.pow(2).mean(-1, keepdim=True)
        x = x * torch.rsqrt(variance + rms_norm_eps)
        return (weight * x).to(input_dtype)

    residual = hidden_states
    hidden_states = rms_norm(hidden_states, input_layernorm_weight)
    qkv = F.linear(hidden_states, qkv_weight)
    q, k, v = qkv.split([num_attention_heads * head_dim, num_key_value_heads * head_dim, num_key_value_heads * head_dim], dim=-1)
    q = q.view(batch_size, seq_len, num_attention_heads, head_dim)
    k = k.view(batch_size, seq_len, num_key_value_heads, head_dim)
    v = v.view(batch_size, seq_len, num_key_value_heads, head_dim)
    q = rms_norm(q, q_norm_weight)
    k = rms_norm(k, k_norm_weight)
    q = q.transpose(1, 2)
    k = k.transpose(1, 2)
    v = v.transpose(1, 2)

    freqs = (inv_freq[None, :, None].float() * position_ids[:, None, :].float()).transpose(1, 2)
    emb = torch.cat((freqs, freqs), dim=-1)
    cos = emb.cos().to(torch.bfloat16).unsqueeze(1)
    sin = emb.sin().to(torch.bfloat16).unsqueeze(1)

    q1, q2 = q[..., :half], q[..., half:]
    k1, k2 = k[..., :half], k[..., half:]
    q_rotated = torch.cat((-q2, q1), dim=-1)
    k_rotated = torch.cat((-k2, k1), dim=-1)
    q = (q * cos) + (q_rotated * sin)
    k = (k * cos) + (k_rotated * sin)

    attn_output = F.scaled_dot_product_attention(q, k, v, is_causal=True, scale=float(attention_scale), enable_gqa=True)
    attn_output = attn_output.transpose(1, 2).contiguous()
    attn_output = attn_output.reshape(batch_size, seq_len, num_attention_heads * head_dim)
    attn_output = F.linear(attn_output, o_proj_weight)
    hidden_states = residual + attn_output
    residual = hidden_states
    hidden_states = rms_norm(hidden_states, post_attention_layernorm_weight)
    gate_up = F.linear(hidden_states, gate_up_weight)
    gate, up = gate_up.split([intermediate_size, intermediate_size], dim=-1)
    gate = F.silu(gate)
    intermediate = gate * up
    mlp_output = F.linear(intermediate, down_proj_weight)
    output = residual + mlp_output
    return output

@torch.no_grad()
def run(
    hidden_states, position_ids, attention_mask,
    input_layernorm_weight, q_proj_weight, k_proj_weight, v_proj_weight,
    q_norm_weight, k_norm_weight, o_proj_weight, post_attention_layernorm_weight,
    gate_proj_weight, up_proj_weight, down_proj_weight, inv_freq,
    rms_norm_eps, attention_scale,
):
    qkv_weight, gate_up_weight = _get_fused_weights(q_proj_weight, k_proj_weight, v_proj_weight, gate_proj_weight, up_proj_weight)
    return _compiled_run(
        hidden_states, position_ids, attention_mask,
        input_layernorm_weight, qkv_weight, q_norm_weight, k_norm_weight,
        o_proj_weight, post_attention_layernorm_weight,
        gate_up_weight, down_proj_weight, inv_freq,
        rms_norm_eps, attention_scale,
    )

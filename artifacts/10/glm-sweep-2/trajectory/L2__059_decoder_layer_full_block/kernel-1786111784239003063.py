import torch
import torch.nn.functional as F

@torch.compile(mode="max-autotune-no-cudagraphs", dynamic=True)
def _compiled_run(
    hidden_states, position_ids, attention_mask,
    input_layernorm_weight, q_proj_weight, k_proj_weight, v_proj_weight,
    q_norm_weight, k_norm_weight, o_proj_weight, post_attention_layernorm_weight,
    gate_proj_weight, up_proj_weight, down_proj_weight, inv_freq,
    rms_norm_eps, attention_scale,
):
    batch_size, seq_len, hidden_size = hidden_states.shape
    num_attention_heads = 40
    num_key_value_heads = 8
    head_dim = 128

    def rms_norm(x, weight):
        input_dtype = x.dtype
        x = x.to(torch.float32)
        variance = x.pow(2).mean(-1, keepdim=True)
        x = x * torch.rsqrt(variance + rms_norm_eps)
        return (weight * x).to(input_dtype)

    residual = hidden_states
    hidden_states = rms_norm(hidden_states, input_layernorm_weight)
    q = F.linear(hidden_states, q_proj_weight)
    k = F.linear(hidden_states, k_proj_weight)
    v = F.linear(hidden_states, v_proj_weight)
    q = q.view(batch_size, seq_len, num_attention_heads, head_dim)
    k = k.view(batch_size, seq_len, num_key_value_heads, head_dim)
    v = v.view(batch_size, seq_len, num_key_value_heads, head_dim)
    q = rms_norm(q, q_norm_weight)
    k = rms_norm(k, k_norm_weight)
    q = q.transpose(1, 2)
    k = k.transpose(1, 2)
    v = v.transpose(1, 2)
    inv_freq_expanded = inv_freq[None, :, None].float().expand(batch_size, -1, 1)
    position_ids_expanded = position_ids[:, None, :].float()
    freqs = (inv_freq_expanded @ position_ids_expanded).transpose(1, 2)
    emb = torch.cat((freqs, freqs), dim=-1)
    cos = emb.cos().to(torch.bfloat16)
    sin = emb.sin().to(torch.bfloat16)
    cos = cos.unsqueeze(1)
    sin = sin.unsqueeze(1)
    q1, q2 = q[..., :head_dim // 2], q[..., head_dim // 2:]
    k1, k2 = k[..., :head_dim // 2], k[..., head_dim // 2:]
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
    gate = F.linear(hidden_states, gate_proj_weight)
    up = F.linear(hidden_states, up_proj_weight)
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
    return _compiled_run(
        hidden_states, position_ids, attention_mask,
        input_layernorm_weight, q_proj_weight, k_proj_weight, v_proj_weight,
        q_norm_weight, k_norm_weight, o_proj_weight, post_attention_layernorm_weight,
        gate_proj_weight, up_proj_weight, down_proj_weight, inv_freq,
        rms_norm_eps, attention_scale,
    )

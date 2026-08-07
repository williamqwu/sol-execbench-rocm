import torch
import torch.nn.functional as F

_compiled = None

def _run(
    hidden_states: torch.Tensor,
    input_layernorm_weight: torch.Tensor,
    q_proj_weight: torch.Tensor,
    q_proj_bias: torch.Tensor,
    k_proj_weight: torch.Tensor,
    k_proj_bias: torch.Tensor,
    v_proj_weight: torch.Tensor,
    v_proj_bias: torch.Tensor,
    o_proj_weight: torch.Tensor,
    rope_cos: torch.Tensor,
    rope_sin: torch.Tensor,
    post_attention_layernorm_weight: torch.Tensor,
    gate_proj_weight: torch.Tensor,
    up_proj_weight: torch.Tensor,
    down_proj_weight: torch.Tensor,
    rms_norm_eps: float,
):
    hidden_size = 3584
    num_heads = 28
    num_kv_heads = 4
    head_dim = 128
    num_kv_groups = num_heads // num_kv_heads
    mrope_section = [16, 24, 24]
    scaling = head_dim ** -0.5

    batch_size, seq_len, _ = hidden_states.shape

    residual = hidden_states

    hidden_fp32 = hidden_states.to(torch.float32)
    variance = hidden_fp32.pow(2).mean(-1, keepdim=True)
    hidden_normed = hidden_fp32 * torch.rsqrt(variance + rms_norm_eps)
    hidden_states = (input_layernorm_weight * hidden_normed).to(hidden_states.dtype)

    query_states = F.linear(hidden_states, q_proj_weight, q_proj_bias)
    key_states = F.linear(hidden_states, k_proj_weight, k_proj_bias)
    value_states = F.linear(hidden_states, v_proj_weight, v_proj_bias)

    query_states = query_states.view(batch_size, seq_len, num_heads, head_dim).transpose(1, 2)
    key_states = key_states.view(batch_size, seq_len, num_kv_heads, head_dim).transpose(1, 2)
    value_states = value_states.view(batch_size, seq_len, num_kv_heads, head_dim).transpose(1, 2)

    mrope_section_doubled = [s * 2 for s in mrope_section]

    cos_parts = []
    sin_parts = []
    for i, section_size in enumerate(mrope_section_doubled):
        cos_i = rope_cos[i % 3]
        sin_i = rope_sin[i % 3]
        start_idx = sum(mrope_section_doubled[:i])
        end_idx = start_idx + section_size
        cos_parts.append(cos_i[..., start_idx:end_idx])
        sin_parts.append(sin_i[..., start_idx:end_idx])

    cos_combined = torch.cat(cos_parts, dim=-1)[:, :seq_len, :].unsqueeze(1)
    sin_combined = torch.cat(sin_parts, dim=-1)[:, :seq_len, :].unsqueeze(1)

    def rotate_half(x):
        x1 = x[..., : x.shape[-1] // 2]
        x2 = x[..., x.shape[-1] // 2 :]
        return torch.cat((-x2, x1), dim=-1)

    query_states = (query_states * cos_combined) + (rotate_half(query_states) * sin_combined)
    key_states = (key_states * cos_combined) + (rotate_half(key_states) * sin_combined)

    key_states = key_states[:, :, None, :, :].expand(
        batch_size, num_kv_heads, num_kv_groups, seq_len, head_dim
    ).reshape(batch_size, num_heads, seq_len, head_dim)
    value_states = value_states[:, :, None, :, :].expand(
        batch_size, num_kv_heads, num_kv_groups, seq_len, head_dim
    ).reshape(batch_size, num_heads, seq_len, head_dim)

    attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) * scaling

    causal_mask = torch.triu(
        torch.full((seq_len, seq_len), float('-inf'), device=hidden_states.device, dtype=hidden_states.dtype),
        diagonal=1
    )
    attn_weights = attn_weights + causal_mask

    attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)

    attn_output = torch.matmul(attn_weights, value_states)
    attn_output = attn_output.transpose(1, 2).contiguous()
    attn_output = attn_output.reshape(batch_size, seq_len, hidden_size)
    attn_output = F.linear(attn_output, o_proj_weight)

    hidden_states = residual + attn_output

    residual = hidden_states

    hidden_fp32 = hidden_states.to(torch.float32)
    variance = hidden_fp32.pow(2).mean(-1, keepdim=True)
    hidden_normed = hidden_fp32 * torch.rsqrt(variance + rms_norm_eps)
    hidden_states = (post_attention_layernorm_weight * hidden_normed).to(hidden_states.dtype)

    gate_output = F.linear(hidden_states, gate_proj_weight)
    up_output = F.linear(hidden_states, up_proj_weight)
    intermediate = F.silu(gate_output) * up_output
    hidden_states = F.linear(intermediate, down_proj_weight)

    output = residual + hidden_states
    return output


@torch.no_grad()
def run(
    hidden_states: torch.Tensor,
    input_layernorm_weight: torch.Tensor,
    q_proj_weight: torch.Tensor,
    q_proj_bias: torch.Tensor,
    k_proj_weight: torch.Tensor,
    k_proj_bias: torch.Tensor,
    v_proj_weight: torch.Tensor,
    v_proj_bias: torch.Tensor,
    o_proj_weight: torch.Tensor,
    rope_cos: torch.Tensor,
    rope_sin: torch.Tensor,
    post_attention_layernorm_weight: torch.Tensor,
    gate_proj_weight: torch.Tensor,
    up_proj_weight: torch.Tensor,
    down_proj_weight: torch.Tensor,
    rms_norm_eps: float,
):
    global _compiled
    if _compiled is None:
        _compiled = torch.compile(_run, mode="reduce-overhead")
    return _compiled(
        hidden_states, input_layernorm_weight, q_proj_weight, q_proj_bias,
        k_proj_weight, k_proj_bias, v_proj_weight, v_proj_bias, o_proj_weight,
        rope_cos, rope_sin, post_attention_layernorm_weight,
        gate_proj_weight, up_proj_weight, down_proj_weight, rms_norm_eps,
    )

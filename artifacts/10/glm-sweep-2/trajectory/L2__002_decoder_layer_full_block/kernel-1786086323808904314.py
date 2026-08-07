import torch
import torch.nn.functional as F

@torch.no_grad()
def run(
    hidden_states: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    attention_mask: torch.Tensor,
    input_layernorm_weight: torch.Tensor,
    q_proj_weight: torch.Tensor,
    k_proj_weight: torch.Tensor,
    v_proj_weight: torch.Tensor,
    o_proj_weight: torch.Tensor,
    post_attention_layernorm_weight: torch.Tensor,
    gate_proj_weight: torch.Tensor,
    up_proj_weight: torch.Tensor,
    down_proj_weight: torch.Tensor,
    rms_norm_eps: float,
):
    num_attention_heads = 32
    num_key_value_heads = 8
    head_dim = 128

    batch_size, seq_len, _ = hidden_states.shape
    residual = hidden_states

    # === ATTENTION BLOCK ===
    # 1. Input RMSNorm
    x = hidden_states.to(torch.float32)
    variance = x.pow(2).mean(-1, keepdim=True)
    x = x * torch.rsqrt(variance + rms_norm_eps)
    hidden_states = input_layernorm_weight * x.to(hidden_states.dtype)

    # 2. QKV projections
    query_states = F.linear(hidden_states, q_proj_weight).view(
        batch_size, seq_len, num_attention_heads, head_dim).transpose(1, 2)
    key_states = F.linear(hidden_states, k_proj_weight).view(
        batch_size, seq_len, num_key_value_heads, head_dim).transpose(1, 2)
    value_states = F.linear(hidden_states, v_proj_weight).view(
        batch_size, seq_len, num_key_value_heads, head_dim).transpose(1, 2)

    # 3. Apply RoPE
    cos_expanded = cos.unsqueeze(1)
    sin_expanded = sin.unsqueeze(1)
    half = head_dim // 2
    q1 = query_states[..., :half]
    q2 = query_states[..., half:]
    query_states = (query_states * cos_expanded) + (torch.cat((-q2, q1), dim=-1) * sin_expanded)
    k1 = key_states[..., :half]
    k2 = key_states[..., half:]
    key_states = (key_states * cos_expanded) + (torch.cat((-k2, k1), dim=-1) * sin_expanded)

    # 4. Attention via SDPA (GQA + causal, no mask materialization)
    attn_output = F.scaled_dot_product_attention(
        query_states, key_states, value_states, is_causal=True, enable_gqa=True)
    attn_output = attn_output.transpose(1, 2).contiguous().view(
        batch_size, seq_len, num_attention_heads * head_dim)

    # 5. Output projection and residual
    attn_output = F.linear(attn_output, o_proj_weight)
    hidden_states = residual + attn_output

    # === MLP BLOCK ===
    residual = hidden_states

    # 6. Post-attention RMSNorm
    x = hidden_states.to(torch.float32)
    variance = x.pow(2).mean(-1, keepdim=True)
    x = x * torch.rsqrt(variance + rms_norm_eps)
    hidden_states = post_attention_layernorm_weight * x.to(hidden_states.dtype)

    # 7. SwiGLU MLP
    gate = F.silu(F.linear(hidden_states, gate_proj_weight))
    up = F.linear(hidden_states, up_proj_weight)
    mlp_output = F.linear(gate * up, down_proj_weight)

    # 8. MLP residual
    output = residual + mlp_output

    return output

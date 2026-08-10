import torch
import torch.nn.functional as F

@torch.compile
def _norm_rope_one(states, norm_weight, cos, sin, eps):
    x32 = states.float()
    x = (x32 * torch.rsqrt(x32.square().mean(-1, keepdim=True) + eps) * norm_weight).to(states.dtype)
    x1, x2 = x[..., :64], x[..., 64:]
    c1, c2 = cos[..., :64], cos[..., 64:]
    s1, s2 = sin[..., :64], sin[..., 64:]
    return torch.cat((x1 * c1 - x2 * s1, x2 * c2 + x1 * s2), dim=-1)

@torch.no_grad()
def run(
    hidden_states: torch.Tensor,
    q_proj_weight: torch.Tensor,
    q_proj_bias: torch.Tensor,
    k_proj_weight: torch.Tensor,
    k_proj_bias: torch.Tensor,
    v_proj_weight: torch.Tensor,
    v_proj_bias: torch.Tensor,
    o_proj_weight: torch.Tensor,
    q_norm_weight: torch.Tensor,
    k_norm_weight: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    rms_norm_eps: float,
):
    batch_size, seq_length, _ = hidden_states.shape
    num_attention_heads = 96
    num_key_value_heads = 8
    head_dim = 128
    num_key_value_groups = 12
    scaling = head_dim ** -0.5
    
    query_states = F.linear(hidden_states, q_proj_weight, q_proj_bias)
    key_states = F.linear(hidden_states, k_proj_weight, k_proj_bias)
    value_states = F.linear(hidden_states, v_proj_weight, v_proj_bias)

    query_states = query_states.view(batch_size, seq_length, num_attention_heads, head_dim)
    key_states = key_states.view(batch_size, seq_length, num_key_value_heads, head_dim)
    value_states = value_states.view(batch_size, seq_length, num_key_value_heads, head_dim)

    rope_cos, rope_sin = cos.unsqueeze(2), sin.unsqueeze(2)
    query_states = _norm_rope_one(query_states, q_norm_weight, rope_cos, rope_sin, rms_norm_eps)
    key_states = _norm_rope_one(key_states, k_norm_weight, rope_cos, rope_sin, rms_norm_eps)
    query_states = query_states.transpose(1, 2)
    key_states = key_states.transpose(1, 2)
    value_states = value_states.transpose(1, 2)
    
    # Fused causal grouped-query attention avoids materializing repeated K/V,
    # the quadratic score tensor, and the causal mask.
    attn_output = F.scaled_dot_product_attention(
        query_states, key_states, value_states,
        is_causal=True, scale=scaling, enable_gqa=True,
    )
    
    attn_output = attn_output.transpose(1, 2).contiguous()
    attn_output = attn_output.reshape(batch_size, seq_length, num_attention_heads * head_dim)
    return F.linear(attn_output, o_proj_weight, None)

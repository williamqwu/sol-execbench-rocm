import torch
import torch.nn.functional as F

@torch.compile
def _project_norm_rope(hidden_states, q_proj_weight, q_proj_bias,
                       k_proj_weight, k_proj_bias, v_proj_weight, v_proj_bias,
                       q_norm_weight, k_norm_weight, cos, sin, eps):
    query_states = F.linear(hidden_states, q_proj_weight, q_proj_bias).view(*hidden_states.shape[:2], 96, 128)
    key_states = F.linear(hidden_states, k_proj_weight, k_proj_bias).view(*hidden_states.shape[:2], 8, 128)
    value_states = F.linear(hidden_states, v_proj_weight, v_proj_bias).view(*hidden_states.shape[:2], 8, 128)
    q32 = query_states.float()
    k32 = key_states.float()
    q = (q32 * torch.rsqrt(q32.square().mean(-1, keepdim=True) + eps) * q_norm_weight).to(query_states.dtype)
    k = (k32 * torch.rsqrt(k32.square().mean(-1, keepdim=True) + eps) * k_norm_weight).to(key_states.dtype)
    q1, q2 = q[..., :64], q[..., 64:]
    k1, k2 = k[..., :64], k[..., 64:]
    c1, c2 = cos[..., :64], cos[..., 64:]
    s1, s2 = sin[..., :64], sin[..., 64:]
    q = torch.cat((q1 * c1 - q2 * s1, q2 * c2 + q1 * s2), dim=-1)
    k = torch.cat((k1 * c1 - k2 * s1, k2 * c2 + k1 * s2), dim=-1)
    return q, k, value_states

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
    
    query_states, key_states, value_states = _project_norm_rope(
        hidden_states, q_proj_weight, q_proj_bias, k_proj_weight, k_proj_bias,
        v_proj_weight, v_proj_bias, q_norm_weight, k_norm_weight,
        cos.unsqueeze(2), sin.unsqueeze(2), rms_norm_eps,
    )
    query_states = query_states.transpose(1, 2)
    key_states = key_states.transpose(1, 2)
    value_states = value_states.transpose(1, 2)
    
    # Fused causal grouped-query attention avoids materializing repeated K/V,
    # the quadratic score tensor, and the causal mask.
    attn_output = F.scaled_dot_product_attention(
        query_states, key_states, value_states,
        is_causal=True, scale=scaling, enable_gqa=True,
    )
    
    # Transpose and reshape
    attn_output = attn_output.transpose(1, 2).contiguous()
    attn_output = attn_output.reshape(batch_size, seq_length, num_attention_heads * head_dim)
    
    # Output projection (no bias)
    output = F.linear(attn_output, o_proj_weight, None)
    
    return output

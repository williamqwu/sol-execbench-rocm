import torch
import torch.nn.functional as F


def rms_norm(x, weight, eps):
    """RMSNorm with (1 + weight) scaling as used in Gemma3."""
    x_float = x.float()
    variance = x_float.pow(2).mean(-1, keepdim=True)
    x_normed = x_float * torch.rsqrt(variance + eps)
    output = x_normed * (1.0 + weight.float())
    return output.type_as(x)


def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q, k, cos, sin):
    """Applies Rotary Position Embedding to the query and key tensors."""
    cos = cos.unsqueeze(1)
    sin = sin.unsqueeze(1)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


@torch.compile
def norm_and_rope(q, k, q_weight, k_weight, cos, sin, eps):
    qf = q.float()
    kf = k.float()
    q = (qf * torch.rsqrt(qf.square().mean(-1, keepdim=True) + eps)
         * (1.0 + q_weight.float())).to(q.dtype)
    k = (kf * torch.rsqrt(kf.square().mean(-1, keepdim=True) + eps)
         * (1.0 + k_weight.float())).to(k.dtype)
    cos = cos.unsqueeze(1)
    sin = sin.unsqueeze(1)
    q1, q2 = q.chunk(2, dim=-1)
    k1, k2 = k.chunk(2, dim=-1)
    q = torch.cat((q1 * cos[..., :64] - q2 * sin[..., :64],
                   q2 * cos[..., 64:] + q1 * sin[..., 64:]), dim=-1)
    k = torch.cat((k1 * cos[..., :64] - k2 * sin[..., :64],
                   k2 * cos[..., 64:] + k1 * sin[..., 64:]), dim=-1)
    return q, k


def repeat_kv(hidden_states, n_rep):
    """Repeats key/value heads for grouped-query attention."""
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)


@torch.no_grad()
def run(
    hidden_states: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    attention_mask: torch.Tensor,
    q_proj_weight: torch.Tensor,
    k_proj_weight: torch.Tensor,
    v_proj_weight: torch.Tensor,
    o_proj_weight: torch.Tensor,
    q_norm_weight: torch.Tensor,
    k_norm_weight: torch.Tensor,
    attn_logit_softcapping: float,
    rms_norm_eps: float,
):
    batch_size, seq_len, hidden_size = hidden_states.shape
    num_attention_heads = 24
    num_key_value_heads = 8
    head_dim = 128
    num_key_value_groups = num_attention_heads // num_key_value_heads
    scaling = head_dim ** -0.5
    
    # Project to Q, K, V using F.linear
    query_states = F.linear(hidden_states, q_proj_weight)
    query_states = query_states.view(batch_size, seq_len, num_attention_heads, head_dim).transpose(1, 2)
    
    key_states = F.linear(hidden_states, k_proj_weight)
    key_states = key_states.view(batch_size, seq_len, num_key_value_heads, head_dim).transpose(1, 2)
    
    value_states = F.linear(hidden_states, v_proj_weight)
    value_states = value_states.view(batch_size, seq_len, num_key_value_heads, head_dim).transpose(1, 2)
    
    # Apply Q/K normalization (unique to Gemma3)
    query_states, key_states = norm_and_rope(
        query_states, key_states, q_norm_weight, k_norm_weight,
        cos, sin, rms_norm_eps)
    
    # Repeat K/V for GQA (8 KV heads -> 24 query heads)
    key_states = repeat_kv(key_states, num_key_value_groups)
    value_states = repeat_kv(value_states, num_key_value_groups)
    
    # Compute attention scores with scaling
    attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) * scaling
    
    # Apply attention logit softcapping (unique to Gemma3)
    attn_weights = attn_weights / attn_logit_softcapping
    attn_weights = torch.tanh(attn_weights)
    attn_weights = attn_weights * attn_logit_softcapping
    
    # Apply causal mask
    causal_mask = attention_mask[:, :, :, :key_states.shape[-2]]
    attn_weights = attn_weights + causal_mask
    
    # Softmax
    attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
    
    # Compute attention output
    attn_output = torch.matmul(attn_weights, value_states)
    attn_output = attn_output.transpose(1, 2).contiguous()
    attn_output = attn_output.reshape(batch_size, seq_len, num_attention_heads * head_dim)
    
    # Output projection
    attn_output = F.linear(attn_output, o_proj_weight)
    
    return attn_output

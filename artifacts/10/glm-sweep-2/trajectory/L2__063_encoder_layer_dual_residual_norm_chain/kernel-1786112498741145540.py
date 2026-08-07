import torch
import torch.nn.functional as F

@torch.no_grad()
def run(
    hidden_states: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    pre_sa_norm_weight: torch.Tensor,
    q_proj_weight: torch.Tensor,
    k_proj_weight: torch.Tensor,
    v_proj_weight: torch.Tensor,
    o_proj_weight: torch.Tensor,
    post_sa_norm_weight: torch.Tensor,
    gate_up_proj_weight: torch.Tensor,
    down_proj_weight: torch.Tensor,
    norm_eps: float,
):
    num_heads = 16
    num_key_value_heads = 4
    head_dim = 64

    batch_size, seq_len, hidden_size = hidden_states.shape

    inv_std = hidden_size ** -0.5

    # ---- RMSNorm helper (fused-ish, fp32 reduction) ----
    def rms_norm(x, weight, eps):
        input_dtype = x.dtype
        x = x.to(torch.float32)
        variance = x.pow(2).mean(-1, keepdim=True)
        x = x * torch.rsqrt(variance + eps)
        return (weight * x).to(input_dtype)

    # ============ First Residual-Norm Pattern: Self-Attention ============
    residual_1 = hidden_states
    normed_1 = rms_norm(residual_1, pre_sa_norm_weight, norm_eps)

    # Q/K/V projections
    query_states = torch.matmul(normed_1, q_proj_weight.t())
    key_states = torch.matmul(normed_1, k_proj_weight.t())
    value_states = torch.matmul(normed_1, v_proj_weight.t())

    # Reshape: (batch, heads, seq, head_dim)
    query_states = query_states.view(batch_size, seq_len, num_heads, head_dim).transpose(1, 2)
    key_states = key_states.view(batch_size, seq_len, num_key_value_heads, head_dim).transpose(1, 2)
    value_states = value_states.view(batch_size, seq_len, num_key_value_heads, head_dim).transpose(1, 2)

    # RoPE
    cos_expanded = cos.unsqueeze(1)  # (batch, 1, seq, head_dim)
    sin_expanded = sin.unsqueeze(1)
    # rotate_half via split-and-cat
    q1 = query_states[..., :head_dim // 2]
    q2 = query_states[..., head_dim // 2:]
    q_embed = (query_states * cos_expanded) + (torch.cat((-q2, q1), dim=-1) * sin_expanded)
    k1 = key_states[..., :head_dim // 2]
    k2 = key_states[..., head_dim // 2:]
    k_embed = (key_states * cos_expanded) + (torch.cat((-k2, k1), dim=-1) * sin_expanded)

    # SDPA handles GQA natively via enable_gqa=True (no materialized repeat)
    attn_output = F.scaled_dot_product_attention(
        q_embed, k_embed, value_states, is_causal=False, enable_gqa=True
    )

    # Reshape and output projection
    attn_output = attn_output.transpose(1, 2).reshape(batch_size, seq_len, -1)
    attn_out = torch.matmul(attn_output, o_proj_weight.t())

    hidden_states = residual_1 + attn_out

    # ============ Second Residual-Norm Pattern: MLP ============
    residual_2 = hidden_states
    normed_2 = rms_norm(residual_2, post_sa_norm_weight, norm_eps)

    up_states = torch.matmul(normed_2, gate_up_proj_weight.t())
    gate, up_states = up_states.chunk(2, dim=-1)
    up_states = up_states * F.silu(gate)
    mlp_out = torch.matmul(up_states, down_proj_weight.t())

    output = residual_2 + mlp_out
    return output

import torch
import torch.nn.functional as F

@torch.no_grad()
def run(
    hidden_states: torch.Tensor,
    layer_norm1_weight: torch.Tensor,
    layer_norm1_bias: torch.Tensor,
    q_proj_weight: torch.Tensor,
    q_proj_bias: torch.Tensor,
    k_proj_weight: torch.Tensor,
    k_proj_bias: torch.Tensor,
    v_proj_weight: torch.Tensor,
    v_proj_bias: torch.Tensor,
    out_proj_weight: torch.Tensor,
    out_proj_bias: torch.Tensor,
    layer_norm2_weight: torch.Tensor,
    layer_norm2_bias: torch.Tensor,
    fc1_weight: torch.Tensor,
    fc1_bias: torch.Tensor,
    fc2_weight: torch.Tensor,
    fc2_bias: torch.Tensor,
    layer_norm_eps: float,
):
    B, S, H = hidden_states.shape
    num_heads = 12
    head_dim = H // num_heads
    scale = head_dim ** -0.5

    # ===== Attention Block =====
    residual = hidden_states
    n1 = F.layer_norm(hidden_states, (H,), layer_norm1_weight, layer_norm1_bias, layer_norm_eps)

    # Fused QKV projection: one GEMM for [B,S,H] x [H, 3H]
    qkv_w = torch.cat([q_proj_weight, k_proj_weight, v_proj_weight], dim=0)
    qkv_b = torch.cat([q_proj_bias, k_proj_bias, v_proj_bias], dim=0)
    qkv = torch.matmul(n1, qkv_w.t()) + qkv_b
    q, k, v = qkv.split(H, dim=-1)

    q = q.view(B, S, num_heads, head_dim).transpose(1, 2)
    k = k.view(B, S, num_heads, head_dim).transpose(1, 2)
    v = v.view(B, S, num_heads, head_dim).transpose(1, 2)

    attn_output = F.scaled_dot_product_attention(q, k, v, scale=scale)
    attn_output = attn_output.transpose(1, 2).reshape(B, S, H)

    attn_output = torch.matmul(attn_output, out_proj_weight.t()) + out_proj_bias
    hidden_states = residual + attn_output

    # ===== MLP Block =====
    residual = hidden_states
    n2 = F.layer_norm(hidden_states, (H,), layer_norm2_weight, layer_norm2_bias, layer_norm_eps)

    h = torch.matmul(n2, fc1_weight.t()) + fc1_bias
    h = h * torch.sigmoid(1.702 * h)
    h = torch.matmul(h, fc2_weight.t()) + fc2_bias

    output = residual + h
    return output

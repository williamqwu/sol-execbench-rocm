import torch
import torch.nn.functional as F
from typing import Tuple


@torch.compile(dynamic=True)
def _fused_norm_rope_backward(
    grad_output, x_pre_norm, weight, rstd, cos, sin, D
):
    """Fused RMS-norm backward + RoPE backward.

    RMS norm backward done in fp32, result cast to bf16.
    RoPE backward done in bf16 (matching reference).
    """
    go_f = grad_output.to(torch.float32)
    x_f = x_pre_norm.to(torch.float32)
    w_f = weight.to(torch.float32)
    r_f = rstd.to(torch.float32)

    # --- RMS norm backward (fp32) ---
    x_normed = x_f * r_f
    grad_weight = (go_f * x_normed).sum(dim=(0, 1, 2))

    grad_x_direct = go_f * w_f * r_f
    grad_rstd = (go_f * w_f * x_f).sum(dim=-1, keepdim=True)
    grad_x_from_rstd = grad_rstd * (-r_f.pow(3) * x_f / D)
    grad_pn_f = grad_x_direct + grad_x_from_rstd

    # Cast to bf16 to match reference RoPE backward precision
    grad_pn = grad_pn_f.to(grad_output.dtype)
    x_bf = x_pre_norm  # already bf16

    # --- RoPE backward (bf16) ---
    half_dim = D // 2
    g1 = grad_pn[..., :half_dim]
    g2 = grad_pn[..., half_dim:]
    grad_rotated_inv = torch.cat((g2, -g1), dim=-1)
    grad_pre_rope = grad_pn * cos + grad_rotated_inv * sin

    xr1 = x_bf[..., :half_dim]
    xr2 = x_bf[..., half_dim:]
    xr_inv = torch.cat((-xr2, xr1), dim=-1)
    x_original = x_bf * cos + xr_inv * sin

    grad_cos = (grad_pn * x_original).sum(dim=1)

    xo1 = x_original[..., :half_dim]
    xo2 = x_original[..., half_dim:]
    xo_rotated = torch.cat((-xo2, xo1), dim=-1)
    grad_sin = (grad_pn * xo_rotated).sum(dim=1)

    return grad_pre_rope, grad_weight, grad_cos, grad_sin


@torch.no_grad()
def run(
    grad_query: torch.Tensor,
    grad_key: torch.Tensor,
    grad_value: torch.Tensor,
    hidden_states: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    qkv_weight: torch.Tensor,
    q_norm_weight: torch.Tensor,
    k_norm_weight: torch.Tensor,
    query_pre_norm: torch.Tensor,
    key_pre_norm: torch.Tensor,
    q_rstd: torch.Tensor,
    k_rstd: torch.Tensor,
    eps: float
):
    num_heads = 32
    num_kv_heads = 8
    head_dim = 128

    bsz, seq_len, hidden_size = hidden_states.shape
    qkv_size = num_heads * head_dim + 2 * num_kv_heads * head_dim
    D = head_dim

    cos_expanded = cos.unsqueeze(1)
    sin_expanded = sin.unsqueeze(1)

    grad_qpr, grad_qnw, grad_cos_q, grad_sin_q = _fused_norm_rope_backward(
        grad_query, query_pre_norm, q_norm_weight, q_rstd,
        cos_expanded, sin_expanded, D)

    grad_kpr, grad_knw, grad_cos_k, grad_sin_k = _fused_norm_rope_backward(
        grad_key, key_pre_norm, k_norm_weight, k_rstd,
        cos_expanded, sin_expanded, D)

    grad_cos = grad_cos_q + grad_cos_k
    grad_sin = grad_sin_q + grad_sin_k

    grad_q_flat = grad_qpr.transpose(1, 2).reshape(bsz, seq_len, num_heads * head_dim)
    grad_k_flat = grad_kpr.transpose(1, 2).reshape(bsz, seq_len, num_kv_heads * head_dim)
    grad_v_flat = grad_value.transpose(1, 2).reshape(bsz, seq_len, num_kv_heads * head_dim)
    grad_qkv_states = torch.cat([grad_q_flat, grad_k_flat, grad_v_flat], dim=-1)

    grad_qkv_flat = grad_qkv_states.reshape(-1, qkv_size)
    hidden_flat = hidden_states.reshape(-1, hidden_size)

    grad_hidden_states = F.linear(grad_qkv_flat, qkv_weight.t())
    grad_hidden_states = grad_hidden_states.reshape(bsz, seq_len, hidden_size)
    grad_qkv_weight = torch.matmul(grad_qkv_flat.t(), hidden_flat)

    return (
        grad_hidden_states.to(torch.bfloat16),
        grad_cos.to(torch.bfloat16),
        grad_sin.to(torch.bfloat16),
        grad_qkv_weight.to(torch.bfloat16),
        grad_qnw.to(torch.bfloat16),
        grad_knw.to(torch.bfloat16)
    )

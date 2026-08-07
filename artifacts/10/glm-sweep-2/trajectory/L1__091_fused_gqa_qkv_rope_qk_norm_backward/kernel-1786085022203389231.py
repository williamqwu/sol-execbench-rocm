import torch
import torch.nn.functional as F
from typing import Tuple


@torch.compile(dynamic=True, mode="max-autotune-no-cudagraphs")
def _elementwise_prep(
    grad_query: torch.Tensor,
    grad_key: torch.Tensor,
    grad_value: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    q_norm_weight: torch.Tensor,
    k_norm_weight: torch.Tensor,
    query_pre_norm: torch.Tensor,
    key_pre_norm: torch.Tensor,
    q_rstd: torch.Tensor,
    k_rstd: torch.Tensor,
):
    """Fused RMS norm backward + RoPE backward + reshape. Produces grad_qkv_states."""
    num_heads = 32
    num_kv_heads = 8
    head_dim = 128
    bsz, seq_len, hidden_size = grad_query.shape[0], grad_query.shape[2], 4096

    def rms_norm_bwd(grad_output, x, weight, rstd):
        gf = grad_output.float()
        xf = x.float()
        wf = weight.float()
        rf = rstd.float()
        x_normed = xf * rf
        grad_weight = (gf * x_normed).sum(dim=(0, 1, 2))
        gw_rf = wf * rf
        grad_x_direct = gf * gw_rf
        D = x.shape[-1]
        grad_rstd = (gf * wf * xf).sum(dim=-1, keepdim=True)
        grad_x = grad_x_direct + grad_rstd * (-rf.pow(3) * xf / D)
        return grad_x.to(grad_output.dtype), grad_weight.to(grad_output.dtype)

    gq_pre_norm, gq_norm_w = rms_norm_bwd(grad_query, query_pre_norm, q_norm_weight, q_rstd)
    gk_pre_norm, gk_norm_w = rms_norm_bwd(grad_key, key_pre_norm, k_norm_weight, k_rstd)

    def rope_bwd(grad_output, x_rotated, cos, sin):
        half_dim = grad_output.shape[-1] // 2
        g1 = grad_output[..., :half_dim]
        g2 = grad_output[..., half_dim:]
        grad_rotated_inv = torch.cat((g2, -g1), dim=-1)
        grad_x = grad_output * cos + grad_rotated_inv * sin

        xr1 = x_rotated[..., :half_dim]
        xr2 = x_rotated[..., half_dim:]
        x_rotated_inv = torch.cat((-xr2, xr1), dim=-1)
        x_original = x_rotated * cos + x_rotated_inv * sin

        grad_cos = (grad_output * x_original).sum(dim=1)
        xo1 = x_original[..., :half_dim]
        xo2 = x_original[..., half_dim:]
        x_original_rotated = torch.cat((-xo2, xo1), dim=-1)
        grad_sin = (grad_output * x_original_rotated).sum(dim=1)
        return grad_x, grad_cos, grad_sin

    cos_e = cos.unsqueeze(1)
    sin_e = sin.unsqueeze(1)

    gq_pre_rope, gcos_q, gsin_q = rope_bwd(gq_pre_norm, query_pre_norm, cos_e, sin_e)
    gk_pre_rope, gcos_k, gsin_k = rope_bwd(gk_pre_norm, key_pre_norm, cos_e, sin_e)

    grad_cos = (gcos_q + gcos_k).to(torch.bfloat16)
    grad_sin = (gsin_q + gsin_k).to(torch.bfloat16)

    gq_flat = gq_pre_rope.transpose(1, 2).reshape(bsz, seq_len, num_heads * head_dim)
    gk_flat = gk_pre_rope.transpose(1, 2).reshape(bsz, seq_len, num_kv_heads * head_dim)
    gv_flat = grad_value.transpose(1, 2).reshape(bsz, seq_len, num_kv_heads * head_dim)
    grad_qkv_states = torch.cat([gq_flat, gk_flat, gv_flat], dim=-1)

    return grad_qkv_states, grad_cos, grad_sin, gq_norm_w, gk_norm_w


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
    eps: float,
):
    num_heads = 32
    num_kv_heads = 8
    head_dim = 128
    bsz, seq_len, hidden_size = hidden_states.shape
    qkv_size = num_heads * head_dim + 2 * num_kv_heads * head_dim

    grad_qkv_states, grad_cos, grad_sin, gq_norm_w, gk_norm_w = _elementwise_prep(
        grad_query, grad_key, grad_value, cos, sin,
        q_norm_weight, k_norm_weight,
        query_pre_norm, key_pre_norm, q_rstd, k_rstd,
    )

    grad_qkv_flat = grad_qkv_states.reshape(-1, qkv_size)
    hidden_flat = hidden_states.reshape(-1, hidden_size)

    # Eager GEMMs (already well-tuned)
    grad_hidden_states = torch.mm(grad_qkv_flat, qkv_weight).reshape(bsz, seq_len, hidden_size)
    grad_qkv_weight = torch.mm(grad_qkv_flat.t(), hidden_flat)

    return (
        grad_hidden_states.to(torch.bfloat16),
        grad_cos,
        grad_sin,
        grad_qkv_weight.to(torch.bfloat16),
        gq_norm_w.to(torch.bfloat16),
        gk_norm_w.to(torch.bfloat16),
    )

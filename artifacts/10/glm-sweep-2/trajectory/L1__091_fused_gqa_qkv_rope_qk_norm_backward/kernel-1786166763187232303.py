import torch
import torch.nn.functional as F
from typing import Tuple


def _rms_norm_backward_core(grad_output_f, x_f, weight_f, rstd_f, D):
    x_normed = x_f * rstd_f
    grad_weight = (grad_output_f * x_normed).sum(dim=(0, 1, 2))
    grad_x_direct = grad_output_f * weight_f * rstd_f
    grad_rstd = (grad_output_f * weight_f * x_f).sum(dim=-1, keepdim=True)
    grad_x_from_rstd = grad_rstd * (-rstd_f.pow(3) * x_f / D)
    grad_x = grad_x_direct + grad_x_from_rstd
    return grad_x, grad_weight


def _rope_backward_core(grad_output, x_rotated, cos, sin):
    half_dim = grad_output.shape[-1] // 2
    grad_1 = grad_output[..., :half_dim]
    grad_2 = grad_output[..., half_dim:]
    grad_rotated_inv = torch.cat((grad_2, -grad_1), dim=-1)
    grad_x = grad_output * cos + grad_rotated_inv * sin

    x_rotated_1 = x_rotated[..., :half_dim]
    x_rotated_2 = x_rotated[..., half_dim:]
    x_rotated_inv = torch.cat((-x_rotated_2, x_rotated_1), dim=-1)
    x_original = x_rotated * cos + x_rotated_inv * sin

    grad_cos = (grad_output * x_original).sum(dim=1)

    x_original_1 = x_original[..., :half_dim]
    x_original_2 = x_original[..., half_dim:]
    x_original_rotated = torch.cat((-x_original_2, x_original_1), dim=-1)
    grad_sin = (grad_output * x_original_rotated).sum(dim=1)

    return grad_x, grad_cos, grad_sin


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

    cos_e = cos.unsqueeze(1)
    sin_e = sin.unsqueeze(1)

    grad_q_f = grad_query.to(torch.float32)
    qp_f = query_pre_norm.to(torch.float32)
    qw_f = q_norm_weight.to(torch.float32)
    qr_f = q_rstd.to(torch.float32)
    grad_q_pn_f, grad_qnw = _rms_norm_backward_core(grad_q_f, qp_f, qw_f, qr_f, D)

    grad_k_f = grad_key.to(torch.float32)
    kp_f = key_pre_norm.to(torch.float32)
    kw_f = k_norm_weight.to(torch.float32)
    kr_f = k_rstd.to(torch.float32)
    grad_k_pn_f, grad_knw = _rms_norm_backward_core(grad_k_f, kp_f, kw_f, kr_f, D)

    grad_qpr, grad_cos_q, grad_sin_q = _rope_backward_core(
        grad_q_pn_f, qp_f, cos_e.to(torch.float32), sin_e.to(torch.float32))
    grad_kpr, grad_cos_k, grad_sin_k = _rope_backward_core(
        grad_k_pn_f, kp_f, cos_e.to(torch.float32), sin_e.to(torch.float32))

    grad_cos = grad_cos_q + grad_cos_k
    grad_sin = grad_sin_q + grad_sin_k

    grad_qpr_bf = grad_qpr.to(torch.bfloat16)
    grad_kpr_bf = grad_kpr.to(torch.bfloat16)

    grad_q_flat = grad_qpr_bf.transpose(1, 2).reshape(bsz, seq_len, num_heads * head_dim)
    grad_k_flat = grad_kpr_bf.transpose(1, 2).reshape(bsz, seq_len, num_kv_heads * head_dim)
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

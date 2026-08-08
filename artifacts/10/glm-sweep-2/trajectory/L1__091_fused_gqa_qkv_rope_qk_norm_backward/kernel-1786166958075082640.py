import torch
import torch.nn.functional as F
from typing import Tuple


@torch.compile(dynamic=True)
def _rms_norm_backward_compiled(grad_output, x, weight, rstd, D):
    grad_output_f = grad_output.to(torch.float32)
    x_f = x.to(torch.float32)
    weight_f = weight.to(torch.float32)
    rstd_f = rstd.to(torch.float32)

    x_normed = x_f * rstd_f
    grad_weight = (grad_output_f * x_normed).sum(dim=(0, 1, 2))

    grad_x_direct = grad_output_f * weight_f * rstd_f
    grad_rstd = (grad_output_f * weight_f * x_f).sum(dim=-1, keepdim=True)
    grad_x_from_rstd = grad_rstd * (-rstd_f.pow(3) * x_f / D)
    grad_x = grad_x_direct + grad_x_from_rstd

    return grad_x.to(grad_output.dtype), grad_weight.to(grad_output.dtype)


@torch.compile(dynamic=True)
def _rope_backward_compiled(grad_output, x_rotated, cos, sin):
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

    grad_query_pre_norm, grad_q_norm_weight = _rms_norm_backward_compiled(
        grad_query, query_pre_norm, q_norm_weight, q_rstd, D)

    grad_key_pre_norm, grad_k_norm_weight = _rms_norm_backward_compiled(
        grad_key, key_pre_norm, k_norm_weight, k_rstd, D)

    cos_expanded = cos.unsqueeze(1)
    sin_expanded = sin.unsqueeze(1)

    grad_query_pre_rope, grad_cos_q, grad_sin_q = _rope_backward_compiled(
        grad_query_pre_norm, query_pre_norm, cos_expanded, sin_expanded)

    grad_key_pre_rope, grad_cos_k, grad_sin_k = _rope_backward_compiled(
        grad_key_pre_norm, key_pre_norm, cos_expanded, sin_expanded)

    grad_cos = grad_cos_q + grad_cos_k
    grad_sin = grad_sin_q + grad_sin_k

    grad_query_reshaped = grad_query_pre_rope.transpose(1, 2)
    grad_key_reshaped = grad_key_pre_rope.transpose(1, 2)
    grad_value_reshaped = grad_value.transpose(1, 2)

    grad_query_flat = grad_query_reshaped.reshape(bsz, seq_len, num_heads * head_dim)
    grad_key_flat = grad_key_reshaped.reshape(bsz, seq_len, num_kv_heads * head_dim)
    grad_value_flat = grad_value_reshaped.reshape(bsz, seq_len, num_kv_heads * head_dim)
    grad_qkv_states = torch.cat([grad_query_flat, grad_key_flat, grad_value_flat], dim=-1)

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
        grad_q_norm_weight.to(torch.bfloat16),
        grad_k_norm_weight.to(torch.bfloat16)
    )

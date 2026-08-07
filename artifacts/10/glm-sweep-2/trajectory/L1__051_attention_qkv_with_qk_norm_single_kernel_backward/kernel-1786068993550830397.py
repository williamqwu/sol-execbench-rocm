import torch
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def _norm_bwd_kernel(
    grad_out_ptr,      # [B, H, S, D] bf16
    normed_ptr,        # [B, H, S, D] bf16
    rstd_ptr,          # [B, H, S, 1] f32
    scale_ptr,         # [D] f32
    grad_norm_weight_ptr,  # [D] f32 (atomic add)
    out_ptr,           # [B, S, H*D] bf16  (transposed output)
    B, H, S, D,
    # strides
    sb_grad, sh_grad, ss_grad,
    sb_nm, sh_nm, ss_nm,
    sb_rs, sh_rs, ss_rs,
    sb_out, ss_out,
    BLOCK_D: tl.constexpr,
):
    pid = tl.program_id(0)
    # each program handles one (b, s) row across all heads
    b = pid // S
    s = pid % S
    if b >= B:
        return

    scale = tl.load(scale_ptr + tl.arange(0, BLOCK_D), mask=tl.arange(0, BLOCK_D) < D, other=0.0).to(tl.float32)

    offs_d = tl.arange(0, BLOCK_D)
    d_mask = offs_d < D

    for h in range(H):
        grad_base = b * sb_grad + h * sh_grad + s * ss_grad
        nm_base = b * sb_nm + h * sh_nm + s * ss_nm
        rs_base = b * sb_rs + h * sh_rs + s * ss_rs

        g = tl.load(grad_out_ptr + grad_base + offs_d, mask=d_mask, other=0.0).to(tl.float32)
        nm = tl.load(normed_ptr + nm_base + offs_d, mask=d_mask, other=0.0).to(tl.float32)

        # grad_norm_weight contribution: g * nm
        tl.atomic_add(grad_norm_weight_ptr + offs_d, g * nm, mask=d_mask)

        grad_normed = g * scale
        mean_term = tl.sum(grad_normed * nm, axis=0) / D
        rstd = tl.load(rstd_ptr + rs_base).to(tl.float32)
        grad_t = rstd * (grad_normed - mean_term * nm)

        # write transposed: [B, S, H*D] => position (b, s, h*D + d)
        out_base = b * sb_out + s * ss_out + h * D
        tl.store(out_ptr + out_base + offs_d, grad_t.to(tl.bfloat16), mask=d_mask)


def _norm_backward_triton(grad_out, normed, rstd, scale, num_heads):
    # grad_out: [B, H, S, D], normed: [B, H, S, D], rstd: [B, H, S, 1]
    B, H, S, D = grad_out.shape
    grad_norm_weight = torch.zeros(D, dtype=torch.float32, device=grad_out.device)
    out = torch.empty(B, S, H * D, dtype=torch.bfloat16, device=grad_out.device)

    grid = (B * S,)
    _norm_bwd_kernel[grid](
        grad_out, normed, rstd, scale, grad_norm_weight, out,
        B, H, S, D,
        grad_out.stride(0), grad_out.stride(1), grad_out.stride(2),
        normed.stride(0), normed.stride(1), normed.stride(2),
        rstd.stride(0), rstd.stride(1), rstd.stride(2),
        out.stride(0), out.stride(1),
        BLOCK_D=triton.next_power_of_2(D),
        num_warps=4,
    )
    return grad_norm_weight, out


@torch.compile(dynamic=True, mode="max-autotune-no-cudagraphs")
def _matmul_part(grad_query_proj, grad_key_proj, grad_value_proj,
                 hidden_states, q_weight, k_weight, v_weight):
    num_attention_heads = 4
    num_key_value_heads = 1
    head_dim = 256
    hidden_size = 640

    grad_qkv = torch.cat([grad_query_proj, grad_key_proj, grad_value_proj], dim=-1)
    qkv_weight = torch.cat([q_weight, k_weight, v_weight], dim=0)
    grad_hidden_states = torch.matmul(grad_qkv, qkv_weight)

    grad_qkv_2d = grad_qkv.reshape(-1, grad_qkv.shape[-1])
    hidden_states_2d = hidden_states.reshape(-1, hidden_size)
    grad_qkv_weight = torch.matmul(grad_qkv_2d.t(), hidden_states_2d)
    q_size = num_attention_heads * head_dim
    kv_size = num_key_value_heads * head_dim
    grad_q_weight = grad_qkv_weight[:q_size]
    grad_k_weight = grad_qkv_weight[q_size:q_size + kv_size]
    grad_v_weight = grad_qkv_weight[q_size + kv_size:]
    return grad_hidden_states, grad_q_weight, grad_k_weight, grad_v_weight


@torch.no_grad()
def run(
    grad_query: torch.Tensor,
    grad_key: torch.Tensor,
    grad_value: torch.Tensor,
    hidden_states: torch.Tensor,
    q_weight: torch.Tensor,
    k_weight: torch.Tensor,
    v_weight: torch.Tensor,
    q_norm_weight: torch.Tensor,
    k_norm_weight: torch.Tensor,
    query_transposed: torch.Tensor,
    key_transposed: torch.Tensor,
    q_rstd: torch.Tensor,
    k_rstd: torch.Tensor,
    q_normed: torch.Tensor,
    k_normed: torch.Tensor,
    rms_norm_eps: float,
):
    num_attention_heads = 4
    num_key_value_heads = 1
    head_dim = 256

    q_scale = 1.0 + q_norm_weight.float()
    k_scale = 1.0 + k_norm_weight.float()

    grad_q_norm_weight, grad_query_proj = _norm_backward_triton(
        grad_query, q_normed, q_rstd, q_scale, num_attention_heads)
    grad_k_norm_weight, grad_key_proj = _norm_backward_triton(
        grad_key, k_normed, k_rstd, k_scale, num_key_value_heads)

    grad_value_proj = grad_value.transpose(1, 2).contiguous().view(
        grad_value.shape[0], grad_value.shape[2], num_key_value_heads * head_dim)

    grad_hidden_states, grad_q_weight, grad_k_weight, grad_v_weight = _matmul_part(
        grad_query_proj, grad_key_proj, grad_value_proj,
        hidden_states, q_weight, k_weight, v_weight)

    return (grad_hidden_states, grad_q_weight, grad_k_weight, grad_v_weight,
            grad_q_norm_weight, grad_k_norm_weight)

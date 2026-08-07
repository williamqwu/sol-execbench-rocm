import torch
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def _norm_bwd_transpose_kernel(
    grad_out_ptr, normed_ptr, rstd_ptr, scale_ptr, out_ptr,
    B, H, S, D,
    sb_g, sh_g, ss_g,
    sb_n, sh_n, ss_n,
    sb_r, sh_r, ss_r,
    sb_o, ss_o,
    BLOCK_D: tl.constexpr,
):
    pid = tl.program_id(0)
    b = pid // S
    s = pid % S
    offs_d = tl.arange(0, BLOCK_D)
    d_mask = offs_d < D
    scale = tl.load(scale_ptr + offs_d, mask=d_mask, other=0.0).to(tl.float32)
    for h in range(H):
        gb = b * sb_g + h * sh_g + s * ss_g
        nb = b * sb_n + h * sh_n + s * ss_n
        rb = b * sb_r + h * sh_r + s * ss_r
        g = tl.load(grad_out_ptr + gb + offs_d, mask=d_mask, other=0.0).to(tl.float32)
        nm = tl.load(normed_ptr + nb + offs_d, mask=d_mask, other=0.0).to(tl.float32)
        grad_normed = g * scale
        mean_term = tl.sum(grad_normed * nm, axis=0) / D
        rstd = tl.load(rstd_ptr + rb).to(tl.float32)
        grad_t = rstd * (grad_normed - mean_term * nm)
        ob = b * sb_o + s * ss_o + h * D
        tl.store(out_ptr + ob + offs_d, grad_t.to(tl.bfloat16), mask=d_mask)


def _norm_bwd_transpose(grad_out, normed, rstd, scale):
    B, H, S, D = grad_out.shape
    out = torch.empty(B, S, H * D, dtype=torch.bfloat16, device=grad_out.device)
    grid = (B * S,)
    _norm_bwd_transpose_kernel[grid](
        grad_out, normed, rstd, scale, out,
        B, H, S, D,
        grad_out.stride(0), grad_out.stride(1), grad_out.stride(2),
        normed.stride(0), normed.stride(1), normed.stride(2),
        rstd.stride(0), rstd.stride(1), rstd.stride(2),
        out.stride(0), out.stride(1),
        BLOCK_D=triton.next_power_of_2(D), num_warps=4,
    )
    return out


@torch.compile(dynamic=True, mode="max-autotune-no-cudagraphs")
def _reductions(grad_query, grad_key, q_normed, k_normed):
    grad_q_norm_weight = (grad_query.float() * q_normed.float()).sum(dim=(0, 1, 2))
    grad_k_norm_weight = (grad_key.float() * k_normed.float()).sum(dim=(0, 1, 2))
    return grad_q_norm_weight, grad_k_norm_weight


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
    num_key_value_heads = 1
    head_dim = 256

    q_scale = 1.0 + q_norm_weight.float()
    k_scale = 1.0 + k_norm_weight.float()

    grad_query_proj = _norm_bwd_transpose(grad_query, q_normed, q_rstd, q_scale)
    grad_key_proj = _norm_bwd_transpose(grad_key, k_normed, k_rstd, k_scale)
    grad_value_proj = grad_value.transpose(1, 2).contiguous().view(
        grad_value.shape[0], grad_value.shape[2], num_key_value_heads * head_dim)

    grad_q_norm_weight, grad_k_norm_weight = _reductions(grad_query, grad_key, q_normed, k_normed)

    grad_hidden_states, grad_q_weight, grad_k_weight, grad_v_weight = _matmul_part(
        grad_query_proj, grad_key_proj, grad_value_proj,
        hidden_states, q_weight, k_weight, v_weight)

    return (grad_hidden_states, grad_q_weight, grad_k_weight, grad_v_weight,
            grad_q_norm_weight, grad_k_norm_weight)

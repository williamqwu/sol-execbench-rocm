import torch
import triton
import triton.language as tl


@triton.jit
def _softmax_backward_scale_kernel(
    G_ptr, A_ptr, O_ptr,
    scaling,
    SQ, SK,
    stride_gb, stride_gq, stride_gk,
    stride_ab, stride_aq, stride_ak,
    stride_ob, stride_oq, stride_ok,
    BLOCK_Q: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_b = tl.program_id(1)
    pid_q = tl.program_id(0)
    q_idx = pid_q * BLOCK_Q + tl.arange(0, BLOCK_Q)
    q_mask = q_idx < SQ

    g_base = G_ptr + pid_b * stride_gb
    a_base = A_ptr + pid_b * stride_ab
    o_base = O_ptr + pid_b * stride_ob

    # Pass 1: sum_grad = sum(grad_output * attn_weights) over K
    acc = tl.zeros([BLOCK_Q], dtype=tl.float32)
    for ks in range(0, SK, BLOCK_K):
        k_idx = ks + tl.arange(0, BLOCK_K)
        k_mask = k_idx < SK
        mask = q_mask[:, None] & k_mask[None, :]
        g_ptrs = g_base + q_idx[:, None] * stride_gq + k_idx[None, :] * stride_gk
        a_ptrs = a_base + q_idx[:, None] * stride_aq + k_idx[None, :] * stride_ak
        g = tl.load(g_ptrs, mask=mask, other=0.0).to(tl.float32)
        a = tl.load(a_ptrs, mask=mask, other=0.0).to(tl.float32)
        acc += tl.sum(g * a, axis=1)

    # Pass 2: grad_scaled = attn_weights * (grad_output - sum_grad) * scaling
    for ks in range(0, SK, BLOCK_K):
        k_idx = ks + tl.arange(0, BLOCK_K)
        k_mask = k_idx < SK
        mask = q_mask[:, None] & k_mask[None, :]
        g_ptrs = g_base + q_idx[:, None] * stride_gq + k_idx[None, :] * stride_gk
        a_ptrs = a_base + q_idx[:, None] * stride_aq + k_idx[None, :] * stride_ak
        o_ptrs = o_base + q_idx[:, None] * stride_oq + k_idx[None, :] * stride_ok
        g = tl.load(g_ptrs, mask=mask, other=0.0).to(tl.float32)
        a = tl.load(a_ptrs, mask=mask, other=0.0).to(tl.float32)
        val = a * (g - acc[:, None]) * scaling
        tl.store(o_ptrs, val.to(tl.bfloat16), mask=mask)


@torch.no_grad()
def run(
    grad_output: torch.Tensor,
    query: torch.Tensor,
    key: torch.Tensor,
    attn_weights: torch.Tensor,
    scaling: float,
):
    B, H, SQ, SK = grad_output.shape
    BH = B * H
    g = grad_output.reshape(BH, SQ, SK)
    a = attn_weights.reshape(BH, SQ, SK)
    out = torch.empty_like(g)

    BLOCK_Q = 64
    BLOCK_K = 128
    grid = (triton.cdiv(SQ, BLOCK_Q), BH)
    _softmax_backward_scale_kernel[grid](
        g, a, out, scaling,
        SQ, SK,
        g.stride(0), g.stride(1), g.stride(2),
        a.stride(0), a.stride(1), a.stride(2),
        out.stride(0), out.stride(1), out.stride(2),
        BLOCK_Q=BLOCK_Q, BLOCK_K=BLOCK_K,
        num_warps=4, num_stages=2,
    )
    grad_scaled = out.view(B, H, SQ, SK)

    grad_query = torch.matmul(grad_scaled, key)
    grad_key = torch.matmul(grad_scaled.transpose(-2, -1), query)
    return grad_query, grad_key

import torch
import triton
import triton.language as tl


@triton.jit
def _fused_backward_kernel(
    grad_cos_ptr, grad_sin_ptr, pos_ids_ptr, emb_ptr,
    grad_out_ptr,
    total_patches,
    grad_cos_stride0, grad_cos_stride1,
    pos_ids_stride0, pos_ids_stride1,
    HEAD_DIM: tl.constexpr,
    HEAD_DIM_HALF: tl.constexpr,
    HEAD_DIM_QUARTER: tl.constexpr,
    BLOCK_P: tl.constexpr,
):
    col = tl.program_id(0)
    pb = tl.program_id(1)

    Q = HEAD_DIM_QUARTER
    H = HEAD_DIM_HALF

    c0 = col
    c1 = col + Q
    c0h = col + H
    c1h = col + Q + H

    offs_p = pb * BLOCK_P + tl.arange(0, BLOCK_P)
    mask_p = offs_p < total_patches

    pos0 = tl.load(pos_ids_ptr + offs_p * pos_ids_stride0 + 0 * pos_ids_stride1, mask=mask_p, other=0).to(tl.float32)
    pos1 = tl.load(pos_ids_ptr + offs_p * pos_ids_stride0 + 1 * pos_ids_stride1, mask=mask_p, other=0).to(tl.float32)

    base = offs_p * grad_cos_stride0

    gc0 = tl.load(grad_cos_ptr + base + c0 * grad_cos_stride1, mask=mask_p, other=0.0)
    gs0 = tl.load(grad_sin_ptr + base + c0 * grad_cos_stride1, mask=mask_p, other=0.0)
    e0  = tl.load(emb_ptr      + base + c0 * grad_cos_stride1, mask=mask_p, other=0.0)
    ge0 = -gc0 * tl.sin(e0) + gs0 * tl.cos(e0)

    gc1 = tl.load(grad_cos_ptr + base + c1 * grad_cos_stride1, mask=mask_p, other=0.0)
    gs1 = tl.load(grad_sin_ptr + base + c1 * grad_cos_stride1, mask=mask_p, other=0.0)
    e1  = tl.load(emb_ptr      + base + c1 * grad_cos_stride1, mask=mask_p, other=0.0)
    ge1 = -gc1 * tl.sin(e1) + gs1 * tl.cos(e1)

    gc0h = tl.load(grad_cos_ptr + base + c0h * grad_cos_stride1, mask=mask_p, other=0.0)
    gs0h = tl.load(grad_sin_ptr + base + c0h * grad_cos_stride1, mask=mask_p, other=0.0)
    e0h  = tl.load(emb_ptr      + base + c0h * grad_cos_stride1, mask=mask_p, other=0.0)
    ge0h = -gc0h * tl.sin(e0h) + gs0h * tl.cos(e0h)

    gc1h = tl.load(grad_cos_ptr + base + c1h * grad_cos_stride1, mask=mask_p, other=0.0)
    gs1h = tl.load(grad_sin_ptr + base + c1h * grad_cos_stride1, mask=mask_p, other=0.0)
    e1h  = tl.load(emb_ptr      + base + c1h * grad_cos_stride1, mask=mask_p, other=0.0)
    ge1h = -gc1h * tl.sin(e1h) + gs1h * tl.cos(e1h)

    g0 = ge0 + ge0h
    g1 = ge1 + ge1h

    contrib = pos0 * g0 + pos1 * g1
    partial = tl.sum(contrib)

    tl.atomic_add(grad_out_ptr + col, partial)


@torch.no_grad()
def run(
    grad_cos: torch.Tensor,
    grad_sin: torch.Tensor,
    pos_ids: torch.Tensor,
    inv_freq: torch.Tensor,
    emb: torch.Tensor,
) -> torch.Tensor:
    total_patches = grad_cos.shape[0]
    head_dim = grad_cos.shape[1]
    head_dim_quarter = inv_freq.shape[0]
    head_dim_half = head_dim // 2

    grad_out = torch.zeros(head_dim_quarter, device=grad_cos.device, dtype=torch.float32)

    BLOCK_P = 1024
    num_blocks = triton.cdiv(total_patches, BLOCK_P)
    grid = (head_dim_quarter, num_blocks)
    _fused_backward_kernel[grid](
        grad_cos, grad_sin, pos_ids, emb, grad_out,
        total_patches,
        grad_cos.stride(0), grad_cos.stride(1),
        pos_ids.stride(0), pos_ids.stride(1),
        HEAD_DIM=head_dim,
        HEAD_DIM_HALF=head_dim_half,
        HEAD_DIM_QUARTER=head_dim_quarter,
        BLOCK_P=BLOCK_P,
        num_warps=4,
    )
    return grad_out

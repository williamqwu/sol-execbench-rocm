import torch
import triton
import triton.language as tl

@triton.jit
def _group_norm_kernel(
    x_ptr, w_ptr, b_ptr, out_ptr,
    B, C, H, W, num_groups,
    # number of elements per group = (C//num_groups)*H*W
    group_elems,
    stride_b, stride_c, stride_h, stride_w,
    eps,
    BLOCK: tl.constexpr,
):
    # one program per (batch, group)
    pid = tl.program_id(0)
    b_idx = pid // num_groups
    g_idx = pid % num_groups

    cpg = C // num_groups  # channels per group
    # base pointer for this (b, g): channel g*cpg .. (g+1)*cpg
    # x layout (B, C, H, W) contiguous: element (b,c,h,w) at b*stride_b + c*stride_h*H*W... assume contiguous strides
    # offset = b*stride_b + (g*cpg + c)*H*W + h*W + w
    base = b_idx * stride_b + g_idx * cpg * (H * W)

    # iterate over group_elems in blocks
    sum_x = tl.zeros([BLOCK], dtype=tl.float32)
    sum_xx = tl.zeros([BLOCK], dtype=tl.float32)
    n = 0
    for off in range(0, group_elems, BLOCK):
        idx = off + tl.arange(0, BLOCK)
        mask = idx < group_elems
        # map linear idx within group to pointer offset
        # within a group, elements are contiguous in memory (channels consecutive, then H, W)
        x = tl.load(x_ptr + base + idx, mask=mask, other=0.0).to(tl.float32)
        sum_x += tl.sum(tl.where(mask, x, 0.0))
        sum_xx += tl.sum(tl.where(mask, x * x, 0.0))
        n += tl.minimum(BLOCK, group_elems - off)

    mean = sum_x / n
    var = sum_xx / n - mean * mean
    rsigma = 1.0 / tl.sqrt(var + eps)

    # second pass: write output with affine
    # weight/bias per channel: channel = g*cpg + (idx // (H*W))
    for off in range(0, group_elems, BLOCK):
        idx = off + tl.arange(0, BLOCK)
        mask = idx < group_elems
        x = tl.load(x_ptr + base + idx, mask=mask, other=0.0).to(tl.float32)
        # channel index for affine
        c_local = idx // (H * W)
        c_global = g_idx * cpg + c_local
        wv = tl.load(w_ptr + c_global, mask=mask, other=0.0).to(tl.float32)
        bv = tl.load(b_ptr + c_global, mask=mask, other=0.0).to(tl.float32)
        y = (x - mean) * rsigma * wv + bv
        tl.store(out_ptr + base + idx, y.to(out_ptr.dtype.element_ty), mask=mask)


def run(x, weight, bias, eps):
    B, C, H, W = x.shape
    num_groups = 32
    cpg = C // num_groups
    group_elems = cpg * H * W
    out = torch.empty_like(x)
    BLOCK = 1024
    grid = (B * num_groups,)
    _group_norm_kernel[grid](
        x, weight, bias, out,
        B, C, H, W, num_groups, group_elems,
        x.stride(0), x.stride(1), x.stride(2), x.stride(3),
        eps, BLOCK=BLOCK,
    )
    return out


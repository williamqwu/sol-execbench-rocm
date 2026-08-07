import torch
import triton
import triton.language as tl


@triton.jit
def _group_norm_kernel(
    x_ptr, w_ptr, b_ptr, out_ptr,
    C, H, W, num_groups, cpg, group_elems,
    stride_b, stride_c,
    eps,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    b_idx = pid // num_groups
    g_idx = pid % num_groups
    base = b_idx * stride_b + g_idx * cpg * (H * W)

    # --- Single pass: sum(x) and sum(x*x) in fp64 (stable moment formula) ---
    sum_x = tl.zeros([1], dtype=tl.float64)
    sum_xx = tl.zeros([1], dtype=tl.float64)
    for off in range(0, group_elems, BLOCK):
        idx = off + tl.arange(0, BLOCK)
        mask = idx < group_elems
        x = tl.load(x_ptr + base + idx, mask=mask, other=0.0).to(tl.float64)
        sum_x += tl.sum(tl.where(mask, x, 0.0))
        sum_xx += tl.sum(tl.where(mask, x * x, 0.0))
    n = group_elems.to(tl.float64)
    mean = sum_x / n
    var = sum_xx / n - mean * mean
    rsigma = 1.0 / tl.sqrt(var + eps)
    mean_f = mean.to(tl.float32)
    rsigma_f = rsigma.to(tl.float32)

    # --- Output pass: normalize + affine ---
    for off in range(0, group_elems, BLOCK):
        idx = off + tl.arange(0, BLOCK)
        mask = idx < group_elems
        x = tl.load(x_ptr + base + idx, mask=mask, other=0.0).to(tl.float32)
        c_local = idx // (H * W)
        c_global = g_idx * cpg + c_local
        wv = tl.load(w_ptr + c_global, mask=mask, other=0.0).to(tl.float32)
        bv = tl.load(b_ptr + c_global, mask=mask, other=0.0).to(tl.float32)
        y = (x - mean_f) * rsigma_f * wv + bv
        tl.store(out_ptr + base + idx, y, mask=mask)


@torch.no_grad()
def run(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor, eps: float) -> torch.Tensor:
    B, C, H, W = x.shape
    num_groups = 32
    cpg = C // num_groups
    group_elems = cpg * H * W
    out = torch.empty_like(x)
    BLOCK = 8192
    grid = (B * num_groups,)
    _group_norm_kernel[grid](
        x, weight, bias, out,
        C, H, W, num_groups, cpg, group_elems,
        x.stride(0), x.stride(1),
        eps, BLOCK=BLOCK, num_warps=8, num_stages=2,
    )
    return out

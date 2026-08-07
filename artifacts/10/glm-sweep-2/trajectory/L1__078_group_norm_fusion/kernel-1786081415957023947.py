import torch
import triton
import triton.language as tl


@triton.jit
def _reduce_kernel(
    x_ptr,
    sm_ptr,          # (B*num_groups, n_chunks, 2) partial sums
    C, H, W, num_groups, cpg, group_elems,
    stride_b, stride_c,
    n_chunks,
    CHUNK: tl.constexpr,
):
    # grid: (B*num_groups, n_chunks)
    pid_g = tl.program_id(0)
    pid_c = tl.program_id(1)
    b_idx = pid_g // num_groups
    g_idx = pid_g % num_groups
    base = b_idx * stride_b + g_idx * cpg * (H * W)

    chunk_start = pid_c * CHUNK
    sum_x = tl.zeros([1], dtype=tl.float64)
    sum_xx = tl.zeros([1], dtype=tl.float64)
    for off in range(0, CHUNK, 4096):
        idx = chunk_start + off + tl.arange(0, 4096)
        mask = (idx < group_elems) & (idx < chunk_start + CHUNK)
        x = tl.load(x_ptr + base + idx, mask=mask, other=0.0).to(tl.float64)
        sum_x += tl.sum(tl.where(mask, x, 0.0))
        sum_xx += tl.sum(tl.where(mask, x * x, 0.0))
    # store partials
    out_off = pid_g * n_chunks * 2 + pid_c * 2
    tl.store(sm_ptr + out_off + 0, sum_x)
    tl.store(sm_ptr + out_off + 1, sum_xx)


@triton.jit
def _norm_kernel(
    x_ptr, w_ptr, b_ptr, out_ptr, sm_ptr,
    C, H, W, num_groups, cpg, group_elems,
    stride_b, stride_c,
    n_chunks,
    eps,
    BLOCK: tl.constexpr,
):
    # grid: (B*num_groups,)  -- one program per group, tiled over group_elems
    pid = tl.program_id(0)
    b_idx = pid // num_groups
    g_idx = pid % num_groups
    base = b_idx * stride_b + g_idx * cpg * (H * W)

    # reduce partials -> mean, rstd
    sum_x = tl.zeros([1], dtype=tl.float64)
    sum_xx = tl.zeros([1], dtype=tl.float64)
    for i in range(n_chunks):
        off = pid * n_chunks * 2 + i * 2
        sum_x += tl.load(sm_ptr + off + 0)
        sum_xx += tl.load(sm_ptr + off + 1)
    n = group_elems.to(tl.float64)
    mean = sum_x / n
    var = sum_xx / n - mean * mean
    rsigma = 1.0 / tl.sqrt(var + eps)
    mean_f = mean.to(tl.float32)
    rsigma_f = rsigma.to(tl.float32)

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

    # choose chunk size: aim for ~4 programs per group on average for occupancy,
    # but keep chunks large enough to amortize launch.
    CHUNK = 8192
    n_chunks = (group_elems + CHUNK - 1) // CHUNK
    sm = torch.empty(B * num_groups * n_chunks * 2, dtype=torch.float64, device=x.device)

    _reduce_kernel[(B * num_groups, n_chunks)](
        x, sm,
        C, H, W, num_groups, cpg, group_elems,
        x.stride(0), x.stride(1),
        n_chunks, CHUNK=CHUNK,
        num_warps=8, num_stages=2,
    )
    _norm_kernel[(B * num_groups,)](
        x, weight, bias, out, sm,
        C, H, W, num_groups, cpg, group_elems,
        x.stride(0), x.stride(1),
        n_chunks, eps, BLOCK=8192,
        num_warps=16, num_stages=2,
    )
    return out

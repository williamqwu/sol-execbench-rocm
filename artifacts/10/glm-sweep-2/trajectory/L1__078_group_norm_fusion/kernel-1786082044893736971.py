import torch
import triton
import triton.language as tl


@triton.jit
def _reduce_kernel(
    x_ptr, sm_ptr,
    C, H, W, num_groups, cpg, group_elems,
    stride_b, stride_c, n_chunks,
    CHUNK: tl.constexpr, BLOCK: tl.constexpr,
):
    # grid: (n_groups, n_chunks) -- partial reduction per group/chunk
    pid_g = tl.program_id(0)
    pid_c = tl.program_id(1)
    b_idx = pid_g // num_groups
    g_idx = pid_g % num_groups
    base = b_idx * stride_b + g_idx * cpg * (H * W)
    chunk_start = pid_c * CHUNK
    chunk_end = chunk_start + CHUNK

    sum_x = tl.zeros([], dtype=tl.float64)
    sum_xx = tl.zeros([], dtype=tl.float64)
    for off in range(0, CHUNK, BLOCK):
        idx = chunk_start + off + tl.arange(0, BLOCK)
        mask = (idx < group_elems) & (idx < chunk_end)
        x = tl.load(x_ptr + base + idx, mask=mask, other=0.0).to(tl.float64)
        m = tl.where(mask, x, 0.0)
        sum_x += tl.sum(m)
        sum_xx += tl.sum(m * x)
    out_off = pid_g * n_chunks * 2 + pid_c * 2
    tl.store(sm_ptr + out_off + 0, sum_x)
    tl.store(sm_ptr + out_off + 1, sum_xx)


@triton.jit
def _norm_kernel(
    x_ptr, w_ptr, b_ptr, out_ptr, sm_ptr,
    C, H, W, num_groups, cpg, group_elems,
    stride_b, stride_c, n_chunks, eps,
    BLOCK: tl.constexpr,
):
    # grid: (n_groups, n_chunks) -- tiled output, each program writes one chunk
    pid_g = tl.program_id(0)
    pid_c = tl.program_id(1)
    b_idx = pid_g // num_groups
    g_idx = pid_g % num_groups
    base = b_idx * stride_b + g_idx * cpg * (H * W)
    chunk_start = pid_c * BLOCK
    chunk_end = chunk_start + BLOCK

    # reduce partials -> mean, rstd (cheap: n_chunks scalars)
    sum_x = tl.zeros([], dtype=tl.float64)
    sum_xx = tl.zeros([], dtype=tl.float64)
    for i in range(n_chunks):
        off = pid_g * n_chunks * 2 + i * 2
        sum_x += tl.load(sm_ptr + off + 0)
        sum_xx += tl.load(sm_ptr + off + 1)
    n = group_elems.to(tl.float64)
    mean = sum_x / n
    var = sum_xx / n - mean * mean
    rsigma = 1.0 / tl.sqrt(var + eps)
    mean_f = mean.to(tl.float32)
    rsigma_f = rsigma.to(tl.float32)

    idx = chunk_start + tl.arange(0, BLOCK)
    mask = (idx < group_elems) & (idx < chunk_end)
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

    CHUNK = 8192
    n_chunks = (group_elems + CHUNK - 1) // CHUNK
    n_groups = B * num_groups
    sm = torch.empty(n_groups * n_chunks * 2, dtype=torch.float64, device=x.device)

    _reduce_kernel[(n_groups, n_chunks)](
        x, sm,
        C, H, W, num_groups, cpg, group_elems,
        x.stride(0), x.stride(1), n_chunks,
        CHUNK=CHUNK, BLOCK=4096,
        num_warps=8, num_stages=2,
    )
    # tiled output: BLOCK == CHUNK so each program writes exactly one chunk
    _norm_kernel[(n_groups, n_chunks)](
        x, weight, bias, out, sm,
        C, H, W, num_groups, cpg, group_elems,
        x.stride(0), x.stride(1), n_chunks, eps,
        BLOCK=CHUNK,
        num_warps=8, num_stages=2,
    )
    return out

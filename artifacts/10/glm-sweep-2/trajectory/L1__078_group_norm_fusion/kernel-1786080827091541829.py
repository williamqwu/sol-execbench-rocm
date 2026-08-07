import torch
import triton
import triton.language as tl

@triton.jit
def _fuse_kernel(x_ptr, w_ptr, b_ptr, out_ptr, mean_ptr, rstd_ptr,
                 C, H, W, num_groups, cpg, group_elems,
                 stride_b, stride_c, eps, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    b_idx = pid // num_groups
    g_idx = pid % num_groups
    base = b_idx * stride_b + g_idx * cpg * (H * W)
    mean = tl.load(mean_ptr + pid)
    rstd = tl.load(rstd_ptr + pid)
    for off in range(0, group_elems, BLOCK):
        idx = off + tl.arange(0, BLOCK)
        mask = idx < group_elems
        x = tl.load(x_ptr + base + idx, mask=mask, other=0.0).to(tl.float32)
        c_local = idx // (H * W)
        c_global = g_idx * cpg + c_local
        wv = tl.load(w_ptr + c_global, mask=mask, other=0.0).to(tl.float32)
        bv = tl.load(b_ptr + c_global, mask=mask, other=0.0).to(tl.float32)
        y = (x - mean) * rstd * wv + bv
        tl.store(out_ptr + base + idx, y, mask=mask)

@torch.no_grad()
def run(x, weight, bias, eps):
    B, C, H, W = x.shape
    num_groups = 32
    cpg = C // num_groups
    group_elems = cpg * H * W
    xg = x.view(B, num_groups, cpg, H, W).to(torch.float32)
    mean = xg.mean(dim=[2,3,4])  # (B, num_groups)
    var = xg.var(dim=[2,3,4], unbiased=False)  # (B, num_groups)
    rstd = 1.0 / torch.sqrt(var + eps)
    out = torch.empty_like(x)
    BLOCK = 1024
    _fuse_kernel[(B*num_groups,)](x, weight, bias, out, mean, rstd,
        C, H, W, num_groups, cpg, group_elems, x.stride(0), x.stride(1), eps, BLOCK=BLOCK)
    return out

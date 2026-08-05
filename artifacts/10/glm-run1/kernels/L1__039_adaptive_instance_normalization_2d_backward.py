import torch
import triton
import triton.language as tl

@triton.jit
def _fuse_fwd(
    x_ptr, go_ptr, mean_ptr, std_ptr, weight_ptr,
    xc_ptr, xn_ptr, gos_ptr, goxn_ptr,
    C, HW, n_elems,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n_elems
    x = tl.load(x_ptr + offs, mask=mask)
    go = tl.load(go_ptr + offs, mask=mask)
    nc_idx = offs // HW
    c_idx = nc_idx % C
    mean_val = tl.load(mean_ptr + nc_idx, mask=mask)
    std_val = tl.load(std_ptr + nc_idx, mask=mask)
    weight_val = tl.load(weight_ptr + c_idx, mask=mask)
    xc = x - mean_val
    xn = xc / std_val
    gos = go * weight_val
    goxn = go * xn
    tl.store(xc_ptr + offs, xc, mask=mask)
    tl.store(xn_ptr + offs, xn, mask=mask)
    tl.store(gos_ptr + offs, gos, mask=mask)
    tl.store(goxn_ptr + offs, goxn, mask=mask)

@triton.jit
def _fuse_buf(
    gos_ptr, xc_ptr, std_ptr,
    buf0_ptr, buf1_ptr, buf2_ptr,
    C, HW, n_elems,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n_elems
    gos = tl.load(gos_ptr + offs, mask=mask)
    xc = tl.load(xc_ptr + offs, mask=mask)
    nc_idx = offs // HW
    std_val = tl.load(std_ptr + nc_idx, mask=mask)
    tl.store(buf0_ptr + offs, gos * xc, mask=mask)
    tl.store(buf1_ptr + offs, gos / (-std_val), mask=mask)
    tl.store(buf2_ptr + offs, xc, mask=mask)

@triton.jit
def _fuse_gi(
    gos_ptr, xc_ptr, grad_var_ptr, grad_mean_ptr,
    std_ptr, out_ptr,
    C, HW, n_elems,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n_elems
    gos = tl.load(gos_ptr + offs, mask=mask)
    xc = tl.load(xc_ptr + offs, mask=mask)
    nc_idx = offs // HW
    std_val = tl.load(std_ptr + nc_idx, mask=mask)
    grad_var = tl.load(grad_var_ptr + nc_idx, mask=mask)
    grad_mean = tl.load(grad_mean_ptr + nc_idx, mask=mask)
    inv_std = 1.0 / std_val
    inv_S = 1.0 / tl.cast(HW, tl.float32)
    gi = gos * inv_std + grad_var * (2.0 * inv_S) * xc + grad_mean * inv_S
    tl.store(out_ptr + offs, gi, mask=mask)

@torch.no_grad()
def run(grad_output, x, weight, mean, std):
    N, C, H, W = x.shape
    HW = H * W
    n_elems = N * C * HW
    BLOCK = 2048
    grid = (triton.cdiv(n_elems, BLOCK),)

    xc = torch.empty_like(x)
    xn = torch.empty_like(x)
    gos = torch.empty_like(x)
    goxn = torch.empty_like(x)

    _fuse_fwd[grid](x, grad_output, mean, std, weight, xc, xn, gos, goxn, C, HW, n_elems, BLOCK=BLOCK)

    grad_bias = grad_output.sum(dim=(0, 2, 3))
    grad_weight = goxn.sum(dim=(0, 2, 3))

    buf = torch.empty(3, N, C, H, W, device=grad_output.device)
    _fuse_buf[grid](gos, xc, std, buf[0], buf[1], buf[2], C, HW, n_elems, BLOCK=BLOCK)
    s = buf.sum(dim=(3, 4))
    s_gos_xc = s[0].view(N, C, 1, 1)
    s_gos_negstd = s[1].view(N, C, 1, 1)
    s_xc = s[2].view(N, C, 1, 1)

    grad_var = s_gos_xc * (-0.5) * torch.pow(std, -3)
    grad_mean = s_gos_negstd + grad_var * (-2.0 * s_xc) / HW

    grad_input = torch.empty_like(x)
    _fuse_gi[grid](gos, xc, grad_var, grad_mean, std, grad_input, C, HW, n_elems, BLOCK=BLOCK)

    return grad_input, grad_weight, grad_bias

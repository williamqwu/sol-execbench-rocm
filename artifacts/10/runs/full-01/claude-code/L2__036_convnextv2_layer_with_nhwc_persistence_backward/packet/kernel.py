import torch
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def _dw_prod(res_ptr, gxh_ptr, out_ptr, g0, C, H, W, HW, M, TOT,
             BLOCK: tl.constexpr):
    pid_g = tl.program_id(0)
    pid_m = tl.program_id(1)
    g = g0 + pid_g
    idx = pid_m * BLOCK + tl.arange(0, BLOCK)
    mask = idx < TOT
    m = idx // 49
    j = idx % 49
    b = m // HW
    r = m - b * HW
    h = r // W
    w = r - h * W
    i = j // 7
    k = j - i * 7
    y = h + i - 3
    x = w + k - 3
    inb = (y >= 0) & (y < H) & (x >= 0) & (x < W) & mask
    rv = tl.load(res_ptr + ((b * C + g) * H + y) * W + x, mask=inb, other=0.0)
    gv = tl.load(gxh_ptr + (b * HW + r) * C + g, mask=mask, other=0.0)
    tl.store(out_ptr + pid_g.to(tl.int64) * TOT + idx, rv * gv, mask=mask)


def _dwconv_weight_grad(residual, gxh, B, C, H, W):
    """Bit-exact replacement for the reference's per-channel unfold loop."""
    HW = H * W
    M = B * HW
    TOT = M * 49
    out = torch.empty((C, 49), device=residual.device, dtype=torch.float32)
    # cap scratch buffer at ~256 MB
    chunk = max(1, min(C, int(256e6) // (TOT * 4) if TOT * 4 < int(256e6) else 1))
    BLOCK = 1024
    grid_m = triton.cdiv(TOT, BLOCK)
    buf = torch.empty((chunk, TOT), device=residual.device, dtype=torch.float32)
    for g0 in range(0, C, chunk):
        n = min(chunk, C - g0)
        _dw_prod[(n, grid_m)](residual, gxh, buf, g0, C, H, W, HW, M, TOT,
                              BLOCK=BLOCK, num_warps=4)
        for i in range(n):
            torch.sum(buf[i].view(M, 49), dim=0, out=out[g0 + i])
    return out.view(C, 1, 7, 7)


@torch.no_grad()
def run(
    grad_output: torch.Tensor,
    residual: torch.Tensor,
    x_dwconv: torch.Tensor,
    x_nhwc: torch.Tensor,
    mean: torch.Tensor,
    var: torch.Tensor,
    x_normalized: torch.Tensor,
    x_ln: torch.Tensor,
    x_expanded: torch.Tensor,
    x_gelu: torch.Tensor,
    global_features: torch.Tensor,
    gf_mean: torch.Tensor,
    norm_features: torch.Tensor,
    x_grn_scaled: torch.Tensor,
    x_grn: torch.Tensor,
    dwconv_weight: torch.Tensor,
    layernorm_weight: torch.Tensor,
    pwconv1_weight: torch.Tensor,
    grn_weight: torch.Tensor,
    pwconv2_weight: torch.Tensor,
    drop_mask: torch.Tensor,
    drop_path_prob: float,
    eps: float,
):
    B = grad_output.shape[0]
    C = grad_output.shape[1]
    H = grad_output.shape[2]
    W = grad_output.shape[3]

    grad_residual = grad_output
    grad_x_nchw = grad_output
    if drop_path_prob > 0.0:
        keep_prob = 1 - drop_path_prob
        grad_x_nchw = grad_x_nchw * drop_mask / keep_prob

    grad_x_projected = grad_x_nchw.permute(0, 2, 3, 1)

    grad_x_grn = F.linear(grad_x_projected, pwconv2_weight.t())

    grad_x_projected_flat = grad_x_projected.reshape(-1, grad_x_projected.shape[-1])
    x_grn_flat = x_grn.reshape(-1, x_grn.shape[-1])
    grad_pwconv2_weight = grad_x_projected_flat.t() @ x_grn_flat
    grad_pwconv2_bias = grad_x_projected.sum(dim=(0, 1, 2))

    grad_x_gelu_from_grn = grad_x_grn
    grad_x_grn_scaled = grad_x_grn * grn_weight

    grad_grn_weight = (grad_x_grn * x_grn_scaled).sum(dim=(0, 1, 2), keepdim=True)
    grad_grn_bias = grad_x_grn.sum(dim=(0, 1, 2), keepdim=True)

    grad_x_gelu_from_scaled = grad_x_grn_scaled * norm_features
    grad_norm_features = (grad_x_grn_scaled * x_gelu).sum(dim=(1, 2), keepdim=True)

    grad_x_gelu = grad_x_gelu_from_grn + grad_x_gelu_from_scaled

    grad_global_features = grad_norm_features / (gf_mean + eps)
    grad_gf_mean = -grad_norm_features * global_features / ((gf_mean + eps) ** 2)
    C_expanded = global_features.shape[-1]
    grad_global_features = grad_global_features + grad_gf_mean / C_expanded
    grad_x_gelu = grad_x_gelu + x_gelu * grad_global_features / (global_features + eps)

    sqrt_2_over_pi = 0.7978845608028654
    cdf_coeff = 0.044715
    inner = sqrt_2_over_pi * (x_expanded + cdf_coeff * x_expanded.pow(3))
    tanh_inner = torch.tanh(inner)
    cdf_approx = 0.5 * (1 + tanh_inner)
    pdf_approx = 0.5 * (1 - tanh_inner.pow(2)) * sqrt_2_over_pi * (1 + 3 * cdf_coeff * x_expanded.pow(2))
    gelu_grad = cdf_approx + x_expanded * pdf_approx
    grad_x_expanded = grad_x_gelu * gelu_grad

    grad_x_ln = F.linear(grad_x_expanded, pwconv1_weight.t())

    grad_x_expanded_flat = grad_x_expanded.reshape(-1, grad_x_expanded.shape[-1])
    x_ln_flat = x_ln.reshape(-1, x_ln.shape[-1])
    grad_pwconv1_weight = grad_x_expanded_flat.t() @ x_ln_flat
    grad_pwconv1_bias = grad_x_expanded.sum(dim=(0, 1, 2))

    grad_x_normalized = grad_x_ln * layernorm_weight
    grad_layernorm_weight = (grad_x_ln * x_normalized).sum(dim=(0, 1, 2))
    grad_layernorm_bias = grad_x_ln.sum(dim=(0, 1, 2))

    std = torch.sqrt(var + eps)
    N = x_nhwc.shape[-1]

    grad_x_nhwc = grad_x_normalized / std
    grad_var = -(grad_x_normalized * (x_nhwc - mean)).sum(dim=-1, keepdim=True) / (2 * (var + eps) * std)
    grad_mean = -(grad_x_normalized / std).sum(dim=-1, keepdim=True)
    grad_mean = grad_mean + grad_var * (-2 * (x_nhwc - mean).sum(dim=-1, keepdim=True) / N)

    grad_x_nhwc = grad_x_nhwc + grad_var * (2 * (x_nhwc - mean) / N)
    grad_x_nhwc = grad_x_nhwc + grad_mean / N

    grad_x_dwconv = grad_x_nhwc.permute(0, 3, 1, 2)

    grad_x = F.conv_transpose2d(grad_x_dwconv, dwconv_weight, padding=3, groups=C)
    grad_x = grad_x + grad_residual

    grad_dwconv_weight = _dwconv_weight_grad(residual, grad_x_nhwc, B, C, H, W)

    grad_dwconv_bias = grad_x_dwconv.sum(dim=(0, 2, 3))

    return (
        grad_x,
        grad_dwconv_weight,
        grad_dwconv_bias,
        grad_layernorm_weight,
        grad_layernorm_bias,
        grad_pwconv1_weight,
        grad_pwconv1_bias,
        grad_grn_weight,
        grad_grn_bias,
        grad_pwconv2_weight,
        grad_pwconv2_bias,
    )

import torch
import torch.nn.functional as F
import triton
import triton.language as tl

_SQRT1_2 = tl.constexpr(0.7071067811865476)


@triton.jit
def _bias_gelu(Y, X, B1, N, C4,
               BM: tl.constexpr, BC: tl.constexpr):
    pid_m = tl.program_id(0)
    pid_c = tl.program_id(1)
    rm = pid_m * BM + tl.arange(0, BM)
    rc = pid_c * BC + tl.arange(0, BC)
    mm = rm < N
    mc = rc < C4
    msk = mm[:, None] & mc[None, :]
    off = rm[:, None] * C4 + rc[None, :]
    x = tl.load(X + off, mask=msk, other=0.0)
    b = tl.load(B1 + rc, mask=mc, other=0.0)
    v = x + b[None, :]
    y = v * 0.5 * (1.0 + tl.erf(v * _SQRT1_2))
    tl.store(Y + off, y, mask=msk)


@triton.jit
def _grn(Z, Y, GW, GB, NF, N, C4, HW,
         BM: tl.constexpr, BC: tl.constexpr):
    pid_m = tl.program_id(0)
    pid_c = tl.program_id(1)
    rm = pid_m * BM + tl.arange(0, BM)
    rc = pid_c * BC + tl.arange(0, BC)
    mm = rm < N
    mc = rc < C4
    msk = mm[:, None] & mc[None, :]
    off = rm[:, None] * C4 + rc[None, :]
    y = tl.load(Y + off, mask=msk, other=0.0)
    gw = tl.load(GW + rc, mask=mc, other=0.0)
    gb = tl.load(GB + rc, mask=mc, other=0.0)
    b = rm // HW
    nf = tl.load(NF + b[:, None] * C4 + rc[None, :], mask=msk, other=0.0)
    z = gw[None, :] * (y * nf) + gb[None, :] + y
    tl.store(Z + off, z, mask=msk)


@triton.jit
def _bias_res_t(OUT, M2, B2, X, N, C, HW,
                BM: tl.constexpr, BC: tl.constexpr):
    pid_m = tl.program_id(0)
    pid_c = tl.program_id(1)
    rm = pid_m * BM + tl.arange(0, BM)
    rc = pid_c * BC + tl.arange(0, BC)
    mm = rm < N
    mc = rc < C
    msk = mm[:, None] & mc[None, :]
    m2 = tl.load(M2 + rm[:, None] * C + rc[None, :], mask=msk, other=0.0)
    b2 = tl.load(B2 + rc, mask=mc, other=0.0)
    t = m2 + b2[None, :]
    b = rm // HW
    hw = rm % HW
    # out[b, c, hw]  <- x[b, c, hw] + t[rm, c]
    oidx = b[:, None] * (C * HW) + rc[None, :] * HW + hw[:, None]
    x = tl.load(X + oidx, mask=msk, other=0.0)
    tl.store(OUT + oidx, x + t, mask=msk)


@torch.no_grad()
def run(
    x: torch.Tensor,
    dwconv_weight: torch.Tensor,
    dwconv_bias: torch.Tensor,
    layernorm_weight: torch.Tensor,
    layernorm_bias: torch.Tensor,
    pwconv1_weight: torch.Tensor,
    pwconv1_bias: torch.Tensor,
    grn_weight: torch.Tensor,
    grn_bias: torch.Tensor,
    pwconv2_weight: torch.Tensor,
    pwconv2_bias: torch.Tensor,
    eps: float,
    layer_norm_eps: float,
):
    B, C, H, W = x.shape
    HW = H * W
    N = B * HW
    C4 = 4 * C

    out = F.conv2d(x, dwconv_weight, dwconv_bias, padding=3, groups=C)
    out = out.permute(0, 2, 3, 1)
    out = F.layer_norm(out, (C,), layernorm_weight, layernorm_bias,
                       eps=layer_norm_eps)

    m1 = torch.matmul(out.reshape(N, C), pwconv1_weight.T)

    y = torch.empty_like(m1)
    BM, BC = 32, 64
    grid = (triton.cdiv(N, BM), triton.cdiv(C4, BC))
    _bias_gelu[grid](y, m1, pwconv1_bias, N, C4, BM, BC,
                     enable_fp_fusion=False)

    y4 = y.view(B, H, W, C4)
    gf = torch.linalg.vector_norm(y4, ord=2, dim=(1, 2), keepdim=True)
    nf = gf / (gf.mean(dim=-1, keepdim=True) + eps)

    z = torch.empty_like(y)
    _grn[grid](z, y, grn_weight, grn_bias, nf.reshape(B, C4),
               N, C4, HW, BM, BC, enable_fp_fusion=False)

    m2 = torch.matmul(z, pwconv2_weight.T)

    res = torch.empty_like(x)
    grid2 = (triton.cdiv(N, BM), triton.cdiv(C, BC))
    _bias_res_t[grid2](res, m2, pwconv2_bias, x, N, C, HW, BM, BC,
                       enable_fp_fusion=False)
    return res

import torch
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def _affine_kernel(XP, GFP, DENP, GWP, GBP, YP, M, SP, EPS,
                   C: tl.constexpr, BLOCK_M: tl.constexpr):
    """y = gw*(x*nf) + gb + x   where nf = gf / (den + eps), per batch row."""
    pid = tl.program_id(0)
    rm = pid * BLOCK_M + tl.arange(0, BLOCK_M)
    rc = tl.arange(0, C)
    mask = rm < M
    off = rm[:, None] * C + rc[None, :]
    x = tl.load(XP + off, mask=mask[:, None], other=0.0)
    b = rm // SP
    gf = tl.load(GFP + b[:, None] * C + rc[None, :], mask=mask[:, None], other=0.0)
    den = tl.load(DENP + b, mask=mask, other=1.0)[:, None]
    nf = gf / (den + EPS)
    gw = tl.load(GWP + rc)[None, :]
    gb = tl.load(GBP + rc)[None, :]
    y = gw * (x * nf) + gb + x
    tl.store(YP + off, y, mask=mask[:, None])


@torch.no_grad()
def run(hidden_states, pwconv1_weight, pwconv1_bias, grn_weight, grn_bias,
        pwconv2_weight, pwconv2_bias, eps):
    B, H, W, D = hidden_states.shape
    C = pwconv1_weight.shape[0]
    SP = H * W
    M = B * SP

    xf = hidden_states.reshape(M, D)
    x0 = torch.addmm(pwconv1_bias, xf, pwconv1_weight.t())
    x = F.gelu(x0)

    x4 = x.view(B, H, W, C)
    gf = torch.linalg.vector_norm(x4, ord=2, dim=(1, 2), keepdim=True)
    den = gf.mean(dim=-1).reshape(B)

    y = torch.empty_like(x)
    BLOCK_M = 16
    _affine_kernel[(triton.cdiv(M, BLOCK_M),)](
        x, gf, den, grn_weight, grn_bias, y, M, SP, eps,
        C=C, BLOCK_M=BLOCK_M, num_warps=8, enable_fp_fusion=False)

    out = torch.addmm(pwconv2_bias, y, pwconv2_weight.t())
    return out.view(B, H, W, pwconv2_weight.shape[0])

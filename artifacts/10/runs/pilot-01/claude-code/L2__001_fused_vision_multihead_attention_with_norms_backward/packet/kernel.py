import torch
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def _smax_bwd_ew(GAW, AW, SG, O, S, BL: tl.constexpr):
    r = tl.program_id(0).to(tl.int64)
    o = tl.arange(0, BL)
    m = o < S
    b = r * S + o
    ga = tl.load(GAW + b, mask=m, other=0.0)
    a = tl.load(AW + b, mask=m, other=0.0)
    sg = tl.load(SG + r)
    tl.store(O + b, a * (ga - sg), mask=m)


# stage 1: from grad_x_norm produce  pw = gxn*xnorm (for grad_ln_weight),
# gm = (gxn*lw)/std, and prod = (gxn*lw)*xc  -- all elementwise, bit-exact.
@triton.jit
def _ln_stage1(GXN, X, XM, XV, LW, GM, PROD, PW, XC, STDO, eps,
               E: tl.constexpr, BL: tl.constexpr):
    r = tl.program_id(0).to(tl.int64)
    o = tl.arange(0, BL)
    m = o < E
    b = r * E + o
    gxn = tl.load(GXN + b, mask=m, other=0.0)
    x = tl.load(X + b, mask=m, other=0.0)
    lw = tl.load(LW + o, mask=m, other=0.0)
    xm = tl.load(XM + r)
    xv = tl.load(XV + r)
    std = tl.sqrt_rn(xv + eps)
    xc = x - xm
    tl.store(PW + b, gxn * (xc / std), mask=m)
    tl.store(XC + b, xc, mask=m)
    gxnm = gxn * lw
    tl.store(GM + b, gxnm / std, mask=m)
    tl.store(PROD + b, gxnm * xc, mask=m)
    tl.store(STDO + r, std)


# stage 2: grad_x = go + (gm - mg - xc*mgx)
@triton.jit
def _ln_stage2(GM, XC, MG, MGX, GO, GX, E: tl.constexpr, BL: tl.constexpr):
    r = tl.program_id(0).to(tl.int64)
    o = tl.arange(0, BL)
    m = o < E
    b = r * E + o
    gm = tl.load(GM + b, mask=m, other=0.0)
    xc = tl.load(XC + b, mask=m, other=0.0)
    go = tl.load(GO + b, mask=m, other=0.0)
    mg = tl.load(MG + r)
    mgx = tl.load(MGX + r)
    # `+ 0.0` blocks FMA contraction of xc*mgx into the subtract, which the
    # reference (separate torch ops) does not do. Required for 1-ULP tolerance.
    t = (xc * mgx) + 0.0
    tl.store(GX + b, go + (gm - mg - t), mask=m)


@torch.no_grad()
def run(grad_output, x, x_mean, x_var, x_norm, ln_weight, qkv_weight,
        q, k, v, attn_weights, attn_output, out_weight, scale, norm_eps):
    B, S, E = x.shape
    H, D = 16, 64
    N = B * S
    E3 = 3 * E

    grad_attn_output = F.linear(grad_output, out_weight.t())
    go2 = grad_output.reshape(-1, E)
    grad_out_weight = torch.matmul(go2.t(), attn_output.reshape(-1, E))
    grad_out_bias = go2.sum(0)

    gAOh = grad_attn_output.view(B, S, H, D).transpose(1, 2).contiguous()

    grad_v = torch.matmul(attn_weights.transpose(-2, -1), gAOh)
    gaw = torch.matmul(gAOh, v.transpose(-2, -1))

    sg = (gaw * attn_weights).sum(dim=-1, keepdim=True)
    gas = torch.empty_like(gaw)
    _smax_bwd_ew[(B * H * S,)](gaw, attn_weights, sg, gas, S,
                               BL=triton.next_power_of_2(S), num_warps=8)
    del gaw

    grad_qkv = torch.empty((B, S, E3), device=x.device, dtype=x.dtype)
    qv = grad_qkv.view(B, S, 3, H, D)
    torch.mul(torch.matmul(gas, k), scale, out=qv[:, :, 0].permute(0, 2, 1, 3))
    qv[:, :, 1].permute(0, 2, 1, 3).copy_(
        torch.matmul(gas.transpose(-2, -1), q * scale))
    qv[:, :, 2].permute(0, 2, 1, 3).copy_(grad_v)
    del gas, grad_v

    grad_x_norm = F.linear(grad_qkv, qkv_weight.t())
    gq2 = grad_qkv.reshape(-1, E3)
    grad_qkv_weight = torch.matmul(gq2.t(), x_norm.reshape(-1, E))
    grad_qkv_bias = gq2.sum(0)
    del grad_qkv

    gm = torch.empty_like(x)
    prod = torch.empty_like(x)
    pw = torch.empty_like(x)
    xc = torch.empty_like(x)
    stdo = torch.empty((N,), device=x.device, dtype=x.dtype)
    BL = triton.next_power_of_2(E)
    _ln_stage1[(N,)](grad_x_norm, x, x_mean, x_var, ln_weight,
                     gm, prod, pw, xc, stdo, norm_eps, E=E, BL=BL, num_warps=8)

    grad_ln_weight = pw.reshape(-1, E).sum(0)
    grad_ln_bias = grad_x_norm.reshape(-1, E).sum(0)
    del pw, grad_x_norm

    std2 = stdo.view(N, 1)
    mg = gm.reshape(N, E).mean(dim=-1, keepdim=True)
    mgx = prod.reshape(N, E).mean(dim=-1, keepdim=True) / (std2 * std2)
    del prod

    grad_x = torch.empty_like(x)
    _ln_stage2[(N,)](gm, xc, mg, mgx, grad_output, grad_x, E=E, BL=BL,
                     num_warps=8)

    return (grad_x, grad_qkv_weight, grad_qkv_bias, grad_out_weight,
            grad_out_bias, grad_ln_weight, grad_ln_bias)

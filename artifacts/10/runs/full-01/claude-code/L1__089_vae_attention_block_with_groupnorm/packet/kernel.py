import torch
import triton
import triton.language as tl


# ---------------------------------------------------------------------------
# Kernel 1: group-norm (normalize + affine) fused with the (B,C,L) -> (B,L,C)
# transpose.  Mean/var come from torch so their reduction tree matches the
# reference bit-for-bit; everything after them is elementwise and reproduced
# exactly:  (x - mean) / sqrt(var + eps) * w + b
# ---------------------------------------------------------------------------
@triton.jit
def _gn_transpose(
    X, MEAN, VAR, W, B, OUT,
    L, C, CPG,
    eps,
    stride_xb, stride_xc,
    stride_ob, stride_ol,
    stride_mb, stride_mg,
    BLOCK_C: tl.constexpr, BLOCK_L: tl.constexpr,
):
    pid_b = tl.program_id(0)
    pid_c = tl.program_id(1)
    pid_l = tl.program_id(2)

    offs_c = pid_c * BLOCK_C + tl.arange(0, BLOCK_C)
    offs_l = pid_l * BLOCK_L + tl.arange(0, BLOCK_L)
    mask_c = offs_c < C
    mask_l = offs_l < L

    # read x[b, c, l] : coalesced along l
    xp = X + pid_b * stride_xb + offs_c[:, None] * stride_xc + offs_l[None, :]
    x = tl.load(xp, mask=mask_c[:, None] & mask_l[None, :], other=0.0)

    grp = offs_c // CPG
    mean = tl.load(MEAN + pid_b * stride_mb + grp * stride_mg, mask=mask_c, other=0.0)
    var = tl.load(VAR + pid_b * stride_mb + grp * stride_mg, mask=mask_c, other=0.0)
    w = tl.load(W + offs_c, mask=mask_c, other=0.0)
    b = tl.load(B + offs_c, mask=mask_c, other=0.0)

    # Reproduce the reference's rounding exactly: a true IEEE divide by
    # sqrt(var+eps), then a *separate* multiply and add.  Triton would
    # otherwise contract the mul+add into an FMA, which skips the
    # intermediate rounding torch performs and breaks the tolerance.
    rstd = tl.sqrt_rn(var + eps)
    y = tl.div_rn(x - mean[:, None], rstd[:, None])
    p = y * w[:, None]
    p = tl.inline_asm_elementwise(
        "v_mov_b32 $0, $1", "=v,v", [p], dtype=tl.float32, is_pure=True, pack=1
    )
    y = p + b[:, None]

    # write out[b, l, c] : coalesced along c
    op = OUT + pid_b * stride_ob + offs_l[:, None] * stride_ol + offs_c[None, :]
    tl.store(op, tl.trans(y), mask=mask_l[:, None] & mask_c[None, :])


# ---------------------------------------------------------------------------
# Kernel 2: scale + softmax fused into one pass over the score matrix.
# softmax(s*scale) with max subtraction; multiplying by a positive constant
# commutes with max, so folding the scale in is exact.
# ---------------------------------------------------------------------------
@triton.jit
def _scale_softmax(
    S, OUT, N, scale,
    stride_r,
    BLOCK_N: tl.constexpr,
):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK_N)
    mask = offs < N
    p = tl.load(S + row * stride_r + offs, mask=mask, other=-float("inf"))
    p = p * scale
    p = tl.where(mask, p, -float("inf"))
    m = tl.max(p, 0)
    e = tl.exp(p - m)
    e = tl.where(mask, e, 0.0)
    s = tl.sum(e, 0)
    tl.store(OUT + row * stride_r + offs, e / s, mask=mask)


# ---------------------------------------------------------------------------
# Kernel 3: output-projection bias + (B,L,C) -> (B,C,L) transpose + residual,
# all in one pass.  Reproduces  x + (mm(attn, Wo^T) + bias).
# ---------------------------------------------------------------------------
@triton.jit
def _bias_transpose_residual(
    T, BIAS, X, OUT,
    L, C,
    stride_tb, stride_tl,
    stride_xb, stride_xc,
    BLOCK_C: tl.constexpr, BLOCK_L: tl.constexpr,
):
    pid_b = tl.program_id(0)
    pid_c = tl.program_id(1)
    pid_l = tl.program_id(2)

    offs_c = pid_c * BLOCK_C + tl.arange(0, BLOCK_C)
    offs_l = pid_l * BLOCK_L + tl.arange(0, BLOCK_L)
    mask_c = offs_c < C
    mask_l = offs_l < L

    # read t[b, l, c] : coalesced along c
    tp = T + pid_b * stride_tb + offs_l[:, None] * stride_tl + offs_c[None, :]
    t = tl.load(tp, mask=mask_l[:, None] & mask_c[None, :], other=0.0)
    bias = tl.load(BIAS + offs_c, mask=mask_c, other=0.0)
    t = t + bias[None, :]

    tt = tl.trans(t)  # (BLOCK_C, BLOCK_L)

    xp = X + pid_b * stride_xb + offs_c[:, None] * stride_xc + offs_l[None, :]
    m2 = mask_c[:, None] & mask_l[None, :]
    x = tl.load(xp, mask=m2, other=0.0)

    tl.store(OUT + pid_b * stride_xb + offs_c[:, None] * stride_xc + offs_l[None, :],
             x + tt, mask=m2)


def _next_pow2(n):
    return 1 << (n - 1).bit_length()


@torch.no_grad()
def run(
    x: torch.Tensor,
    group_norm_weight: torch.Tensor,
    group_norm_bias: torch.Tensor,
    query_weight: torch.Tensor,
    query_bias: torch.Tensor,
    key_weight: torch.Tensor,
    key_bias: torch.Tensor,
    value_weight: torch.Tensor,
    value_bias: torch.Tensor,
    proj_out_weight: torch.Tensor,
    proj_out_bias: torch.Tensor,
    eps: float,
):
    batch, channels, height, width = x.shape
    num_groups = 32
    cpg = channels // num_groups
    L = height * width
    C = channels

    x = x.contiguous()
    xg = x.view(batch, num_groups, cpg, height, width)

    # Reference reduction trees, kept as-is (bit-exact).
    mean = xg.mean(dim=(2, 3, 4), keepdim=True)
    var = xg.var(dim=(2, 3, 4), keepdim=True, unbiased=False)
    mean = mean.view(batch, num_groups)
    var = var.view(batch, num_groups)

    # --- fused group-norm + transpose -> x_seq (B, L, C) ---
    x_seq = torch.empty((batch, L, C), device=x.device, dtype=x.dtype)
    BC, BL = 64, 64
    grid = (batch, triton.cdiv(C, BC), triton.cdiv(L, BL))
    _gn_transpose[grid](
        x, mean, var, group_norm_weight, group_norm_bias, x_seq,
        L, C, cpg, eps,
        x.stride(0), x.stride(1),
        x_seq.stride(0), x_seq.stride(1),
        mean.stride(0), mean.stride(1),
        BLOCK_C=BC, BLOCK_L=BL, num_warps=8,
    )

    # --- fused QKV projection (one GEMM, bit-exact vs three) ---
    x2 = x_seq.view(-1, C)
    qkv_w = torch.cat([query_weight, key_weight, value_weight], 0)
    qkv_b = torch.cat([query_bias, key_bias, value_bias], 0)
    qkv = torch.addmm(qkv_b, x2, qkv_w.t())
    qkv = qkv.view(batch, L, 3 * C)
    q = qkv[:, :, :C]
    k = qkv[:, :, C:2 * C]
    v = qkv[:, :, 2 * C:]

    # --- attention ---
    scores = torch.bmm(q, k.transpose(1, 2))

    BLOCK_N = _next_pow2(L)
    nw = 8 if BLOCK_N >= 2048 else 4
    sv = scores.view(-1, L)
    probs = torch.empty_like(sv)
    _scale_softmax[(sv.shape[0],)](
        sv, probs, L, C ** -0.5, sv.stride(0),
        BLOCK_N=BLOCK_N, num_warps=nw,
    )
    del scores
    probs = probs.view(batch, L, L)

    attn = torch.bmm(probs, v)
    del probs

    # --- output projection (bias folded into the transpose kernel) ---
    t = torch.mm(attn.view(-1, C), proj_out_weight.t())
    t = t.view(batch, L, C)

    out = torch.empty_like(x)
    _bias_transpose_residual[grid](
        t, proj_out_bias, x, out,
        L, C,
        t.stride(0), t.stride(1),
        x.stride(0), x.stride(1),
        BLOCK_C=BC, BLOCK_L=BL, num_warps=8,
    )
    return out.view(batch, channels, height, width)

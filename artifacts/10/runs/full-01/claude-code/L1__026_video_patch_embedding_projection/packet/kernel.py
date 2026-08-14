import torch
import triton
import triton.language as tl


# ---------------------------------------------------------------------------
# Problem:
#   video (B,F,C,H,W) fp16 -> conv3d with 1x P x P kernel, stride (1,P,P)
#   -> tokens (B, F*S, N) with N = hidden = 5120, S = (H/P)*(W/P)
#   -> + spatial_pos[s] + temporal_pos[f] -> LayerNorm over N.
#
# The conv is exactly a GEMM: each output token row is a dot product of a
# K = C*P*P vector gathered from the video with the (N, K) weight matrix.
#
# The output tensor dominates all memory traffic (e.g. 10.4 GB for the
# 224x224 workload vs 130 MB of input video), so the whole point is to write
# it exactly once.  Two kernels share the same tiling:
#   pass 1: recompute the GEMM, keep nothing, reduce row sum / sum-of-squares
#   pass 2: recompute the GEMM, normalise, store the single output tile
# Pass 1 touches no big output buffer, so total traffic is ~ 1x output write
# plus a few hundred MB of re-read inputs.
#
# Numerics follow the reference op-by-op: the conv accumulates in fp32 then
# rounds to fp16, each positional-embedding add rounds to fp16, LN statistics
# accumulate in fp32 and are rounded to fp16 exactly where torch rounds them.
# ---------------------------------------------------------------------------


@triton.jit
def _row_index(m, F, S, C, HW, W, W2, P):
    """row m -> (video base offset, s, f)"""
    fs = m % (F * S)
    b = m // (F * S)
    f = fs // S
    s = fs % S
    h2 = s // W2
    w2 = s % W2
    base = (b * F + f) * C * HW + (h2 * P) * W + w2 * P
    return base, s, f


@triton.jit
def _stats_kernel(
    VID, WGT, BIAS, SP, TP, MEAN, RSTD,
    M, N, F, S, C, HW, W, W2,
    eps,
    P: tl.constexpr, K: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
):
    pid = tl.program_id(0)
    rm = pid * BLOCK_M + tl.arange(0, BLOCK_M)
    mask_m = rm < M
    m = tl.where(mask_m, rm, 0)

    base, s, f = _row_index(m, F, S, C, HW, W, W2, P)

    # k -> (c, p, q) with k = c*P*P + p*P + q
    k = tl.arange(0, K)
    c = k // (P * P)
    p = (k // P) % P
    q = k % P
    koff = c * HW + p * W + q

    a = tl.load(VID + base[:, None] + koff[None, :]).to(tl.float16)

    acc_s = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc_q = tl.zeros([BLOCK_M], dtype=tl.float32)

    for n0 in range(0, N, BLOCK_N):
        rn = n0 + tl.arange(0, BLOCK_N)
        wt = tl.load(WGT + rn[None, :] * K + k[:, None]).to(tl.float16)
        acc = tl.dot(a, wt, out_dtype=tl.float32)
        x = acc.to(tl.float16) + tl.load(BIAS + rn)[None, :]          # conv bias
        x = x.to(tl.float16) + tl.load(SP + s[:, None] * N + rn[None, :])
        x = x.to(tl.float16) + tl.load(TP + f[:, None] * N + rn[None, :])
        xf = x.to(tl.float16).to(tl.float32)
        acc_s += tl.sum(xf, 1)
        acc_q += tl.sum(xf * xf, 1)

    mean = acc_s / N
    var = acc_q / N - mean * mean
    mean16 = mean.to(tl.float16)
    var16 = var.to(tl.float16)
    denom = tl.sqrt((var16.to(tl.float32) + eps).to(tl.float16).to(tl.float32)).to(tl.float16)
    tl.store(MEAN + rm, mean16, mask=mask_m)
    tl.store(RSTD + rm, denom, mask=mask_m)


@triton.jit
def _out_kernel(
    VID, WGT, BIAS, SP, TP, MEAN, RSTD, NW, NB, OUT,
    M, N, F, S, C, HW, W, W2,
    P: tl.constexpr, K: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    mask_m = rm < M
    m = tl.where(mask_m, rm, 0)

    base, s, f = _row_index(m, F, S, C, HW, W, W2, P)

    k = tl.arange(0, K)
    c = k // (P * P)
    p = (k // P) % P
    q = k % P
    koff = c * HW + p * W + q
    a = tl.load(VID + base[:, None] + koff[None, :]).to(tl.float16)

    mean = tl.load(MEAN + m)[:, None]
    denom = tl.load(RSTD + m)[:, None].to(tl.float32)

    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    wt = tl.load(WGT + rn[None, :] * K + k[:, None]).to(tl.float16)
    acc = tl.dot(a, wt, out_dtype=tl.float32)
    x = acc.to(tl.float16) + tl.load(BIAS + rn)[None, :]
    x = x.to(tl.float16) + tl.load(SP + s[:, None] * N + rn[None, :])
    x = x.to(tl.float16) + tl.load(TP + f[:, None] * N + rn[None, :])
    x = x.to(tl.float16)

    d = (x - mean).to(tl.float16).to(tl.float32)
    t = (d / denom).to(tl.float16)
    t = (t * tl.load(NW + rn)[None, :]).to(tl.float16)
    t = (t + tl.load(NB + rn)[None, :]).to(tl.float16)

    om = m.to(tl.int64)
    tl.store(OUT + om[:, None] * N + rn[None, :], t, mask=mask_m[:, None])


@torch.no_grad()
def run(
    video: torch.Tensor,
    patch_projection_weight: torch.Tensor,
    patch_projection_bias: torch.Tensor,
    spatial_pos_embedding: torch.Tensor,
    temporal_pos_embedding: torch.Tensor,
    norm_weight: torch.Tensor,
    norm_bias: torch.Tensor,
    eps: float,
):
    B, F, C, H, W = video.shape
    N = patch_projection_weight.shape[0]
    P = patch_projection_weight.shape[3]
    H2, W2 = H // P, W // P
    S = H2 * W2
    K = C * P * P
    M = B * F * S

    video = video.contiguous()
    wgt = patch_projection_weight.reshape(N, K).contiguous()
    sp = spatial_pos_embedding.reshape(S, N).contiguous()
    tp = temporal_pos_embedding.reshape(F, N).contiguous()

    out = torch.empty((B, F * S, N), dtype=video.dtype, device=video.device)
    mean = torch.empty((M,), dtype=video.dtype, device=video.device)
    rstd = torch.empty((M,), dtype=video.dtype, device=video.device)

    BLOCK_M = 32
    BLOCK_N = 256

    _stats_kernel[(triton.cdiv(M, BLOCK_M),)](
        video, wgt, patch_projection_bias, sp, tp, mean, rstd,
        M, N, F, S, C, H * W, W, W2,
        float(eps),
        P=P, K=K, BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N,
        num_warps=8, num_stages=2,
    )
    _out_kernel[(triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))](
        video, wgt, patch_projection_bias, sp, tp, mean, rstd,
        norm_weight, norm_bias, out,
        M, N, F, S, C, H * W, W, W2,
        P=P, K=K, BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N,
        num_warps=8, num_stages=2,
    )
    return out

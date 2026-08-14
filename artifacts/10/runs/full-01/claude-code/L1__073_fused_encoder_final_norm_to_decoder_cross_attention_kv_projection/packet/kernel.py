# Fused RMSNorm -> K/V projection -> multi-head layout, for MI355X (gfx950).
#
# Numerics note (measured, do not "optimize" away):
#   The reference computes  fp16(x * rsqrt(mean(x^2)+eps) * g) @ W^T .
#   Applying the RMS scale AFTER the GEMM is algebraically identical and allows
#   a single pass over x, but it changes the fp16 rounding of the GEMM input
#   and drops the matched-element ratio to ~0.83 (needs >= 0.99). Measured on
#   all 16 workloads. The two-pass form below is therefore required: pass 1
#   computes the row sum-of-squares, pass 2 rounds to fp16 exactly as the
#   reference does before the dot. x is re-read in pass 2 from cache.
#
# The scale/gain product is kept in fp32 and the rounding to fp16 happens at
# exactly the same point as in the reference.
import torch
import triton
import triton.language as tl


@triton.jit
def _fused_rmsnorm_kv(
    X, NW, WK, WV, KO, VO,
    M, S, eps,
    BLOCK_M: tl.constexpr,
    BLOCK_K: tl.constexpr,
    H: tl.constexpr,
    N: tl.constexpr,
    HD: tl.constexpr,
    NUM_K: tl.constexpr,
    DT: tl.constexpr,
):
    pid = tl.program_id(0)
    offs_m = pid * BLOCK_M + tl.arange(0, BLOCK_M)
    mask_m = offs_m < M
    rm = tl.where(mask_m, offs_m, 0)

    xbase = X + rm[:, None] * H

    # pass 1: row sum of squares (fp32)
    ss = tl.zeros((BLOCK_M,), tl.float32)
    for i in tl.static_range(NUM_K):
        ok = i * BLOCK_K + tl.arange(0, BLOCK_K)
        x = tl.load(xbase + ok[None, :]).to(tl.float32)
        ss += tl.sum(x * x, axis=1)
    scale = 1.0 / tl.sqrt(ss * (1.0 / H) + eps)

    # pass 2: normalize (round to input dtype) then project
    offs_n = tl.arange(0, N)
    acck = tl.zeros((BLOCK_M, N), tl.float32)
    accv = tl.zeros((BLOCK_M, N), tl.float32)
    wkbase = WK + offs_n[:, None] * H
    wvbase = WV + offs_n[:, None] * H
    for i in tl.static_range(NUM_K):
        ok = i * BLOCK_K + tl.arange(0, BLOCK_K)
        x = tl.load(xbase + ok[None, :]).to(tl.float32)
        g = tl.load(NW + ok).to(tl.float32)
        xn = ((x * scale[:, None]) * g[None, :]).to(DT)
        wk = tl.load(wkbase + ok[None, :])
        wv = tl.load(wvbase + ok[None, :])
        acck = tl.dot(xn, tl.trans(wk), acck)
        accv = tl.dot(xn, tl.trans(wv), accv)

    b = rm // S
    s = rm % S
    head = offs_n // HD
    d = offs_n - head * HD
    off = (b[:, None] * (2 * S * HD) + head[None, :] * (S * HD)
           + s[:, None] * HD + d[None, :])
    tl.store(KO + off, acck.to(DT), mask=mask_m[:, None])
    tl.store(VO + off, accv.to(DT), mask=mask_m[:, None])


_TL_DT = {
    torch.float16: tl.float16,
    torch.bfloat16: tl.bfloat16,
    torch.float32: tl.float32,
}


def _pick(M):
    # Measured on MI355X (gfx950), fp16, H=1024, N=128.
    # Small M is launch-latency bound; large M is weight-reload bound.
    if M <= 1024:
        return 16, 256, 8, 2
    return 32, 128, 4, 1


@torch.no_grad()
def run(
    encoder_hidden_states: torch.Tensor,
    norm_weight: torch.Tensor,
    k_proj_weight: torch.Tensor,
    v_proj_weight: torch.Tensor,
    eps: float,
):
    x = encoder_hidden_states
    if not x.is_contiguous():
        x = x.contiguous()
    B, S, H = x.shape
    NKV = 2
    HD = 64
    N = k_proj_weight.shape[0]
    M = B * S

    keys = torch.empty((B, NKV, S, HD), dtype=x.dtype, device=x.device)
    values = torch.empty((B, NKV, S, HD), dtype=x.dtype, device=x.device)

    BLOCK_M, BLOCK_K, nw, ns = _pick(M)
    grid = (triton.cdiv(M, BLOCK_M),)
    _fused_rmsnorm_kv[grid](
        x, norm_weight, k_proj_weight, v_proj_weight, keys, values,
        M, S, eps,
        BLOCK_M=BLOCK_M, BLOCK_K=BLOCK_K,
        H=H, N=N, HD=HD, NUM_K=H // BLOCK_K,
        DT=_TL_DT[x.dtype],
        num_warps=nw, num_stages=ns,
    )
    return keys, values

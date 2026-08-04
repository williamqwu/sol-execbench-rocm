import math

import torch
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def _gelu(x):
    return x * 0.5 * (1.0 + tl.math.erf(x * 0.7071067811865476))


# ---------------- conv1: [B,1,80,T] -> NHWC [B,40,T2,384] ----------------
@triton.jit
def conv1_kernel(X, W, Bi, Y,
                 T, T2, M,
                 BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    mm = m < M
    b = m // (40 * T2)
    r = m % (40 * T2)
    of = r // T2
    ot = r % T2

    k = tl.arange(0, 16)
    kf = k // 3
    kt = k % 3
    idx_f = 2 * of[:, None] + kf[None, :] - 1
    idx_t = 2 * ot[:, None] + kt[None, :] - 1
    off = (b[:, None].to(tl.int64) * 80 + idx_f) * T + idx_t
    msk = (k[None, :] < 9) & (idx_f >= 0) & (idx_f < 80) & (idx_t >= 0) & (idx_t < T) & mm[:, None]
    a = tl.load(X + off, mask=msk, other=0.0)

    n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    w = tl.load(W + n[None, :] * 16 + k[:, None])

    acc = tl.dot(a, w, out_dtype=tl.float32)
    acc += tl.load(Bi + n)[None, :].to(tl.float32)
    acc = acc.to(tl.bfloat16).to(tl.float32)
    acc = _gelu(acc)
    tl.store(Y + m[:, None].to(tl.int64) * 384 + n[None, :], acc.to(tl.bfloat16), mask=mm[:, None])


# -------- conv2/3 (A): one program covers all C=384 outputs -------------
@triton.jit
def conv_kernel3(X, W, Bi, Y,
                 F_, T, F2, T2, M, C: tl.constexpr,
                 OUT_MODE: tl.constexpr,
                 BLOCK_M: tl.constexpr, BLOCK_K: tl.constexpr):
    pid_m = tl.program_id(0)
    m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    mm = m < M
    b = m // (F2 * T2)
    r = m % (F2 * T2)
    of = r // T2
    ot = r % T2

    n = tl.arange(0, 128)
    acc0 = tl.zeros((BLOCK_M, 128), dtype=tl.float32)
    acc1 = tl.zeros((BLOCK_M, 128), dtype=tl.float32)
    acc2 = tl.zeros((BLOCK_M, 128), dtype=tl.float32)
    kc = tl.arange(0, BLOCK_K)
    for kf in tl.static_range(3):
        idx_f = 2 * of + kf - 1
        vf = (idx_f >= 0) & (idx_f < F_) & mm
        for kt in tl.static_range(3):
            idx_t = 2 * ot + kt - 1
            v = vf & (idx_t >= 0) & (idx_t < T)
            base = ((b.to(tl.int64) * F_ + idx_f) * T + idx_t) * C
            wbase = (kf * 3 + kt) * C * C
            for k0 in range(0, C, BLOCK_K):
                a = tl.load(X + base[:, None] + (k0 + kc)[None, :], mask=v[:, None], other=0.0)
                wp = W + wbase + (k0 + kc)[:, None] * C + n[None, :]
                acc0 = tl.dot(a, tl.load(wp), acc0)
                acc1 = tl.dot(a, tl.load(wp + 128), acc1)
                acc2 = tl.dot(a, tl.load(wp + 256), acc2)

    for j in tl.static_range(3):
        nn = j * 128 + n
        acc = tl.where(j == 0, acc0, tl.where(j == 1, acc1, acc2))
        acc += tl.load(Bi + nn)[None, :].to(tl.float32)
        acc = acc.to(tl.bfloat16).to(tl.float32)
        acc = _gelu(acc)
        if OUT_MODE == 0:
            o = m[:, None].to(tl.int64) * C + nn[None, :]
        else:
            o = (((b.to(tl.int64) * T2 + ot) * F2 + of) * C)[:, None] + nn[None, :]
        tl.store(Y + o, acc.to(tl.bfloat16), mask=mm[:, None])


# -------- conv2/3 (B): tiled over N (better for small problems) ---------
@triton.jit
def conv_kernel(X, W, Bi, Y,
                F_, T, F2, T2, M, C: tl.constexpr,
                OUT_MODE: tl.constexpr,
                BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    mm = m < M
    b = m // (F2 * T2)
    r = m % (F2 * T2)
    of = r // T2
    ot = r % T2
    n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    kc = tl.arange(0, BLOCK_K)
    for kf in tl.static_range(3):
        idx_f = 2 * of + kf - 1
        vf = (idx_f >= 0) & (idx_f < F_) & mm
        for kt in tl.static_range(3):
            idx_t = 2 * ot + kt - 1
            v = vf & (idx_t >= 0) & (idx_t < T)
            base = ((b.to(tl.int64) * F_ + idx_f) * T + idx_t) * C
            wbase = (kf * 3 + kt) * C * C
            for k0 in range(0, C, BLOCK_K):
                a = tl.load(X + base[:, None] + (k0 + kc)[None, :], mask=v[:, None], other=0.0)
                wt = tl.load(W + wbase + (k0 + kc)[:, None] * C + n[None, :])
                acc = tl.dot(a, wt, acc)

    acc += tl.load(Bi + n)[None, :].to(tl.float32)
    acc = acc.to(tl.bfloat16).to(tl.float32)
    acc = _gelu(acc)
    if OUT_MODE == 0:
        o = m[:, None].to(tl.int64) * C + n[None, :]
    else:
        o = (((b.to(tl.int64) * T2 + ot) * F2 + of) * C)[:, None] + n[None, :]
    tl.store(Y + o, acc.to(tl.bfloat16), mask=mm[:, None])


# ---------------- epilogue: scale + positional embedding ----------------
@triton.jit
def _epi(X, PE, Y, es, N, D: tl.constexpr, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    row = pid // (D // BLOCK)
    col = (pid % (D // BLOCK)) * BLOCK + tl.arange(0, BLOCK)
    o = row.to(tl.int64) * D + col
    x = tl.load(X + o).to(tl.float32) * es
    x = x.to(tl.bfloat16).to(tl.float32)
    p = tl.load(PE + (row % N).to(tl.int64) * D + col).to(tl.float32)
    tl.store(Y + o, (x + p).to(tl.bfloat16))


def _cfg(M):
    if M >= 200000:
        return (0, 0, 64, 8, 2)
    return (64, 128, 64, 4, 1)


@torch.no_grad()
def run(
    input_features: torch.Tensor,
    conv2d1_weight: torch.Tensor,
    conv2d1_bias: torch.Tensor,
    conv2d2_weight: torch.Tensor,
    conv2d2_bias: torch.Tensor,
    conv2d3_weight: torch.Tensor,
    conv2d3_bias: torch.Tensor,
    conv_out_weight: torch.Tensor,
    positional_embedding: torch.Tensor,
    embed_scale: float,
):
    B, _, Fm, T = input_features.shape
    C = conv2d1_weight.shape[0]
    dev = input_features.device

    # --- weight layouts ---
    w1 = torch.zeros((C, 16), device=dev, dtype=torch.bfloat16)
    w1[:, :9] = conv2d1_weight.reshape(C, 9)
    w2 = conv2d2_weight.permute(2, 3, 1, 0).contiguous()
    w3 = conv2d3_weight.permute(2, 3, 1, 0).contiguous()

    # --- stage 1 ---
    F1 = (Fm - 1) // 2 + 1
    T1 = (T - 1) // 2 + 1
    M1 = B * F1 * T1
    y1 = torch.empty((M1, C), device=dev, dtype=torch.bfloat16)
    conv1_kernel[(triton.cdiv(M1, 64), C // 128)](
        input_features, w1, conv2d1_bias, y1, T, T1, M1,
        BLOCK_M=64, BLOCK_N=128, num_warps=4, num_stages=2)

    # --- stage 2 ---
    F2 = (F1 - 1) // 2 + 1
    T2 = (T1 - 1) // 2 + 1
    M2 = B * F2 * T2
    y2 = torch.empty((M2, C), device=dev, dtype=torch.bfloat16)
    bm, bn, bk, nw, ns = _cfg(M2)
    if bm == 0:
        conv_kernel3[(triton.cdiv(M2, 128),)](
            y1, w2, conv2d2_bias, y2, F1, T1, F2, T2, M2, C, 0,
            BLOCK_M=128, BLOCK_K=bk, num_warps=nw, num_stages=ns)
    else:
        conv_kernel[(triton.cdiv(M2, bm), C // bn)](
            y1, w2, conv2d2_bias, y2, F1, T1, F2, T2, M2, C, 0,
            BLOCK_M=bm, BLOCK_N=bn, BLOCK_K=bk, num_warps=nw, num_stages=ns)

    # --- stage 3 (writes [B*T3, F3*C]) ---
    F3 = (F2 - 1) // 2 + 1
    T3 = (T2 - 1) // 2 + 1
    M3 = B * F3 * T3
    y3 = torch.empty((B * T3, F3 * C), device=dev, dtype=torch.bfloat16)
    bm, bn, bk, nw, ns = _cfg(M3)
    if bm == 0:
        conv_kernel3[(triton.cdiv(M3, 128),)](
            y2, w3, conv2d3_bias, y3, F2, T2, F3, T3, M3, C, 1,
            BLOCK_M=128, BLOCK_K=bk, num_warps=nw, num_stages=ns)
    else:
        conv_kernel[(triton.cdiv(M3, bm), C // bn)](
            y2, w3, conv2d3_bias, y3, F2, T2, F3, T3, M3, C, 1,
            BLOCK_M=bm, BLOCK_N=bn, BLOCK_K=bk, num_warps=nw, num_stages=ns)

    # --- linear (weight columns reordered to match f*C+c layout) ---
    D = conv_out_weight.shape[0]
    wo = conv_out_weight.view(D, C, F3).permute(0, 2, 1).reshape(D, F3 * C).contiguous()
    z = torch.mm(y3, wo.t())

    # --- scale + positional embedding ---
    out = torch.empty_like(z)
    BLK = 256
    _epi[(B * T3 * (D // BLK),)](z, positional_embedding, out, embed_scale, T3,
                                 D=D, BLOCK=BLK, num_warps=4)
    return out.view(B, T3, D)

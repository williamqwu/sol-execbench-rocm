import math
import torch
import triton
import triton.language as tl


@triton.jit
def _gelu(x):
    return x * 0.5 * (1.0 + tl.math.erf(x * 0.7071067811865476))


# ---------------- conv1: [B,1,80,T] -> NHWC [B,40,T2,384] ----------------
@triton.jit
def conv1_kernel(X, W, Bi, Y,
                 B, T, T2, M,
                 BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, N: tl.constexpr):
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
    wo = n[None, :] * 9 + k[:, None]
    w = tl.load(W + wo, mask=(k[:, None] < 9), other=0.0)

    acc = tl.dot(a, w, out_dtype=tl.float32)
    acc += tl.load(Bi + n)[None, :].to(tl.float32)
    acc = acc.to(tl.bfloat16).to(tl.float32)
    acc = _gelu(acc)
    tl.store(Y + m[:, None].to(tl.int64) * N + n[None, :], acc.to(tl.bfloat16), mask=mm[:, None])


def conv1(x, w, bias):
    B = x.shape[0]
    T = x.shape[3]
    T2 = (T - 1) // 2 + 1
    M = B * 40 * T2
    y = torch.empty((B, 40, T2, 384), device=x.device, dtype=torch.bfloat16)
    BLOCK_M = 64
    BLOCK_N = 128
    conv1_kernel[(triton.cdiv(M, BLOCK_M), 3)](x, w, bias, y, B, T, T2, M,
                                               BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, N=384,
                                               num_warps=4, num_stages=2)
    return y, T2


# -------- conv2/3 v2: one program covers all C outputs (3 n-tiles) -------
@triton.jit
def conv_kernel3(X, W, Bi, Y,
                 F, T, F2, T2, M, C: tl.constexpr,
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
        vf = (idx_f >= 0) & (idx_f < F) & mm
        for kt in tl.static_range(3):
            idx_t = 2 * ot + kt - 1
            v = vf & (idx_t >= 0) & (idx_t < T)
            base = ((b.to(tl.int64) * F + idx_f) * T + idx_t) * C
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


# ---------------- conv2/3: NHWC [B,F,T,C] -> NHWC [B,F2,T2,C] ------------
@triton.jit
def conv_kernel(X, W, Bi, Y,
                F, T, F2, T2, M, C: tl.constexpr,
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
        vf = (idx_f >= 0) & (idx_f < F) & mm
        for kt in tl.static_range(3):
            idx_t = 2 * ot + kt - 1
            v = vf & (idx_t >= 0) & (idx_t < T)
            base = ((b.to(tl.int64) * F + idx_f) * T + idx_t) * C
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
        # write [B, T2, F2*C] with index (b, ot, of*C + n)
        o = (((b.to(tl.int64) * T2 + ot) * F2 + of) * C)[:, None] + n[None, :]
    tl.store(Y + o, acc.to(tl.bfloat16), mask=mm[:, None])


def convn(x, wp, bias, F, T, out_mode, B, cfg):
    C = 384
    F2 = (F - 1) // 2 + 1
    T2 = (T - 1) // 2 + 1
    M = B * F2 * T2
    if out_mode == 0:
        y = torch.empty((B, F2, T2, C), device=x.device, dtype=torch.bfloat16)
    else:
        y = torch.empty((B, T2, C * F2), device=x.device, dtype=torch.bfloat16)
    BM, BN, BK, nw, ns = cfg
    if BN == 0:
        conv_kernel3[(triton.cdiv(M, BM),)](
            x, wp, bias, y, F, T, F2, T2, M, C, out_mode,
            BLOCK_M=BM, BLOCK_K=BK, num_warps=nw, num_stages=ns)
    else:
        conv_kernel[(triton.cdiv(M, BM), triton.cdiv(C, BN))](
            x, wp, bias, y, F, T, F2, T2, M, C, out_mode,
            BLOCK_M=BM, BLOCK_N=BN, BLOCK_K=BK, num_warps=nw, num_stages=ns)
    return y, F2, T2

import torch, triton
import triton.language as tl

H = 2048


@triton.jit
def gw_kernel(GO, R, GW, M,
              BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
              GROUP_M: tl.constexpr, N: tl.constexpr, EVEN_K: tl.constexpr):
    """GW[i,j] = sum_k GO[k,i] * R[k,j]   (i,j < N=2048, k < M)"""
    pid = tl.program_id(0)
    num_m: tl.constexpr = N // BLOCK_M
    num_n: tl.constexpr = N // BLOCK_N
    wid: tl.constexpr = GROUP_M * num_n
    gid = pid // wid
    first = gid * GROUP_M
    gsz = min(num_m - first, GROUP_M)
    pid_m = first + ((pid % wid) % gsz)
    pid_n = (pid % wid) // gsz

    offs_i = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_j = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    a_ptr = GO + offs_i[None, :] + offs_k[:, None] * N   # (BK, BM) = GO[k, i]
    b_ptr = R + offs_k[:, None] * N + offs_j[None, :]    # (BK, BN)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k in tl.range(0, tl.cdiv(M, BLOCK_K)):
        if EVEN_K:
            a = tl.load(a_ptr)
            b = tl.load(b_ptr)
        else:
            lim = M - k * BLOCK_K
            a = tl.load(a_ptr, mask=offs_k[:, None] < lim, other=0.0)
            b = tl.load(b_ptr, mask=offs_k[:, None] < lim, other=0.0)
        acc = tl.dot(tl.trans(a), b, acc)
        a_ptr += BLOCK_K * N
        b_ptr += BLOCK_K * N

    tl.store(GW + offs_i[:, None] * N + offs_j[None, :], acc.to(GW.dtype.element_ty))


@triton.jit
def ga_kernel(GO, W, OUT, M, S,
              BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
              GROUP_M: tl.constexpr, N: tl.constexpr, HD: tl.constexpr,
              NH: tl.constexpr, EVEN_M: tl.constexpr):
    """C[m,n] = sum_k GO[m,k]*W[k,n], stored as OUT[b, h, s, d]."""
    pid = tl.program_id(0)
    num_m = tl.cdiv(M, BLOCK_M)
    num_n: tl.constexpr = N // BLOCK_N
    wid: tl.constexpr = GROUP_M * num_n
    gid = pid // wid
    first = gid * GROUP_M
    gsz = min(num_m - first, GROUP_M)
    pid_m = first + ((pid % wid) % gsz)
    pid_n = (pid % wid) // gsz

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    rm = offs_m if EVEN_M else offs_m % M
    a_ptr = GO + rm[:, None] * N + offs_k[None, :]
    b_ptr = W + offs_k[:, None] * N + offs_n[None, :]

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k in tl.range(0, N // BLOCK_K):
        a = tl.load(a_ptr)
        b = tl.load(b_ptr)
        acc = tl.dot(a, b, acc)
        a_ptr += BLOCK_K
        b_ptr += BLOCK_K * N

    # OUT[b,h,s,d] with m = b*S+s, n = h*HD+d
    row = (offs_m // S) * (NH * S * HD) + (offs_m % S) * HD
    col = (offs_n // HD) * (S * HD) + (offs_n % HD)
    off = row[:, None] + col[None, :]
    c = acc.to(OUT.dtype.element_ty)
    if EVEN_M:
        tl.store(OUT + off, c)
    else:
        tl.store(OUT + off, c, mask=(offs_m < M)[:, None])


def gw_run(go2d, r2d, cfg, out=None):
    M = go2d.shape[0]
    if out is None:
        out = torch.empty((H, H), device=go2d.device, dtype=torch.bfloat16)
    BM, BN, BK, GM, nw, ns = cfg
    grid = ((H // BM) * (H // BN),)
    gw_kernel[grid](go2d, r2d, out, M, BM, BN, BK, GM, H, M % BK == 0,
                    num_warps=nw, num_stages=ns)
    return out


def ga_run(go2d, w, B, S, cfg, out=None):
    M = go2d.shape[0]
    if out is None:
        out = torch.empty((B, 32, S, 64), device=go2d.device, dtype=torch.bfloat16)
    BM, BN, BK, GM, nw, ns = cfg
    grid = (triton.cdiv(M, BM) * (H // BN),)
    ga_kernel[grid](go2d, w, out, M, S, BM, BN, BK, GM, H, 64, 32, M % BM == 0,
                    num_warps=nw, num_stages=ns)
    return out

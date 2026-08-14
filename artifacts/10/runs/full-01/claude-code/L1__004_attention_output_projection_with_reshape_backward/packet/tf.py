import torch, triton
import triton.language as tl

H = 2048


@triton.jit
def _gw_tile(GO, R, GW, M, pid,
             BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
             N: tl.constexpr, EVEN_K: tl.constexpr):
    num_n: tl.constexpr = N // BLOCK_N
    pid_m = pid // num_n
    pid_n = pid % num_n
    offs_i = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_j = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    a_ptr = GO + offs_i[None, :] + offs_k[:, None] * N
    b_ptr = R + offs_k[:, None] * N + offs_j[None, :]
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
def _ga_tile(GO, W, GA, M, S, p,
             BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
             N: tl.constexpr, HD: tl.constexpr, NH: tl.constexpr,
             GROUP_M: tl.constexpr, EVEN_M: tl.constexpr):
    num_n: tl.constexpr = N // BLOCK_N
    num_m = tl.cdiv(M, BLOCK_M)
    wid: tl.constexpr = GROUP_M * num_n
    gid = p // wid
    first = gid * GROUP_M
    gsz = min(num_m - first, GROUP_M)
    pid_m = first + ((p % wid) % gsz)
    pid_n = (p % wid) // gsz

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
    row = (offs_m // S) * (NH * S * HD) + (offs_m % S) * HD
    col = (offs_n // HD) * (S * HD) + (offs_n % HD)
    off = row[:, None] + col[None, :]
    c = acc.to(GA.dtype.element_ty)
    if EVEN_M:
        tl.store(GA + off, c)
    else:
        tl.store(GA + off, c, mask=(offs_m < M)[:, None])


@triton.jit
def fused_kernel(GO, R, W, GA, GW, M, S, NUM_GW,
                 BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
                 N: tl.constexpr, HD: tl.constexpr, NH: tl.constexpr,
                 GROUP_M: tl.constexpr, EVEN_K: tl.constexpr, EVEN_M: tl.constexpr):
    pid = tl.program_id(0)
    if pid < NUM_GW:
        _gw_tile(GO, R, GW, M, pid, BLOCK_M, BLOCK_N, BLOCK_K, N, EVEN_K)
    else:
        _ga_tile(GO, W, GA, M, S, pid - NUM_GW, BLOCK_M, BLOCK_N, BLOCK_K,
                 N, HD, NH, GROUP_M, EVEN_M)


def fused_run(go2d, r2d, w, B, S, cfg):
    M = go2d.shape[0]
    BM, BN, BK, GM, nw, ns = cfg
    ga = torch.empty((B, 32, S, 64), device=go2d.device, dtype=torch.bfloat16)
    gw = torch.empty((H, H), device=go2d.device, dtype=torch.bfloat16)
    num_gw = (H // BM) * (H // BN)
    num_ga = triton.cdiv(M, BM) * (H // BN)
    fused_kernel[(num_gw + num_ga,)](
        go2d, r2d, w, ga, gw, M, S, num_gw,
        BM, BN, BK, H, 64, 32, GM, M % BK == 0, M % BM == 0,
        num_warps=nw, num_stages=ns)
    return ga, gw

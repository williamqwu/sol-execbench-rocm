import torch, triton
import triton.language as tl

H = 2048


@triton.jit
def _gw_tile(GO, R, GW, M, pid,
             BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr,
             N: tl.constexpr, EVEN_K: tl.constexpr, GROUP: tl.constexpr):
    num_n: tl.constexpr = N // BN
    num_m: tl.constexpr = N // BM
    wid: tl.constexpr = GROUP * num_n
    gid = pid // wid
    first = gid * GROUP
    gsz = min(num_m - first, GROUP)
    pid_m = first + ((pid % wid) % gsz)
    pid_n = (pid % wid) // gsz

    offs_i = pid_m * BM + tl.arange(0, BM)
    offs_j = pid_n * BN + tl.arange(0, BN)
    offs_k = tl.arange(0, BK)

    a_ptr = GO + offs_i[None, :] + offs_k[:, None] * N
    b_ptr = R + offs_k[:, None] * N + offs_j[None, :]
    acc = tl.zeros((BM, BN), dtype=tl.float32)
    for k in tl.range(0, tl.cdiv(M, BK)):
        if EVEN_K:
            a = tl.load(a_ptr)
            b = tl.load(b_ptr)
        else:
            lim = M - k * BK
            a = tl.load(a_ptr, mask=offs_k[:, None] < lim, other=0.0)
            b = tl.load(b_ptr, mask=offs_k[:, None] < lim, other=0.0)
        acc = tl.dot(tl.trans(a), b, acc)
        a_ptr += BK * N
        b_ptr += BK * N
    tl.store(GW + offs_i[:, None] * N + offs_j[None, :], acc.to(GW.dtype.element_ty))


@triton.jit
def _ga_tile(GO, W, GA, M, S, p,
             BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr,
             N: tl.constexpr, HD: tl.constexpr, NH: tl.constexpr,
             GROUP: tl.constexpr, EVEN_M: tl.constexpr):
    num_n: tl.constexpr = N // BN
    num_m = tl.cdiv(M, BM)
    wid: tl.constexpr = GROUP * num_n
    gid = p // wid
    first = gid * GROUP
    gsz = min(num_m - first, GROUP)
    pid_m = first + ((p % wid) % gsz)
    pid_n = (p % wid) // gsz

    offs_m = pid_m * BM + tl.arange(0, BM)
    offs_n = pid_n * BN + tl.arange(0, BN)
    offs_k = tl.arange(0, BK)
    rm = offs_m if EVEN_M else offs_m % M
    a_ptr = GO + rm[:, None] * N + offs_k[None, :]
    b_ptr = W + offs_k[:, None] * N + offs_n[None, :]
    acc = tl.zeros((BM, BN), dtype=tl.float32)
    for k in tl.range(0, N // BK):
        a = tl.load(a_ptr)
        b = tl.load(b_ptr)
        acc = tl.dot(a, b, acc)
        a_ptr += BK
        b_ptr += BK * N
    row = (offs_m // S) * (NH * S * HD) + (offs_m % S) * HD
    col = (offs_n // HD) * (S * HD) + (offs_n % HD)
    off = row[:, None] + col[None, :]
    c = acc.to(GA.dtype.element_ty)
    if EVEN_M:
        tl.store(GA + off, c)
    else:
        tl.store(GA + off, c, mask=(offs_m < M)[:, None])


@triton.jit
def fused2(GO, R, W, GA, GW, M, S, NUM_GW,
           BM1: tl.constexpr, BN1: tl.constexpr, BK1: tl.constexpr, G1: tl.constexpr,
           BM2: tl.constexpr, BN2: tl.constexpr, BK2: tl.constexpr, G2: tl.constexpr,
           N: tl.constexpr, HD: tl.constexpr, NH: tl.constexpr,
           EVEN_K: tl.constexpr, EVEN_M: tl.constexpr):
    pid = tl.program_id(0)
    if pid < NUM_GW:
        _gw_tile(GO, R, GW, M, pid, BM1, BN1, BK1, N, EVEN_K, G1)
    else:
        _ga_tile(GO, W, GA, M, S, pid - NUM_GW, BM2, BN2, BK2, N, HD, NH, G2, EVEN_M)


def run2(go2d, r2d, w, B, S, c1, c2, nw, ns):
    M = go2d.shape[0]
    BM1, BN1, BK1, G1 = c1
    BM2, BN2, BK2, G2 = c2
    ga = torch.empty((B, 32, S, 64), device=go2d.device, dtype=torch.bfloat16)
    gw = torch.empty((H, H), device=go2d.device, dtype=torch.bfloat16)
    n_gw = (H // BM1) * (H // BN1)
    n_ga = triton.cdiv(M, BM2) * (H // BN2)
    fused2[(n_gw + n_ga,)](go2d, r2d, w, ga, gw, M, S, n_gw,
                           BM1, BN1, BK1, G1, BM2, BN2, BK2, G2,
                           H, 64, 32, M % BK1 == 0, M % BM2 == 0,
                           num_warps=nw, num_stages=ns)
    return ga, gw

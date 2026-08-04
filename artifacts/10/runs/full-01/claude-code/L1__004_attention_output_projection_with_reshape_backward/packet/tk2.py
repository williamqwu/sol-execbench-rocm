import torch, triton
import triton.language as tl

H = 2048


@triton.jit
def _gw_body(GO, R, GW, M, pid,
             BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr,
             N: tl.constexpr, EVEN_K: tl.constexpr, GROUP: tl.constexpr):
    """GW[i,j] = sum_k GO[k,i]*R[k,j]. Load GO tile as (BM,BK) directly (fast form)."""
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

    a_ptr = GO + offs_i[:, None] + offs_k[None, :] * N   # (BM,BK) = GO^T
    b_ptr = R + offs_k[:, None] * N + offs_j[None, :]    # (BK,BN)
    acc = tl.zeros((BM, BN), dtype=tl.float32)
    for k in tl.range(0, tl.cdiv(M, BK)):
        if EVEN_K:
            a = tl.load(a_ptr)
            b = tl.load(b_ptr)
        else:
            lim = M - k * BK
            a = tl.load(a_ptr, mask=offs_k[None, :] < lim, other=0.0)
            b = tl.load(b_ptr, mask=offs_k[:, None] < lim, other=0.0)
        acc = tl.dot(a, b, acc)
        a_ptr += BK * N
        b_ptr += BK * N
    tl.store(GW + offs_i[:, None] * N + offs_j[None, :], acc.to(GW.dtype.element_ty))


@triton.jit
def _ga_body(GO, W, GA, M, S, p,
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
def gw_k(GO, R, GW, M, BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr,
         N: tl.constexpr, EVEN_K: tl.constexpr, GROUP: tl.constexpr):
    _gw_body(GO, R, GW, M, tl.program_id(0), BM, BN, BK, N, EVEN_K, GROUP)


@triton.jit
def ga_k(GO, W, GA, M, S, BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr,
         N: tl.constexpr, HD: tl.constexpr, NH: tl.constexpr,
         GROUP: tl.constexpr, EVEN_M: tl.constexpr):
    _ga_body(GO, W, GA, M, S, tl.program_id(0), BM, BN, BK, N, HD, NH, GROUP, EVEN_M)


@triton.jit
def fused_k(GO, R, W, GA, GW, M, S, NUM_GW,
            BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr,
            N: tl.constexpr, HD: tl.constexpr, NH: tl.constexpr,
            GROUP: tl.constexpr, EVEN_K: tl.constexpr, EVEN_M: tl.constexpr):
    pid = tl.program_id(0)
    if pid < NUM_GW:
        _gw_body(GO, R, GW, M, pid, BM, BN, BK, N, EVEN_K, GROUP)
    else:
        _ga_body(GO, W, GA, M, S, pid - NUM_GW, BM, BN, BK, N, HD, NH, GROUP, EVEN_M)


def gw_run(go2d, r2d, cfg):
    M = go2d.shape[0]
    BM, BN, BK, G, nw, ns = cfg
    out = torch.empty((H, H), device=go2d.device, dtype=torch.bfloat16)
    gw_k[((H // BM) * (H // BN),)](go2d, r2d, out, M, BM, BN, BK, H,
                                   M % BK == 0, G, num_warps=nw, num_stages=ns)
    return out


def ga_run(go2d, w, B, S, cfg):
    M = go2d.shape[0]
    BM, BN, BK, G, nw, ns = cfg
    out = torch.empty((B, 32, S, 64), device=go2d.device, dtype=torch.bfloat16)
    ga_k[(triton.cdiv(M, BM) * (H // BN),)](go2d, w, out, M, S, BM, BN, BK, H, 64, 32,
                                            G, M % BM == 0, num_warps=nw, num_stages=ns)
    return out


def fused_run(go2d, r2d, w, B, S, cfg):
    M = go2d.shape[0]
    BM, BN, BK, G, nw, ns = cfg
    ga = torch.empty((B, 32, S, 64), device=go2d.device, dtype=torch.bfloat16)
    gw = torch.empty((H, H), device=go2d.device, dtype=torch.bfloat16)
    n_gw = (H // BM) * (H // BN)
    n_ga = triton.cdiv(M, BM) * (H // BN)
    fused_k[(n_gw + n_ga,)](go2d, r2d, w, ga, gw, M, S, n_gw, BM, BN, BK, H, 64, 32,
                            G, M % BK == 0, M % BM == 0, num_warps=nw, num_stages=ns)
    return ga, gw

import torch
import triton
import triton.language as tl

H = 2048
NH = 32
HD = 64


@triton.jit
def _gw_body(GO, R, GW, M, pid,
             BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr,
             N: tl.constexpr, EVEN_K: tl.constexpr, GROUP: tl.constexpr):
    """grad_weight[i, j] = sum_k GO[k, i] * R[k, j]   (fp32 accumulate)."""
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

    # GO is (M, N) row-major; this reads the (BM, BK) transposed tile GO[k, i].
    a_ptr = GO + offs_i[:, None] + offs_k[None, :] * N
    b_ptr = R + offs_k[:, None] * N + offs_j[None, :]

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
             N: tl.constexpr, HDc: tl.constexpr, NHc: tl.constexpr,
             GROUP: tl.constexpr, EVEN_M: tl.constexpr):
    """C[m, n] = sum_k GO[m, k] * W[k, n], scattered straight into (B, NH, S, HD)."""
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

    # m = b*S + s, n = h*HD + d  ->  GA[b, h, s, d]; the transpose is free here.
    row = (offs_m // S) * (NHc * S * HDc) + (offs_m % S) * HDc
    col = (offs_n // HDc) * (S * HDc) + (offs_n % HDc)
    off = row[:, None] + col[None, :]
    c = acc.to(GA.dtype.element_ty)
    if EVEN_M:
        tl.store(GA + off, c)
    else:
        tl.store(GA + off, c, mask=(offs_m < M)[:, None])


@triton.jit
def fused_k(GO, R, W, GA, GW, M, S, NUM_GW,
            BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr,
            N: tl.constexpr, HDc: tl.constexpr, NHc: tl.constexpr,
            GROUP: tl.constexpr, EVEN_K: tl.constexpr, EVEN_M: tl.constexpr):
    """Both GEMMs in one launch: tiles [0, NUM_GW) do grad_weight, the rest grad_attn."""
    pid = tl.program_id(0)
    if pid < NUM_GW:
        _gw_body(GO, R, GW, M, pid, BM, BN, BK, N, EVEN_K, GROUP)
    else:
        _ga_body(GO, W, GA, M, S, pid - NUM_GW, BM, BN, BK, N, HDc, NHc, GROUP, EVEN_M)


# M -> (BLOCK_M, BLOCK_N, BLOCK_K, GROUP, num_warps, num_stages), from a hardware sweep.
_TABLE = {
    256:  (64, 64, 64, 4, 4, 3),
    512:  (64, 64, 64, 1, 4, 3),
    1024: (128, 256, 64, 4, 8, 3),
    2048: (128, 256, 64, 4, 8, 3),
    4096: (128, 256, 64, 1, 8, 3),
    8192: (128, 256, 64, 4, 8, 3),
}


def _pick(M):
    cfg = _TABLE.get(M)
    if cfg is not None:
        return cfg
    if M <= 512:
        return (64, 64, 64, 1, 4, 3)
    if M <= 1024:
        return (64, 128, 64, 1, 4, 3)
    return (128, 256, 64, 4, 8, 3)


@torch.no_grad()
def run(
    grad_output: torch.Tensor,
    reshaped: torch.Tensor,
    weight: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    batch_size, seq_len, hidden_size = grad_output.shape

    go = grad_output.contiguous().view(-1, hidden_size)
    r = reshaped.contiguous().view(-1, hidden_size)
    w = weight.contiguous()
    M = go.shape[0]

    ga = torch.empty((batch_size, NH, seq_len, HD),
                     device=go.device, dtype=torch.bfloat16)
    gw = torch.empty((hidden_size, hidden_size),
                     device=go.device, dtype=torch.bfloat16)

    BM, BN, BK, G, nw, ns = _pick(M)
    n_gw = (hidden_size // BM) * (hidden_size // BN)
    n_ga = triton.cdiv(M, BM) * (hidden_size // BN)

    fused_k[(n_gw + n_ga,)](
        go, r, w, ga, gw, M, seq_len, n_gw,
        BM, BN, BK, hidden_size, HD, NH, G,
        M % BK == 0, M % BM == 0,
        num_warps=nw, num_stages=ns,
    )
    return ga, gw

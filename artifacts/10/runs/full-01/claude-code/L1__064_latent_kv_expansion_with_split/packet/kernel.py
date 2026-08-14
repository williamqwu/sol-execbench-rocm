import torch
import triton
import triton.language as tl


@triton.jit
def _rmsnorm_bf16(
    A, Wn, Out, M, eps,
    BM: tl.constexpr, K: tl.constexpr,
):
    pid = tl.program_id(0)
    rows = pid * BM + tl.arange(0, BM)
    mask = rows < M
    ko = tl.arange(0, K)
    off = rows[:, None] * K + ko[None, :]
    a = tl.load(A + off, mask=mask[:, None], other=0.0).to(tl.float32)
    scale = tl.rsqrt(tl.sum(a * a, 1) / K + eps)
    wn = tl.load(Wn + ko).to(tl.float32)
    tl.store(Out + off, (a * scale[:, None] * wn[None, :]).to(tl.bfloat16),
             mask=mask[:, None])


@triton.jit
def _gemm_split(
    A,          # [M, 512] bf16 (already normalized)
    Wb,         # [32768, 512] bf16
    O,          # [2, B, 128, S, 128] bf16
    M, S,
    half_stride,
    BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr,
    K: tl.constexpr, HD: tl.constexpr, GM: tl.constexpr,
    EVEN_M: tl.constexpr,
):
    pid = tl.program_id(0)
    num_pid_m = tl.cdiv(M, BM)
    num_pid_n = tl.cdiv(32768, BN)
    # grouped ordering for L2 reuse of the weight tiles
    num_in_group = GM * num_pid_n
    gid = pid // num_in_group
    first_m = gid * GM
    gsize = min(num_pid_m - first_m, GM)
    pid_m = first_m + ((pid % num_in_group) % gsize)
    pid_n = (pid % num_in_group) // gsize

    offs_m = pid_m * BM + tl.arange(0, BM)
    offs_n = pid_n * BN + tl.arange(0, BN)
    if not EVEN_M:
        offs_m = tl.where(offs_m < M, offs_m, 0)
    offs_m = tl.max_contiguous(tl.multiple_of(offs_m, BM), BM)
    offs_n = tl.max_contiguous(tl.multiple_of(offs_n, BN), BN)

    ko = tl.arange(0, BK)
    a_ptr = A + offs_m[:, None] * K + ko[None, :]
    b_ptr = Wb + offs_n[None, :] * K + ko[:, None]

    acc = tl.zeros((BM, BN), dtype=tl.float32)
    for k0 in tl.static_range(0, K, BK):
        a = tl.load(a_ptr + k0)
        bt = tl.load(b_ptr + k0)
        acc = tl.dot(a, bt, acc)

    out = acc.to(tl.bfloat16)

    # column c -> head c//256, is_v = (c%256)>=128, d = c%128
    head = offs_n // (2 * HD)
    rem = offs_n % (2 * HD)
    is_v = rem // HD
    d = rem % HD
    rm = pid_m * BM + tl.arange(0, BM)
    b_idx = rm // S
    s_idx = rm % S
    out_off = (is_v * half_stride + head * (S * HD))[None, :] \
        + (b_idx * (128 * S * HD) + s_idx * HD)[:, None] + d[None, :]
    if EVEN_M:
        tl.store(O + out_off, out)
    else:
        tl.store(O + out_off, out, mask=(rm < M)[:, None])


_HD = 128
_NH = 128


def _pick(M):
    # (BM, BN, BK, GM, num_warps, num_stages)
    if M <= 128:
        return (64, 128, 128, 8, 4, 2)
    if M <= 512:
        return (64, 256, 128, 8, 8, 2)
    if M <= 2048:
        return (128, 256, 128, 8, 8, 2)
    return (256, 256, 64, 8, 8, 2)


@torch.no_grad()
def run(
    compressed_kv: torch.Tensor,
    kv_a_layernorm_weight: torch.Tensor,
    kv_b_proj_weight: torch.Tensor,
    eps: float,
):
    bsz, seq_len, K = compressed_kv.shape
    M = bsz * seq_len

    a = compressed_kv.contiguous().view(M, K)
    wn = kv_a_layernorm_weight.contiguous()
    wb = kv_b_proj_weight.contiguous()

    an = torch.empty_like(a)
    BMN = 8 if M >= 8 else 1
    _rmsnorm_bf16[(triton.cdiv(M, BMN),)](
        a, wn, an, M, float(eps), BM=BMN, K=K, num_warps=8, num_stages=1,
    )

    out = torch.empty((2, bsz, _NH, seq_len, _HD), device=a.device, dtype=a.dtype)

    BM, BN, BK, GM, nw, ns = _pick(M)
    grid = (triton.cdiv(M, BM) * (32768 // BN),)
    _gemm_split[grid](
        an, wb, out, M, seq_len,
        bsz * _NH * seq_len * _HD,
        BM=BM, BN=BN, BK=BK, K=K, HD=_HD, GM=GM,
        EVEN_M=(M % BM == 0),
        num_warps=nw, num_stages=ns,
    )
    return out[0], out[1]

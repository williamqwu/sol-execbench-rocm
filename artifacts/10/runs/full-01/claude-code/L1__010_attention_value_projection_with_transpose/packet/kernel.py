import torch
import triton
import triton.language as tl


# ---------------------------------------------------------------------------
# Problem: value projection + reshape + transpose, fused.
#
#   hidden_states : [B, S, 5120]  bf16
#   v_proj_weight : [1024, 5120]  bf16
#   out           : [B, 8, S, 128] bf16   (out[b,h,s,d] = sum_k A[b,s,k]*W[h*128+d,k])
#
# The GEMM is [M, 5120] x [5120, 1024] with M = B*S.  N = 1024 is small, so for
# small M there is not enough tile parallelism to fill 256 CUs and the kernel
# becomes latency/occupancy bound while the 10 MiB weight read dominates.
# Split-K fixes that: partition the K=5120 reduction across SPLIT programs,
# write fp32 partials, then reduce.  The reduction pass also performs the
# reshape+transpose scatter, so the transpose is free.
#
# For large M there is already ample parallelism and the extra workspace
# round-trip costs more than it saves, so we use a single fused pass that
# scatters straight to the transposed layout.
# ---------------------------------------------------------------------------


@triton.jit
def _partial_k(
    A, W, WS,
    M,
    K: tl.constexpr,
    SPLIT: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_K: tl.constexpr,
    EVEN_M: tl.constexpr,
):
    """Split-K partial products -> WS[SPLIT, M, 1024] fp32."""
    BLOCK_N: tl.constexpr = 128
    KS: tl.constexpr = K // SPLIT

    pid = tl.program_id(0)
    pid_k = pid % SPLIT
    r = pid // SPLIT
    pid_n = r % 8
    pid_m = r // 8

    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    rk = pid_k * KS + tl.arange(0, BLOCK_K)

    rma = rm if EVEN_M else tl.where(rm < M, rm, 0)
    a_ptrs = A + rma[:, None] * K + rk[None, :]
    w_ptrs = W + rn[:, None] * K + rk[None, :]

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for _ in range(0, KS, BLOCK_K):
        acc = tl.dot(tl.load(a_ptrs), tl.trans(tl.load(w_ptrs)), acc)
        a_ptrs += BLOCK_K
        w_ptrs += BLOCK_K

    o = WS + pid_k * (M * 1024) + rm[:, None] * 1024 + rn[None, :]
    if EVEN_M:
        tl.store(o, acc)
    else:
        tl.store(o, acc, mask=(rm < M)[:, None])


@triton.jit
def _reduce_scatter(
    WS, C,
    M, S,
    SPLIT: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """Sum the SPLIT fp32 partials and scatter into [B,8,S,128] bf16."""
    pid = tl.program_id(0)
    off = pid * BLOCK + tl.arange(0, BLOCK)
    total = M * 1024
    mask = off < total

    m = off // 1024
    n = off - m * 1024

    acc = tl.zeros((BLOCK,), dtype=tl.float32)
    for i in tl.static_range(SPLIT):
        acc += tl.load(WS + i * total + off, mask=mask, other=0.0)

    b = m // S
    s = m - b * S
    h = n // 128
    d = n - h * 128
    tl.store(
        C + b * (8 * S * 128) + h * (S * 128) + s * 128 + d,
        acc.to(C.dtype.element_ty),
        mask=mask,
    )


@triton.jit
def _fused(
    A, W, C,
    M, S,
    BLOCK_M: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
    EVEN_M: tl.constexpr,
):
    """Single-pass GEMM writing directly to the transposed layout."""
    K: tl.constexpr = 5120
    BLOCK_N: tl.constexpr = 128

    pid = tl.program_id(0)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    nig = GROUP_M * 8
    gid = pid // nig
    fm = gid * GROUP_M
    gsm = min(num_pid_m - fm, GROUP_M)
    pid_m = fm + ((pid % nig) % gsm)
    pid_n = (pid % nig) // gsm

    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    rk = tl.arange(0, BLOCK_K)

    rma = rm if EVEN_M else tl.where(rm < M, rm, 0)
    a_ptrs = A + rma[:, None] * K + rk[None, :]
    w_ptrs = W + rn[:, None] * K + rk[None, :]

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for _ in range(0, K, BLOCK_K):
        acc = tl.dot(tl.load(a_ptrs), tl.trans(tl.load(w_ptrs)), acc)
        a_ptrs += BLOCK_K
        w_ptrs += BLOCK_K

    out = acc.to(C.dtype.element_ty)
    b = rm // S
    s = rm - b * S
    cp = (
        C
        + b[:, None] * (8 * S * 128)
        + pid_n * (S * 128)
        + s[:, None] * 128
        + tl.arange(0, 128)[None, :]
    )
    if EVEN_M:
        tl.store(cp, out)
    else:
        tl.store(cp, out, mask=(rm < M)[:, None])


# --- configuration chosen from an on-device sweep over both strategies -------
# (SPLIT, BLOCK_M, BLOCK_K, num_warps, num_stages)
_SPLIT_CFG = [
    (256,  (8, 64, 64, 8, 3)),
    (512,  (4, 64, 64, 8, 3)),
    (1024, (4, 128, 64, 8, 3)),
    (2048, (2, 128, 64, 8, 3)),
]
# (BLOCK_M, BLOCK_K, GROUP_M, num_warps, num_stages)
_FUSED_CFG = (128, 64, 8, 8, 3)


def _split_cfg(M):
    for lim, cfg in _SPLIT_CFG:
        if M <= lim:
            return cfg
    return None


@torch.no_grad()
def run(hidden_states: torch.Tensor, v_proj_weight: torch.Tensor) -> torch.Tensor:
    batch_size, seq_len, hidden_size = hidden_states.shape
    M = batch_size * seq_len

    a = hidden_states.reshape(M, hidden_size)
    if not a.is_contiguous():
        a = a.contiguous()
    w = v_proj_weight
    if not w.is_contiguous():
        w = w.contiguous()

    c = torch.empty(
        (batch_size, 8, seq_len, 128),
        device=hidden_states.device,
        dtype=hidden_states.dtype,
    )

    cfg = _split_cfg(M)
    if cfg is not None:
        SPLIT, BM, BK, nw, ns = cfg
        ws = torch.empty(SPLIT * M * 1024, device=a.device, dtype=torch.float32)
        _partial_k[(triton.cdiv(M, BM) * 8 * SPLIT,)](
            a, w, ws, M, 5120, SPLIT, BM, BK, M % BM == 0,
            num_warps=nw, num_stages=ns,
        )
        BL = 1024
        _reduce_scatter[(triton.cdiv(M * 1024, BL),)](
            ws, c, M, seq_len, SPLIT, BL, num_warps=4,
        )
    else:
        BM, BK, GM, nw, ns = _FUSED_CFG
        _fused[(triton.cdiv(M, BM) * 8,)](
            a, w, c, M, seq_len, BM, BK, GM, M % BM == 0,
            num_warps=nw, num_stages=ns,
        )
    return c

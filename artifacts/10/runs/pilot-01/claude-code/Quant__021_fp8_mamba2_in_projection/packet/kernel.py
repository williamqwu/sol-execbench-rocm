import torch
import triton
import triton.language as tl

E4M3_MAX = tl.constexpr(448.0)
RECIP_MAX = tl.constexpr(0.0022321429569274187)
FP8 = tl.constexpr(tl.float8e4nv)


# ---------------------------------------------------------------------------
# Activation quantization: BlockWise1x128  (1 row x 128 cols per scale block)
# ---------------------------------------------------------------------------
@triton.jit
def _quant_act_kernel(
    X, QX, SX,
    M, K,
    stride_xm, stride_qm, stride_sm,
    BLOCK_M: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_k = tl.program_id(1)

    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rk = pid_k * 128 + tl.arange(0, 128)
    mask = rm[:, None] < M

    x = tl.load(X + rm[:, None] * stride_xm + rk[None, :], mask=mask, other=0.0)
    x = x.to(tl.float32)

    amax = tl.max(tl.abs(x), 1)
    s = amax * RECIP_MAX
    s = tl.maximum(s, 1e-12)

    q = tl.fdiv(x, tl.broadcast_to(s[:, None], x.shape), ieee_rounding=True)
    q = tl.minimum(tl.maximum(q, -E4M3_MAX), E4M3_MAX)

    tl.store(QX + rm[:, None] * stride_qm + rk[None, :], q.to(FP8), mask=mask)
    tl.store(SX + rm * stride_sm + pid_k, s, mask=rm < M)


# ---------------------------------------------------------------------------
# Weight quantization: BlockWise128x128
# ---------------------------------------------------------------------------
@triton.jit
def _quant_w_kernel(
    W, QW, SW,
    N, K,
    stride_wn, stride_qn, stride_sn,
):
    pid_n = tl.program_id(0)
    pid_k = tl.program_id(1)

    rk = pid_k * 128 + tl.arange(0, 128)
    rn = pid_n * 128 + tl.arange(0, 128)

    w = tl.load(W + rn[:, None] * stride_wn + rk[None, :]).to(tl.float32)
    m = tl.max(tl.abs(w))
    s = tl.maximum(m * RECIP_MAX, 1e-12)

    q = tl.fdiv(w, tl.full(w.shape, 1.0, tl.float32) * s, ieee_rounding=True)
    q = tl.minimum(tl.maximum(q, -E4M3_MAX), E4M3_MAX)

    tl.store(QW + rn[:, None] * stride_qn + rk[None, :], q.to(FP8))
    tl.store(SW + pid_n * stride_sn + pid_k, s)


# ---------------------------------------------------------------------------
# Block-scaled FP8 GEMM:  C[M,N] = sum_kb sa[m,kb]*sb[nb,kb] * (QA @ QB^T)
# ---------------------------------------------------------------------------
@triton.jit
def _gemm_kernel(
    A, B, C, SA, SB,
    M, N, K,
    stride_am, stride_bn, stride_cm,
    stride_sam, stride_sbn,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    GROUP_M: tl.constexpr,
    NUM_KB: tl.constexpr,
):
    BLOCK_K: tl.constexpr = 128

    pid = tl.program_id(0)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    num_pid_in_group = GROUP_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    rm = (pid_m * BLOCK_M + tl.arange(0, BLOCK_M)) % M
    rn = (pid_n * BLOCK_N + tl.arange(0, BLOCK_N)) % N
    rk = tl.arange(0, BLOCK_K)

    a_ptrs = A + rm[:, None] * stride_am + rk[None, :]
    b_ptrs = B + rn[:, None] * stride_bn + rk[None, :]
    sa_ptrs = SA + rm * stride_sam
    sb_ptrs = SB + (rn // 128) * stride_sbn

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for kb in tl.range(0, NUM_KB):
        a = tl.load(a_ptrs)
        b = tl.load(b_ptrs)
        sa = tl.load(sa_ptrs + kb)
        sb = tl.load(sb_ptrs + kb)
        p = tl.dot(a, tl.trans(b))
        acc += p * (sa[:, None] * sb[None, :])
        a_ptrs += BLOCK_K
        b_ptrs += BLOCK_K

    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    mask = (rm[:, None] < M) & (rn[None, :] < N)
    tl.store(C + rm[:, None] * stride_cm + rn[None, :], acc.to(tl.bfloat16), mask=mask)


def _cfg(M):
    if M <= 128:
        return dict(BLOCK_M=128, BLOCK_N=64, GROUP_M=8, num_warps=4, num_stages=3)
    if M <= 256:
        return dict(BLOCK_M=128, BLOCK_N=128, GROUP_M=8, num_warps=8, num_stages=2)
    if M <= 2048:
        return dict(BLOCK_M=128, BLOCK_N=64, GROUP_M=4, num_warps=4, num_stages=2)
    return dict(BLOCK_M=64, BLOCK_N=256, GROUP_M=1, num_warps=4, num_stages=1)


@torch.no_grad()
def run(hidden_states: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    M, K = hidden_states.shape
    N, _ = weight.shape
    dev = hidden_states.device
    nkb = K // 128
    nnb = N // 128

    qx = torch.empty((M, K), dtype=torch.float8_e4m3fn, device=dev)
    sx = torch.empty((M, nkb), dtype=torch.float32, device=dev)
    qw = torch.empty((N, K), dtype=torch.float8_e4m3fn, device=dev)
    sw = torch.empty((nnb, nkb), dtype=torch.float32, device=dev)

    BM = 32
    _quant_act_kernel[(triton.cdiv(M, BM), nkb)](
        hidden_states, qx, sx, M, K,
        hidden_states.stride(0), qx.stride(0), sx.stride(0),
        BLOCK_M=BM, num_warps=4,
    )
    _quant_w_kernel[(nnb, nkb)](
        weight, qw, sw, N, K,
        weight.stride(0), qw.stride(0), sw.stride(0),
        num_warps=4,
    )

    out = torch.empty((M, N), dtype=torch.bfloat16, device=dev)
    cfg = _cfg(M)
    grid = (triton.cdiv(M, cfg["BLOCK_M"]) * triton.cdiv(N, cfg["BLOCK_N"]),)
    _gemm_kernel[grid](
        qx, qw, out, sx, sw,
        M, N, K,
        qx.stride(0), qw.stride(0), out.stride(0),
        sx.stride(0), sw.stride(0),
        BLOCK_M=cfg["BLOCK_M"], BLOCK_N=cfg["BLOCK_N"], GROUP_M=cfg["GROUP_M"],
        NUM_KB=nkb,
        num_warps=cfg["num_warps"], num_stages=cfg["num_stages"],
    )
    return out

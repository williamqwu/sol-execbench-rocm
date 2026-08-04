import torch
import triton
import triton.language as tl

E4M3_MAX = tl.constexpr(448.0)
# torch lowers `t / 448.0` (python scalar) to a reciprocal-multiply; mirror it exactly.
E4M3_MAX_RECIP = tl.constexpr(1.0 / 448.0)
FP8 = tl.constexpr(tl.float8e4nv)

HIDDEN = tl.constexpr(1536)
QKV_OUT = tl.constexpr(4608)
NUM_HEADS = 16
HEAD_DIM = 96
_HIDDEN = 1536
_QKV_OUT = 4608


# ---------------------------------------------------------------------------
# Activation quantisation: BlockWise1x128 along K
# ---------------------------------------------------------------------------
@triton.jit
def _quant_act(
    x_ptr, qx_ptr, sx_ptr,
    M,
    stride_xm, stride_qm, stride_sm,
    BLOCK_M: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_k = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_k = pid_k * 128 + tl.arange(0, 128)
    mask_m = offs_m < M

    x = tl.load(
        x_ptr + offs_m[:, None] * stride_xm + offs_k[None, :],
        mask=mask_m[:, None],
        other=0.0,
    ).to(tl.float32)

    amax = tl.max(tl.abs(x), axis=1)
    scale = amax * E4M3_MAX_RECIP
    scale = tl.maximum(scale, 1e-12)

    q = x / scale[:, None]
    q = tl.minimum(tl.maximum(q, -E4M3_MAX), E4M3_MAX)

    tl.store(
        qx_ptr + offs_m[:, None] * stride_qm + offs_k[None, :],
        q.to(FP8),
        mask=mask_m[:, None],
    )
    tl.store(sx_ptr + offs_m * stride_sm + pid_k, scale, mask=mask_m)


# ---------------------------------------------------------------------------
# Weight quantisation: BlockWise128x128 on w^T (K, N) == 128x128 tiles of w (N, K)
# ---------------------------------------------------------------------------
@triton.jit
def _quant_w(
    w_ptr, qw_ptr, sw_ptr,
    stride_wn, stride_qn, stride_sn,
):
    pid_n = tl.program_id(0)
    pid_k = tl.program_id(1)

    offs_n = pid_n * 128 + tl.arange(0, 128)
    offs_k = pid_k * 128 + tl.arange(0, 128)

    w = tl.load(w_ptr + offs_n[:, None] * stride_wn + offs_k[None, :]).to(tl.float32)

    amax = tl.max(tl.abs(w))
    scale = amax * E4M3_MAX_RECIP
    scale = tl.maximum(scale, 1e-12)

    q = w / scale
    q = tl.minimum(tl.maximum(q, -E4M3_MAX), E4M3_MAX)

    tl.store(qw_ptr + offs_n[:, None] * stride_qn + offs_k[None, :], q.to(FP8))
    tl.store(sw_ptr + pid_n * stride_sn + pid_k, scale)


# ---------------------------------------------------------------------------
# Blockwise-scaled FP8 GEMM with bias, writing straight into q / k / v
# ---------------------------------------------------------------------------
@triton.jit
def _gemm(
    qx_ptr, qw_ptr, sx_ptr, sw_ptr, bias_ptr,
    q_ptr,
    M,
    stride_am, stride_bn, stride_sxm, stride_swn, stride_om,
    NB_PER_OUT: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    GROUP_M: tl.constexpr,
    NUM_KB: tl.constexpr,
):
    pid = tl.program_id(0)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = QKV_OUT // BLOCK_N

    # grouped ordering for L2 reuse
    num_pid_in_group = GROUP_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    mask_m = offs_m < M
    offs_am = tl.where(mask_m, offs_m, 0)

    offs_k = tl.arange(0, 128)

    a_ptrs = qx_ptr + offs_am[:, None] * stride_am + offs_k[None, :]
    # (128, BLOCK_N) tile of W^T, loaded directly from the (N, K) weight
    b_ptrs = qw_ptr + offs_k[:, None] + offs_n[None, :] * stride_bn

    # weight scale block index along N (BLOCK_N is a multiple of 128)
    offs_swn = pid_n * (BLOCK_N // 128) + tl.arange(0, BLOCK_N // 128)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for kb in tl.range(0, NUM_KB):
        a = tl.load(a_ptrs)
        b = tl.load(b_ptrs)
        p = tl.dot(a, b, out_dtype=tl.float32)

        sx = tl.load(sx_ptr + offs_am * stride_sxm + kb)
        sw = tl.load(sw_ptr + offs_swn * stride_swn + kb)
        scl = sx[:, None] * tl.reshape(
            tl.broadcast_to(sw[:, None], (BLOCK_N // 128, 128)), (BLOCK_N,)
        )[None, :]
        acc += p * scl

        a_ptrs += 128
        b_ptrs += 128

    bias = tl.load(bias_ptr + offs_n).to(tl.float32)
    acc += bias[None, :]
    out = acc.to(tl.bfloat16)

    # q / k / v are the three slices of one (3, M, 16, 96) buffer
    sel = pid_n // NB_PER_OUT
    offs_on = offs_n - sel * HIDDEN
    base = q_ptr + sel * M * HIDDEN

    tl.store(
        base + offs_m[:, None] * stride_om + offs_on[None, :],
        out,
        mask=mask_m[:, None],
    )


def _cfg(M):
    if M <= 256:
        return dict(BLOCK_M=64, BLOCK_N=128, GROUP_M=1, num_warps=4, num_stages=2)
    if M <= 1024:
        return dict(BLOCK_M=128, BLOCK_N=128, GROUP_M=4, num_warps=8, num_stages=2)
    return dict(BLOCK_M=128, BLOCK_N=256, GROUP_M=8, num_warps=8, num_stages=2)


@torch.no_grad()
def run(hidden_states, qkv_weight, qkv_bias):
    M = hidden_states.shape[0]
    dev = hidden_states.device
    K = _HIDDEN
    N = _QKV_OUT
    num_kb = K // 128

    qx = torch.empty((M, K), dtype=torch.float8_e4m3fn, device=dev)
    sx = torch.empty((M, num_kb), dtype=torch.float32, device=dev)
    qw = torch.empty((N, K), dtype=torch.float8_e4m3fn, device=dev)
    sw = torch.empty((N // 128, num_kb), dtype=torch.float32, device=dev)

    BM_Q = 32
    _quant_act[(triton.cdiv(M, BM_Q), num_kb)](
        hidden_states, qx, sx,
        M,
        hidden_states.stride(0), qx.stride(0), sx.stride(0),
        BLOCK_M=BM_Q, num_warps=4, num_stages=1,
    )

    _quant_w[(N // 128, num_kb)](
        qkv_weight, qw, sw,
        qkv_weight.stride(0), qw.stride(0), sw.stride(0),
        num_warps=8, num_stages=1,
    )

    out = torch.empty((3, M, NUM_HEADS, HEAD_DIM), dtype=torch.bfloat16, device=dev)
    q, k, v = out[0], out[1], out[2]

    cfg = _cfg(M)
    BLOCK_M = cfg["BLOCK_M"]
    BLOCK_N = cfg["BLOCK_N"]
    grid = (triton.cdiv(M, BLOCK_M) * (N // BLOCK_N),)
    _gemm[grid](
        qx, qw, sx, sw, qkv_bias,
        out,
        M,
        qx.stride(0), qw.stride(0), sx.stride(0), sw.stride(0), K,
        NB_PER_OUT=_HIDDEN // BLOCK_N,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        GROUP_M=cfg["GROUP_M"],
        NUM_KB=num_kb,
        num_warps=cfg["num_warps"],
        num_stages=cfg["num_stages"],
    )

    return q, k, v

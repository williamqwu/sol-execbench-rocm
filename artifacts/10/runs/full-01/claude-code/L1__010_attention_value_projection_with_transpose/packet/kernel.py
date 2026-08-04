import torch
import triton
import triton.language as tl


# out[b, h, s, d] = sum_k hidden[b, s, k] * w[h*128+d, k]
#
# hidden : [B, S, K]  bf16, contiguous
# w      : [N, K]     bf16, contiguous   (N = 1024 = 8 * 128)
# out    : [B, 8, S, 128] bf16, contiguous
#
# The transpose is folded into the GEMM epilogue: nothing but the GEMM's own
# output is ever written, so the reshape+transpose+contiguous of the reference
# costs zero extra memory traffic.


@triton.jit
def _vproj_kernel(
    a_ptr, w_ptr, o_ptr,
    S,
    K: tl.constexpr,
    N: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    BM: tl.constexpr,
    BN: tl.constexpr,
    BK: tl.constexpr,
    GROUP_M: tl.constexpr,
    EVEN_M: tl.constexpr,
):
    pid = tl.program_id(0)
    bid = tl.program_id(1)

    num_pid_m = tl.cdiv(S, BM)
    num_pid_n: tl.constexpr = N // BN

    # grouped ordering for L2 reuse of the weight tiles
    num_pid_in_group = GROUP_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    offs_m = pid_m * BM + tl.arange(0, BM)
    offs_n = pid_n * BN + tl.arange(0, BN)
    offs_k = tl.arange(0, BK)

    if EVEN_M:
        offs_am = offs_m
    else:
        offs_am = tl.where(offs_m < S, offs_m, 0)

    a_ptrs = a_ptr + bid.to(tl.int64) * (S * K) + offs_am[:, None] * K + offs_k[None, :]
    w_ptrs = w_ptr + offs_n[:, None] * K + offs_k[None, :]

    acc = tl.zeros((BM, BN), dtype=tl.float32)
    for k in tl.range(0, K // BK):
        a = tl.load(a_ptrs)
        w = tl.load(w_ptrs)
        acc = tl.dot(a, tl.trans(w), acc)
        a_ptrs += BK
        w_ptrs += BK

    o = acc.to(o_ptr.dtype.element_ty)

    h = offs_n // HEAD_DIM
    d = offs_n % HEAD_DIM
    o_ptrs = (
        o_ptr
        + bid.to(tl.int64) * (N * S)
        + h[None, :] * (HEAD_DIM * S)
        + offs_m[:, None] * HEAD_DIM
        + d[None, :]
    )
    if EVEN_M:
        tl.store(o_ptrs, o)
    else:
        tl.store(o_ptrs, o, mask=offs_m[:, None] < S)


@triton.jit
def _vproj_splitk_kernel(
    a_ptr, w_ptr, o_ptr,
    S,
    K: tl.constexpr,
    N: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    BM: tl.constexpr,
    BN: tl.constexpr,
    BK: tl.constexpr,
    SPLIT_K: tl.constexpr,
    EVEN_M: tl.constexpr,
):
    pid = tl.program_id(0)
    bid = tl.program_id(1)
    pid_k = tl.program_id(2)

    num_pid_n: tl.constexpr = N // BN
    pid_m = pid // num_pid_n
    pid_n = pid % num_pid_n

    offs_m = pid_m * BM + tl.arange(0, BM)
    offs_n = pid_n * BN + tl.arange(0, BN)
    offs_k = pid_k * (K // SPLIT_K) + tl.arange(0, BK)

    if EVEN_M:
        offs_am = offs_m
    else:
        offs_am = tl.where(offs_m < S, offs_m, 0)

    a_ptrs = a_ptr + bid.to(tl.int64) * (S * K) + offs_am[:, None] * K + offs_k[None, :]
    w_ptrs = w_ptr + offs_n[:, None] * K + offs_k[None, :]

    acc = tl.zeros((BM, BN), dtype=tl.float32)
    for k in tl.range(0, K // (BK * SPLIT_K)):
        a = tl.load(a_ptrs)
        w = tl.load(w_ptrs)
        acc = tl.dot(a, tl.trans(w), acc)
        a_ptrs += BK
        w_ptrs += BK

    h = offs_n // HEAD_DIM
    d = offs_n % HEAD_DIM
    o_ptrs = (
        o_ptr
        + bid.to(tl.int64) * (N * S)
        + h[None, :] * (HEAD_DIM * S)
        + offs_m[:, None] * HEAD_DIM
        + d[None, :]
    )
    if EVEN_M:
        tl.atomic_add(o_ptrs, acc, sem="relaxed")
    else:
        tl.atomic_add(o_ptrs, acc, mask=offs_m[:, None] < S, sem="relaxed")


@triton.jit
def _cast_kernel(src, dst, n_elem, BLK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLK + tl.arange(0, BLK)
    m = offs < n_elem
    v = tl.load(src + offs, mask=m, other=0.0)
    tl.store(dst + offs, v.to(dst.dtype.element_ty), mask=m)


_HEAD_DIM = 128
_NUM_KV = 8


def _pick(M):
    # (BM, BN, BK, GROUP_M, num_warps, num_stages)
    if M <= 256:
        return (64, 128, 128, 8, 4, 2)
    if M <= 1024:
        return (128, 128, 64, 8, 4, 2)
    if M <= 4096:
        return (128, 256, 64, 8, 8, 2)
    return (256, 256, 64, 8, 8, 2)


@torch.no_grad()
def run(hidden_states: torch.Tensor, v_proj_weight: torch.Tensor) -> torch.Tensor:
    B, S, K = hidden_states.shape
    N = v_proj_weight.shape[0]

    hidden_states = hidden_states.contiguous()
    v_proj_weight = v_proj_weight.contiguous()

    out = torch.empty((B, _NUM_KV, S, _HEAD_DIM), dtype=hidden_states.dtype,
                      device=hidden_states.device)

    BM, BN, BK, GM, nw, ns = _pick(B * S)
    grid = (triton.cdiv(S, BM) * (N // BN), B)
    _vproj_kernel[grid](
        hidden_states, v_proj_weight, out,
        S, K, N, _HEAD_DIM,
        BM=BM, BN=BN, BK=BK, GROUP_M=GM,
        EVEN_M=(S % BM == 0),
        num_warps=nw, num_stages=ns,
    )
    return out

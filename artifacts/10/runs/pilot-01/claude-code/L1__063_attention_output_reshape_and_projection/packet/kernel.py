"""Fused attention-output reshape + output projection for MI355X (gfx950).

reference:
    x = attn_output.transpose(1, 2).reshape(B, S, H*D)   # materialised copy
    out = x @ o_proj_weight.t()

We fuse the transpose/reshape straight into the GEMM's A-operand addressing, so
the [B, S, H*D] intermediate (M*K*2 bytes written *and* read back) never exists.

A logical row m == (b, s); a logical k == (h, d) with D == 128 a power of two,
so h = k >> 7 and d = k & 127 are shifts.  A[m, k] lives at
    b*stride_ab + h*stride_ah + s*stride_as + d
which is affine in (m, k) once split that way -> ordinary Triton pointer math,
and each row of a K-tile is a run of >=64 contiguous bf16 (>=128 B) so the loads
stay coalesced.

Accumulation is fp32 (as torch's bf16 matmul does); only the final store rounds
to bf16.
"""

import torch
import triton
import triton.language as tl


# --------------------------------------------------------------------------- #
# fused reshape + GEMM
# --------------------------------------------------------------------------- #
@triton.jit
def _fused_proj_kernel(
    A_ptr, W_ptr, C_ptr,
    M, N, S,
    stride_ab, stride_ah, stride_as,
    stride_wn,
    stride_cm,
    K: tl.constexpr,
    D: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    SPLIT_K: tl.constexpr,
    GROUP_M: tl.constexpr,
    ONE_BATCH: tl.constexpr,
    EVEN_M: tl.constexpr,
):
    pid = tl.program_id(0)
    pid_k = tl.program_id(1)

    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)

    # grouped ordering: keeps the weight tiles hot in LLC across neighbouring
    # m-blocks.
    num_pid_in_group = GROUP_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    if EVEN_M:
        m_mask = tl.full((BLOCK_M,), 1, tl.int1)
        rm = offs_m
    else:
        m_mask = offs_m < M
        rm = tl.where(m_mask, offs_m, 0)

    # row base:  m -> (b, s)
    if ONE_BATCH:
        a_row = rm * stride_as
    else:
        b = rm // S
        s = rm % S
        a_row = b * stride_ab + s * stride_as

    K_SPLIT: tl.constexpr = K // SPLIT_K
    k0 = pid_k * K_SPLIT

    offs_k = k0 + tl.arange(0, BLOCK_K)
    # k -> (h, d)
    a_col = (offs_k // D) * stride_ah + (offs_k % D)

    a_ptrs = A_ptr + a_row[:, None] + a_col[None, :]
    w_ptrs = W_ptr + offs_n[:, None] * stride_wn + offs_k[None, :]

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for _ in tl.range(0, K_SPLIT // BLOCK_K, num_stages=2):
        if EVEN_M:
            a = tl.load(a_ptrs)
        else:
            a = tl.load(a_ptrs, mask=m_mask[:, None], other=0.0)
        w = tl.load(w_ptrs)
        acc = tl.dot(a, tl.trans(w), acc)

        offs_k += BLOCK_K
        a_col = (offs_k // D) * stride_ah + (offs_k % D)
        a_ptrs = A_ptr + a_row[:, None] + a_col[None, :]
        w_ptrs += BLOCK_K

    c_ptrs = C_ptr + offs_m[:, None] * stride_cm + offs_n[None, :]
    if SPLIT_K == 1:
        tl.store(c_ptrs, acc.to(C_ptr.dtype.element_ty), mask=m_mask[:, None])
    else:
        tl.atomic_add(c_ptrs, acc, mask=m_mask[:, None], sem="relaxed")


# --------------------------------------------------------------------------- #
# config selection
# --------------------------------------------------------------------------- #
# (BLOCK_M, BLOCK_N, BLOCK_K, SPLIT_K, GROUP_M, num_warps, num_stages)
_DEFAULT = (128, 128, 128, 1, 8, 8, 2)

_TABLE = (
    # (m_upper_bound, cfg)
    (160, (128, 128, 128, 8, 8, 8, 2)),
    (320, (128, 128, 128, 4, 8, 8, 2)),
    (640, (128, 128, 128, 2, 8, 8, 2)),
    (1 << 30, (128, 128, 128, 1, 8, 8, 2)),
)


def _pick(M):
    for bound, cfg in _TABLE:
        if M <= bound:
            return cfg
    return _DEFAULT


@torch.no_grad()
def run(attn_output: torch.Tensor, o_proj_weight: torch.Tensor) -> torch.Tensor:
    bsz, num_heads, seq_len, v_head_dim = attn_output.shape
    hidden_size, intermediate_size = o_proj_weight.shape

    # Fallbacks: anything the fast path does not model exactly.
    if (
        attn_output.stride(3) != 1
        or o_proj_weight.stride(1) != 1
        or intermediate_size != num_heads * v_head_dim
        or v_head_dim & (v_head_dim - 1) != 0
        or intermediate_size % 128 != 0
        or hidden_size % 128 != 0
        or attn_output.dtype != o_proj_weight.dtype
        or not attn_output.is_cuda
    ):
        x = attn_output.transpose(1, 2).reshape(bsz, seq_len, intermediate_size)
        return torch.matmul(x, o_proj_weight.t())

    M = bsz * seq_len
    N = hidden_size
    K = intermediate_size

    BLOCK_M, BLOCK_N, BLOCK_K, SPLIT_K, GROUP_M, num_warps, num_stages = _pick(M)

    while SPLIT_K > 1 and (K // SPLIT_K) % BLOCK_K != 0:
        SPLIT_K //= 2

    if SPLIT_K == 1:
        out = torch.empty((bsz, seq_len, N), device=attn_output.device,
                          dtype=attn_output.dtype)
        c = out
    else:
        c = torch.zeros((bsz, seq_len, N), device=attn_output.device,
                        dtype=torch.float32)

    grid = (triton.cdiv(M, BLOCK_M) * triton.cdiv(N, BLOCK_N), SPLIT_K)

    _fused_proj_kernel[grid](
        attn_output, o_proj_weight, c,
        M, N, seq_len,
        attn_output.stride(0), attn_output.stride(1), attn_output.stride(2),
        o_proj_weight.stride(0),
        N,
        K=K,
        D=v_head_dim,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        BLOCK_K=BLOCK_K,
        SPLIT_K=SPLIT_K,
        GROUP_M=GROUP_M,
        ONE_BATCH=(bsz == 1),
        EVEN_M=(M % BLOCK_M == 0),
        num_warps=num_warps,
        num_stages=num_stages,
    )

    if SPLIT_K != 1:
        out = c.to(attn_output.dtype)
    return out

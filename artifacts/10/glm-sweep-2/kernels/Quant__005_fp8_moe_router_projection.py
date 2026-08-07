import torch
import triton
import triton.language as tl


@triton.jit
def _gemm_kernel(
    A, W, SA, SB, C,
    M, N, K, nb,
    sam, sak, swn, swk, ssa, ssb,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    NUM_STAGES: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    rk = tl.arange(0, BLOCK_K)
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for kb in tl.range(0, nb, num_stages=NUM_STAGES):
        koff = kb * BLOCK_K
        a = tl.load(A + rm[:, None] * sam + (koff + rk)[None, :])
        b = tl.load(W + rn[:, None] * swn + (koff + rk)[None, :])
        p = tl.dot(a, b.T, out_dtype=tl.float32)
        sa = tl.load(SA + rm * ssa + kb)
        sb = tl.load(SB + kb)
        acc += p * (sa[:, None] * sb)
    tl.store(C + rm[:, None] * N + rn[None, :], acc.to(tl.bfloat16))


_BLOCK_M = 64
_BLOCK_N = 64
_BLOCK_K = 128
_NUM_STAGES = 4
_NUM_WARPS = 8


@torch.no_grad()
def run(hidden_states, gate_weight, scale_hidden, scale_weight):
    num_experts = 64
    M, K = hidden_states.shape
    w = gate_weight[:num_experts]
    N = num_experts
    nb = K // 128
    c = torch.empty(M, N, device=hidden_states.device, dtype=torch.bfloat16)
    grid = (triton.cdiv(M, _BLOCK_M), triton.cdiv(N, _BLOCK_N))
    _gemm_kernel[grid](
        hidden_states, w, scale_hidden, scale_weight, c,
        M, N, K, nb,
        hidden_states.stride(0), hidden_states.stride(1),
        w.stride(0), w.stride(1),
        scale_hidden.stride(0), scale_hidden.stride(1),
        BLOCK_M=_BLOCK_M, BLOCK_N=_BLOCK_N, BLOCK_K=_BLOCK_K,
        NUM_STAGES=_NUM_STAGES,
        num_warps=_NUM_WARPS,
    )
    return c

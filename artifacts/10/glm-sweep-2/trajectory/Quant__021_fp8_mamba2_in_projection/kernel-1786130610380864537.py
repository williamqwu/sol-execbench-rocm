import torch
import triton
import triton.language as tl

E4M3_MAX = 448.0


@triton.jit
def _dequant_1x128_kernel(x_ptr, out_ptr, M, K, xsm, xsk, osm, osk, BLOCK_K: tl.constexpr):
    pid_m = tl.program_id(0)
    pid_k = tl.program_id(1)
    k_offs = pid_k * BLOCK_K + tl.arange(0, BLOCK_K)
    x = tl.load(x_ptr + pid_m * xsm + k_offs * xsk, mask=k_offs < K, other=0.0).to(tl.float32)
    amax = tl.max(tl.abs(x), axis=0)
    scale = tl.maximum(amax / 448.0, 1e-12)
    q = tl.clamp(x / scale, -448.0, 448.0).to(tl.float8e4nv)
    dq = (q.to(tl.float32) * scale).to(tl.bfloat16)
    tl.store(out_ptr + pid_m * osm + k_offs * osk, dq, mask=k_offs < K)


@triton.jit
def _dequant_128x128_kernel(x_ptr, out_ptr, N, K, BN: tl.constexpr, BK: tl.constexpr,
                            xsn, xsk, osn, osk):
    pid_n = tl.program_id(0)
    pid_k = tl.program_id(1)
    n_offs = pid_n * BN + tl.arange(0, BN)
    k_offs = pid_k * BK + tl.arange(0, BK)
    mask = (n_offs[:, None] < N) & (k_offs[None, :] < K)
    x = tl.load(x_ptr + n_offs[:, None] * xsn + k_offs[None, :] * xsk, mask=mask, other=0.0).to(tl.float32)
    amax = tl.max(tl.max(tl.abs(x), axis=1), axis=0)
    scale = tl.maximum(amax / 448.0, 1e-12)
    q = tl.clamp(x / scale, -448.0, 448.0).to(tl.float8e4nv)
    dq = (q.to(tl.float32) * scale).to(tl.bfloat16)
    tl.store(out_ptr + n_offs[:, None] * osn + k_offs[None, :] * osk, dq, mask=mask)


@torch.no_grad()
def run(hidden_states: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    M, K = hidden_states.shape
    N = weight.shape[0]
    a = torch.empty(M, K, dtype=torch.bfloat16, device=hidden_states.device)
    b = torch.empty(N, K, dtype=torch.bfloat16, device=hidden_states.device)
    _dequant_1x128_kernel[(M, K // 128)](
        hidden_states, a, M, K, hidden_states.stride(0), hidden_states.stride(1),
        a.stride(0), a.stride(1), BLOCK_K=128)
    _dequant_128x128_kernel[(N // 128, K // 128)](
        weight, b, N, K, 128, 128, weight.stride(0), weight.stride(1), b.stride(0), b.stride(1))
    return a @ b.T

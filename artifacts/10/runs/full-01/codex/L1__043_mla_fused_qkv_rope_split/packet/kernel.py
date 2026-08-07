import torch
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def _rmsnorm_inplace_kernel(x_ptr, weight_ptr, eps, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    offsets = tl.arange(0, BLOCK)
    mask = offsets < 1536
    x = tl.load(x_ptr + row * 1536 + offsets, mask=mask, other=0.0).to(tl.float32)
    variance = tl.sum(x * x, axis=0) * (1.0 / 1536.0)
    normed_bf16 = (x * tl.rsqrt(variance + eps)).to(tl.bfloat16)
    weight = tl.load(weight_ptr + offsets, mask=mask, other=0.0)
    tl.store(x_ptr + row * 1536 + offsets, normed_bf16 * weight, mask=mask)


# Kept separate from the entry point so candidate tiles can be benchmarked
# without changing the correctness-first hipBLASLt projection path.
@triton.jit
def _q_b_mm_kernel(
    a_ptr,
    b_ptr,
    c_ptr,
    M: tl.constexpr,
    BM: tl.constexpr,
    BN: tl.constexpr,
    BK: tl.constexpr,
    GROUP_M: tl.constexpr,
):
    N: tl.constexpr = 24576
    K: tl.constexpr = 1536
    pid = tl.program_id(0)
    num_pid_m = tl.cdiv(M, BM)
    num_pid_n = tl.cdiv(N, BN)
    num_pid_in_group = GROUP_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_M
    group_size_m = tl.minimum(num_pid_m - first_pid_m, GROUP_M)
    pid_m = first_pid_m + (pid % num_pid_in_group) % group_size_m
    pid_n = (pid % num_pid_in_group) // group_size_m

    offs_m = pid_m * BM + tl.arange(0, BM)
    offs_n = pid_n * BN + tl.arange(0, BN)
    offs_k = tl.arange(0, BK)
    a_ptrs = a_ptr + offs_m[:, None] * K + offs_k[None, :]
    b_ptrs = b_ptr + offs_k[:, None] + offs_n[None, :] * K
    acc = tl.zeros((BM, BN), dtype=tl.float32)
    for _ in range(0, K, BK):
        a = tl.load(a_ptrs, mask=offs_m[:, None] < M, other=0.0)
        b = tl.load(b_ptrs)
        acc += tl.dot(a, b)
        a_ptrs += BK
        b_ptrs += BK
    out_ptrs = c_ptr + offs_m[:, None] * N + offs_n[None, :]
    tl.store(out_ptrs, acc, mask=(offs_m[:, None] < M) & (offs_n[None, :] < N))


@torch.no_grad()
def run(
    hidden_states,
    q_a_proj_weight,
    q_a_layernorm_weight,
    q_b_proj_weight,
    kv_a_proj_weight,
    rms_norm_eps,
):
    batch_size, seq_len, _ = hidden_states.shape

    q_latent = F.linear(hidden_states, q_a_proj_weight)
    rows = batch_size * seq_len
    _rmsnorm_inplace_kernel[(rows,)](
        q_latent, q_a_layernorm_weight, rms_norm_eps, BLOCK=2048, num_warps=4
    )

    q = F.linear(q_latent, q_b_proj_weight).view(batch_size, seq_len, 128, 192)
    q_nope = q[..., :128]
    q_pe = q[..., 128:]

    kv = F.linear(hidden_states, kv_a_proj_weight)
    compressed_kv = kv[..., :512]
    k_pe = kv[..., 512:].view(batch_size, seq_len, 1, 64)
    return q_nope, q_pe, compressed_kv, k_pe

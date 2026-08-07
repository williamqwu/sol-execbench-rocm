import torch
import triton
import triton.language as tl


N = 256
K = 7168


@triton.jit
def _gemv_kernel(
    a_ptr, b_ptr, c_ptr,
    TOTAL_K: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    n = tl.program_id(0)
    k = tl.arange(0, BLOCK_K)
    acc = tl.zeros((BLOCK_K,), tl.float32)
    for k_base in range(0, TOTAL_K, BLOCK_K):
        mask = k_base + k < TOTAL_K
        a = tl.load(a_ptr + k_base + k, mask=mask, other=0.0)
        b = tl.load(b_ptr + n * TOTAL_K + k_base + k, mask=mask, other=0.0)
        acc += a.to(tl.float32) * b.to(tl.float32)
    tl.store(c_ptr + n, tl.sum(acc, axis=0))


@triton.jit
def _gemm_kernel(
    a_ptr,
    b_ptr,
    c_ptr,
    M: tl.constexpr,
    TOTAL_N: tl.constexpr,
    TOTAL_K: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid = tl.program_id(0)
    num_pid_n = tl.cdiv(TOTAL_N, BLOCK_N)
    pid_m = pid // num_pid_n
    pid_n = pid - pid_m * num_pid_n

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    a_ptrs = a_ptr + offs_m[:, None] * TOTAL_K + offs_k[None, :]
    b_ptrs = b_ptr + offs_n[None, :] * TOTAL_K + offs_k[:, None]

    acc = tl.zeros((BLOCK_M, BLOCK_N), tl.float32)
    for _ in range(0, TOTAL_K, BLOCK_K):
        a = tl.load(a_ptrs, mask=offs_m[:, None] < M, other=0.0)
        b = tl.load(b_ptrs)
        acc = tl.dot(a, b, acc)
        a_ptrs += BLOCK_K
        b_ptrs += BLOCK_K

    c_ptrs = c_ptr + offs_m[:, None] * TOTAL_N + offs_n[None, :]
    tl.store(c_ptrs, acc, mask=offs_m[:, None] < M)


@triton.jit
def _small_m_kernel(
    a_ptr,
    b_ptr,
    c_ptr,
    M: tl.constexpr,
    TOTAL_N: tl.constexpr,
    TOTAL_K: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_n = tl.program_id(0)
    m = tl.arange(0, BLOCK_M)
    n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    k = tl.arange(0, BLOCK_K)
    acc = tl.zeros((BLOCK_M, BLOCK_N), tl.float32)
    for k_base in range(0, TOTAL_K, BLOCK_K):
        a = tl.load(
            a_ptr + m[:, None, None] * TOTAL_K + k_base + k[None, None, :],
            mask=m[:, None, None] < M,
            other=0.0,
        )
        b = tl.load(
            b_ptr + n[None, :, None] * TOTAL_K + k_base + k[None, None, :],
        )
        products = a.to(tl.float32) * b.to(tl.float32)
        acc += tl.sum(products, axis=2)
    c_ptrs = c_ptr + m[:, None] * TOTAL_N + n[None, :]
    tl.store(c_ptrs, acc, mask=m[:, None] < M)


@triton.jit
def _splitk_gemm_kernel(
    a_ptr,
    b_ptr,
    partial_ptr,
    M: tl.constexpr,
    TOTAL_N: tl.constexpr,
    TOTAL_K: tl.constexpr,
    SPLIT_K: tl.constexpr,
    ATOMIC: tl.constexpr,
    N_MAJOR: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid = tl.program_id(0)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(TOTAL_N, BLOCK_N)
    tiles_per_split = num_pid_m * num_pid_n
    pid_k = pid // tiles_per_split
    tile = pid - pid_k * tiles_per_split
    if N_MAJOR:
        pid_n = tile // num_pid_m
        pid_m = tile - pid_n * num_pid_m
    else:
        pid_m = tile // num_pid_n
        pid_n = tile - pid_m * num_pid_n

    m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    k = tl.arange(0, BLOCK_K)
    k_per_split = TOTAL_K // SPLIT_K
    k_start = pid_k * k_per_split
    a_ptrs = a_ptr + m[:, None] * TOTAL_K + k_start + k[None, :]
    b_ptrs = b_ptr + n[None, :] * TOTAL_K + k_start + k[:, None]
    acc = tl.zeros((BLOCK_M, BLOCK_N), tl.float32)
    for k_offset in range(0, k_per_split, BLOCK_K):
        k_mask = k_offset + k < k_per_split
        a = tl.load(
            a_ptrs,
            mask=(m[:, None] < M) & k_mask[None, :],
            other=0.0,
        )
        b = tl.load(b_ptrs, mask=k_mask[:, None], other=0.0)
        acc = tl.dot(a, b, acc)
        a_ptrs += BLOCK_K
        b_ptrs += BLOCK_K

    if ATOMIC:
        out_ptrs = partial_ptr + m[:, None] * TOTAL_N + n[None, :]
        tl.atomic_add(out_ptrs, acc, mask=m[:, None] < M)
    else:
        out_ptrs = (
            partial_ptr + pid_k * M * TOTAL_N
            + m[:, None] * TOTAL_N + n[None, :]
        )
        tl.store(out_ptrs, acc, mask=m[:, None] < M)


@triton.jit
def _splitk_reduce_kernel(
    partial_ptr,
    c_ptr,
    NUMEL: tl.constexpr,
    SPLIT_K: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    offsets = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < NUMEL
    acc = tl.zeros((BLOCK_SIZE,), tl.float32)
    for split in range(SPLIT_K):
        acc += tl.load(
            partial_ptr + split * NUMEL + offsets,
            mask=mask,
            other=0.0,
        )
    tl.store(c_ptr + offsets, acc, mask=mask)


def _launch_gemm(A, B, C, block_m, block_n, block_k, num_warps,
                 waves_per_eu=1, matrix_instr_nonkdim=16, kpack=1,
                 num_stages=1):
    M = A.shape[0]
    grid = (triton.cdiv(M, block_m) * triton.cdiv(N, block_n),)
    _gemm_kernel[grid](
        A, B, C, M=M, TOTAL_N=N, TOTAL_K=K,
        BLOCK_M=block_m, BLOCK_N=block_n, BLOCK_K=block_k,
        num_warps=num_warps, num_stages=num_stages, waves_per_eu=waves_per_eu,
        matrix_instr_nonkdim=matrix_instr_nonkdim, kpack=kpack,
    )


def _launch_splitk(A, B, C, partial, split_k, block_m, block_n, block_k,
                   num_warps=4, waves_per_eu=1, num_stages=1,
                   n_major=False, reduce_block=256, reduce_warps=1):
    M = A.shape[0]
    grid = (
        split_k * triton.cdiv(M, block_m) * triton.cdiv(N, block_n),
    )
    _splitk_gemm_kernel[grid](
        A, B, partial,
        M=M, TOTAL_N=N, TOTAL_K=K, SPLIT_K=split_k,
        ATOMIC=False, N_MAJOR=n_major,
        BLOCK_M=block_m, BLOCK_N=block_n, BLOCK_K=block_k,
        num_warps=num_warps, num_stages=num_stages,
        waves_per_eu=waves_per_eu,
        matrix_instr_nonkdim=16,
    )
    numel = M * N
    _splitk_reduce_kernel[(triton.cdiv(numel, reduce_block),)](
        partial, C, NUMEL=numel, SPLIT_K=split_k,
        BLOCK_SIZE=reduce_block, num_warps=reduce_warps,
    )


def _launch_splitk_atomic(A, B, C, split_k, block_m, block_n, block_k,
                          num_warps=4, waves_per_eu=1):
    M = A.shape[0]
    grid = (
        split_k * triton.cdiv(M, block_m) * triton.cdiv(N, block_n),
    )
    _splitk_gemm_kernel[grid](
        A, B, C,
        M=M, TOTAL_N=N, TOTAL_K=K, SPLIT_K=split_k,
        ATOMIC=True, N_MAJOR=False,
        BLOCK_M=block_m, BLOCK_N=block_n, BLOCK_K=block_k,
        num_warps=num_warps, num_stages=1, waves_per_eu=waves_per_eu,
        matrix_instr_nonkdim=16,
    )


def run(A, B):
    M = A.shape[0]
    C = torch.empty((M, N), device=A.device, dtype=A.dtype)
    if M == 1:
        _gemv_kernel[(N,)](
            A, B, C, TOTAL_K=K, BLOCK_K=256, num_warps=4,
        )
    elif M <= 16:
        split_k = 8
        partial = torch.empty(
            (split_k, M, N), device=A.device, dtype=torch.float32,
        )
        _launch_splitk(
            A, B, C, partial, split_k, 16, 16, 256,
            num_warps=4, waves_per_eu=1, num_stages=3,
        )
    elif M <= 64:
        split_k = 7
        partial = torch.empty(
            (split_k, M, N), device=A.device, dtype=torch.float32,
        )
        _launch_splitk(
            A, B, C, partial, split_k, 32, 16, 256,
            num_warps=4, waves_per_eu=2, num_stages=3,
            n_major=True,
        )
    elif M <= 80:
        split_k = 7
        partial = torch.empty(
            (split_k, M, N), device=A.device, dtype=torch.float32,
        )
        _launch_splitk(
            A, B, C, partial, split_k, 32, 16, 256,
            num_warps=4, waves_per_eu=2, num_stages=2,
            n_major=True,
        )
    elif M <= 1024:
        split_k = 8
        partial = torch.empty(
            (split_k, M, N), device=A.device, dtype=torch.float32,
        )
        _launch_splitk(
            A, B, C, partial, split_k, 128, 64, 128,
            num_warps=8, waves_per_eu=4, num_stages=3,
            n_major=True, reduce_block=1024, reduce_warps=4,
        )
    else:
        C = torch.matmul(A, B.T)
    return C

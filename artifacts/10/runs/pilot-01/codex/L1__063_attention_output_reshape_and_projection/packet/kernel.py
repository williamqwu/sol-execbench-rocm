import torch
import triton
import triton.language as tl


@triton.jit
def _proj_kernel(
    attn,
    weight,
    out,
    M: tl.constexpr,
    SEQ: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    b = offs_m // SEQ
    s = offs_m - b * SEQ
    acc = tl.zeros((BLOCK_M, BLOCK_N), tl.float32)

    for k0 in range(0, K, BLOCK_K):
        k = k0 + offs_k
        head = k // 128
        dim = k - head * 128
        a_ptrs = attn + ((b[:, None] * 128 + head[None, :]) * SEQ + s[:, None]) * 128 + dim[None, :]
        w_ptrs = weight + offs_n[None, :] * K + k[:, None]

        a = tl.load(a_ptrs, mask=offs_m[:, None] < M, other=0.0)
        w = tl.load(w_ptrs)
        acc += tl.dot(a, w, out_dtype=tl.float32)

    out_ptrs = out + offs_m[:, None] * N + offs_n[None, :]
    tl.store(out_ptrs, acc, mask=offs_m[:, None] < M)


@triton.jit
def _reshape_kernel(
    attn,
    reshaped,
    M: tl.constexpr,
    SEQ: tl.constexpr,
    K: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_k = tl.program_id(1)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_k = pid_k * BLOCK_K + tl.arange(0, BLOCK_K)

    b = offs_m // SEQ
    s = offs_m - b * SEQ
    head = offs_k // 128
    dim = offs_k - head * 128

    vals = tl.load(
        attn + ((b[:, None] * 128 + head[None, :]) * SEQ + s[:, None]) * 128 + dim[None, :],
        mask=offs_m[:, None] < M,
        other=0.0,
    )
    tl.store(
        reshaped + offs_m[:, None] * K + offs_k[None, :],
        vals,
        mask=offs_m[:, None] < M,
    )


@torch.no_grad()
def run(attn_output: torch.Tensor, o_proj_weight: torch.Tensor) -> torch.Tensor:
    bsz = attn_output.shape[0]
    seq_len = attn_output.shape[2]
    hidden_size = o_proj_weight.shape[0]
    intermediate_size = o_proj_weight.shape[1]
    m = bsz * seq_len

    reshaped = torch.empty((m, intermediate_size), device=attn_output.device, dtype=torch.bfloat16)
    if m <= 160:
        block_m = 4
        block_k = 512
        copy_warps = 4
    elif m <= 320:
        block_m = 32
        block_k = 128
        copy_warps = 8
    elif bsz == 4 and seq_len == 211:
        block_m = 32
        block_k = 512
        copy_warps = 4
    elif bsz == 8 and seq_len == 131:
        block_m = 8
        block_k = 512
        copy_warps = 4
    elif bsz == 4 and seq_len == 512:
        block_m = 64
        block_k = 512
        copy_warps = 8
    elif bsz == 1 and seq_len == 2048:
        block_m = 4
        block_k = 512
        copy_warps = 4
    elif m <= 2048:
        block_m = 64
        block_k = 256
        copy_warps = 8
    elif bsz == 8 and seq_len == 512:
        block_m = 64
        block_k = 512
        copy_warps = 8
    else:
        block_m = 32
        block_k = 256
        copy_warps = 4

    _reshape_kernel[(triton.cdiv(m, block_m), triton.cdiv(intermediate_size, block_k))](
        attn_output,
        reshaped,
        m,
        seq_len,
        intermediate_size,
        BLOCK_M=block_m,
        BLOCK_K=block_k,
        num_warps=copy_warps,
        num_stages=4,
    )

    output = torch.empty((m, hidden_size), device=attn_output.device, dtype=torch.bfloat16)
    torch.mm(reshaped, o_proj_weight.t(), out=output)
    return output.reshape(bsz, seq_len, hidden_size)

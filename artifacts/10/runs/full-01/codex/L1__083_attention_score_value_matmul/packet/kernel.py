import torch
import triton
import triton.language as tl


@triton.jit
def _attention_value_kernel(
    attention_weights,
    value,
    output,
    Q: tl.constexpr,
    K: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_bh = tl.program_id(1)
    batch = pid_bh // 20
    head = pid_bh - batch * 20

    rows = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    cols = tl.arange(0, 64)
    kk = tl.arange(0, BLOCK_K)

    a_base = attention_weights + (batch * 20 + head) * Q * K
    v_base = value + (batch * 20 + head) * K * 64
    acc = tl.zeros((BLOCK_M, 64), dtype=tl.float32)

    for k0 in range(0, K, BLOCK_K):
        a = tl.load(
            a_base + rows[:, None] * K + (k0 + kk[None, :]),
            mask=(rows[:, None] < Q) & ((k0 + kk[None, :]) < K),
            other=0.0,
        )
        v = tl.load(
            v_base + (k0 + kk[:, None]) * 64 + cols[None, :],
            mask=(k0 + kk[:, None]) < K,
            other=0.0,
        )
        acc += tl.dot(a, v)

    out_ptrs = (
        output
        + batch * Q * 1280
        + rows[:, None] * 1280
        + head * 64
        + cols[None, :]
    )
    tl.store(out_ptrs, acc, mask=rows[:, None] < Q)


@triton.jit
def _transpose_heads_kernel(
    source,
    output,
    Q: tl.constexpr,
    BLOCK_Q: tl.constexpr,
):
    pid_q = tl.program_id(0)
    pid_bh = tl.program_id(1)
    batch = pid_bh // 20
    head = pid_bh - batch * 20
    rows = pid_q * BLOCK_Q + tl.arange(0, BLOCK_Q)
    dims = tl.arange(0, 64)
    x = tl.load(
        source + (batch * 20 + head) * Q * 64 + rows[:, None] * 64 + dims[None, :],
        mask=rows[:, None] < Q,
    )
    tl.store(
        output
        + batch * Q * 1280
        + rows[:, None] * 1280
        + head * 64
        + dims[None, :],
        x,
        mask=rows[:, None] < Q,
    )


def run(attention_weights: torch.Tensor, value: torch.Tensor) -> torch.Tensor:
    batch, _, q_len, k_len = attention_weights.shape
    output = torch.empty(
        (batch, q_len, 1280), device=attention_weights.device, dtype=torch.bfloat16
    )

    # For the very long reduction, rocBLAS' GEMM is faster than a small-N
    # Triton tile.  Its result is already BF16, so a tiny layout kernel can
    # perform the required transpose without changing any numerics.
    if k_len > 1024:
        temporary = torch.bmm(
            attention_weights.view(batch * 20, q_len, k_len),
            value.view(batch * 20, k_len, 64),
        )
        block_q = 32
        _transpose_heads_kernel[(triton.cdiv(q_len, block_q), batch * 20)](
            temporary,
            output,
            Q=q_len,
            BLOCK_Q=block_q,
            num_warps=4,
            num_stages=1,
        )
        return output

    # The output width is fixed at 64 per head.  Tune the row tile to balance
    # value-matrix reuse against the small ragged shapes in the workload.
    if k_len <= 77:
        num_stages = 1
        if q_len <= 512:
            if batch >= 8:
                block_m, block_k, num_warps = 128, 128, 4
            else:
                block_m, block_k, num_warps = 32, 32, 2
        elif q_len < 768:
            block_m, block_k, num_warps = 32, 32, 2
        elif q_len <= 1024:
            block_m, block_k, num_warps = 64, 128, 2
        elif q_len <= 2048:
            block_m, block_k, num_warps = 64, 128, 4
        elif batch <= 2:
            block_m, block_k, num_warps = 128, 32, 4
        else:
            block_m, block_k, num_warps = 256, 32, 8
    elif k_len <= 256:
        if batch <= 2:
            block_m, block_k = 64, 64
        elif batch <= 4:
            block_m, block_k = 128, 64
        else:
            block_m, block_k = 128, 32
        num_warps = 4
        num_stages = 3
    elif k_len <= 512:
        block_m = 16
        block_k = 32
        num_warps = 4
        num_stages = 2
    else:
        block_m = 64
        block_k = 64
        num_warps = 4
        num_stages = 4

    grid = (triton.cdiv(q_len, block_m), batch * 20)
    _attention_value_kernel[grid](
        attention_weights,
        value,
        output,
        Q=q_len,
        K=k_len,
        BLOCK_M=block_m,
        BLOCK_K=block_k,
        num_warps=num_warps,
        num_stages=num_stages,
    )
    return output

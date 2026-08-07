import torch
import triton
import triton.language as tl


@triton.jit
def _softmax_backward_scale(
    grad_ptr,
    weight_ptr,
    scaled_ptr,
    scaling,
    n_cols: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK_N)
    mask = cols < n_cols
    offset = row * n_cols + cols

    grad = tl.load(grad_ptr + offset, mask=mask, other=0.0).to(tl.float32)
    weight = tl.load(weight_ptr + offset, mask=mask, other=0.0).to(tl.float32)
    row_sum = tl.sum(grad * weight, axis=0)

    # The reference rounds the softmax derivative to bf16, then rounds once
    # more after multiplying by the (float32) scale.
    logits = (weight * (grad - row_sum)).to(tl.bfloat16)
    scaled = (logits.to(tl.float32) * scaling).to(tl.bfloat16)
    tl.store(scaled_ptr + offset, scaled, mask=mask)


@triton.jit
def _softmax_backward_scale_rows(
    grad_ptr,
    weight_ptr,
    scaled_ptr,
    scaling,
    n_rows,
    n_cols: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    row = tl.program_id(0) * BLOCK_M + tl.arange(0, BLOCK_M)
    col = tl.arange(0, BLOCK_N)
    mask = (row[:, None] < n_rows) & (col[None, :] < n_cols)
    offset = row[:, None] * n_cols + col[None, :]
    grad = tl.load(grad_ptr + offset, mask=mask, other=0.0).to(tl.float32)
    weight = tl.load(weight_ptr + offset, mask=mask, other=0.0).to(tl.float32)
    row_sum = tl.sum(grad * weight, axis=1)
    logits = (weight * (grad - row_sum[:, None])).to(tl.bfloat16)
    scaled = (logits.to(tl.float32) * scaling).to(tl.bfloat16)
    tl.store(scaled_ptr + offset, scaled, mask=mask)


@triton.jit
def _both_batched_matmuls(
    scaled_ptr,
    query_ptr,
    key_ptr,
    grad_query_ptr,
    grad_key_ptr,
    M: tl.constexpr,
    N: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    tile = tl.program_id(0)
    bh = tl.program_id(1)
    rows = tile * BLOCK_M + tl.arange(0, BLOCK_M)
    dims = tl.arange(0, 128)
    red = tl.arange(0, BLOCK_K)

    acc_q = tl.zeros((BLOCK_M, 128), tl.float32)
    for start in range(0, N, BLOCK_K):
        cols = start + red
        a = tl.load(
            scaled_ptr + bh * M * N + rows[:, None] * N + cols[None, :],
            mask=(rows[:, None] < M) & (cols[None, :] < N),
            other=0.0,
        )
        b = tl.load(
            key_ptr + bh * N * 128 + cols[:, None] * 128 + dims[None, :],
            mask=cols[:, None] < N,
            other=0.0,
        )
        acc_q += tl.dot(a, b)

    acc_k = tl.zeros((BLOCK_M, 128), tl.float32)
    for start in range(0, M, BLOCK_K):
        cols = start + red
        # Load [reduction, output rows], then transpose for the dot product.
        a = tl.load(
            scaled_ptr + bh * M * N + cols[:, None] * N + rows[None, :],
            mask=(cols[:, None] < M) & (rows[None, :] < N),
            other=0.0,
        )
        b = tl.load(
            query_ptr + bh * M * 128 + cols[:, None] * 128 + dims[None, :],
            mask=cols[:, None] < M,
            other=0.0,
        )
        acc_k += tl.dot(tl.trans(a), b)

    tl.store(
        grad_query_ptr + bh * M * 128 + rows[:, None] * 128 + dims[None, :],
        acc_q,
        mask=rows[:, None] < M,
    )
    tl.store(
        grad_key_ptr + bh * N * 128 + rows[:, None] * 128 + dims[None, :],
        acc_k,
        mask=rows[:, None] < N,
    )


@triton.jit
def _fused_softmax_dq(
    grad_ptr,
    weight_ptr,
    key_ptr,
    scaled_ptr,
    grad_query_ptr,
    scaling,
    M: tl.constexpr,
    N: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    tile = tl.program_id(0)
    bh = tl.program_id(1)
    rows = tile * BLOCK_M + tl.arange(0, BLOCK_M)
    red = tl.arange(0, BLOCK_K)
    dims = tl.arange(0, 128)

    row_sum = tl.zeros((BLOCK_M,), tl.float32)
    for start in range(0, N, BLOCK_K):
        cols = start + red
        offsets = bh * M * N + rows[:, None] * N + cols[None, :]
        mask = (rows[:, None] < M) & (cols[None, :] < N)
        grad = tl.load(grad_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        weight = tl.load(weight_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        row_sum += tl.sum(grad * weight, axis=1)

    acc = tl.zeros((BLOCK_M, 128), tl.float32)
    for start in range(0, N, BLOCK_K):
        cols = start + red
        offsets = bh * M * N + rows[:, None] * N + cols[None, :]
        mask = (rows[:, None] < M) & (cols[None, :] < N)
        grad = tl.load(grad_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        weight = tl.load(weight_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        logits = (weight * (grad - row_sum[:, None])).to(tl.bfloat16)
        scaled = (logits.to(tl.float32) * scaling).to(tl.bfloat16)
        tl.store(scaled_ptr + offsets, scaled, mask=mask)
        key = tl.load(
            key_ptr + bh * N * 128 + cols[:, None] * 128 + dims[None, :],
            mask=cols[:, None] < N,
            other=0.0,
        )
        acc += tl.dot(scaled, key)

    tl.store(
        grad_query_ptr + bh * M * 128 + rows[:, None] * 128 + dims[None, :],
        acc,
        mask=rows[:, None] < M,
    )


@triton.jit
def _fused_small_square(
    grad_ptr,
    weight_ptr,
    query_ptr,
    key_ptr,
    scaled_ptr,
    grad_query_ptr,
    grad_key_ptr,
    scaling,
    M: tl.constexpr,
    N: tl.constexpr,
    BLOCK: tl.constexpr,
):
    bh = tl.program_id(0)
    rows = tl.arange(0, BLOCK)
    cols = tl.arange(0, BLOCK)
    offsets = bh * M * N + rows[:, None] * N + cols[None, :]
    mask = (rows[:, None] < M) & (cols[None, :] < N)

    grad = tl.load(grad_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    weight = tl.load(weight_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    row_sum = tl.sum(grad * weight, axis=1)
    logits = (weight * (grad - row_sum[:, None])).to(tl.bfloat16)
    scaled = (logits.to(tl.float32) * scaling).to(tl.bfloat16)
    tl.store(scaled_ptr + offsets, scaled, mask=mask)

    dims = tl.arange(0, BLOCK)
    key = tl.load(
        key_ptr + bh * N * 128 + cols[:, None] * 128 + dims[None, :],
        mask=(cols[:, None] < N) & (dims[None, :] < 128),
        other=0.0,
    )
    acc_q = tl.dot(scaled, key)
    tl.store(
        grad_query_ptr + bh * M * 128 + rows[:, None] * 128 + dims[None, :],
        acc_q,
        mask=(rows[:, None] < M) & (dims[None, :] < 128),
    )

    query = tl.load(
        query_ptr + bh * M * 128 + rows[:, None] * 128 + dims[None, :],
        mask=(rows[:, None] < M) & (dims[None, :] < 128),
        other=0.0,
    )
    acc_k = tl.dot(tl.trans(scaled), query)
    tl.store(
        grad_key_ptr + bh * N * 128 + cols[:, None] * 128 + dims[None, :],
        acc_k,
        mask=(cols[:, None] < N) & (dims[None, :] < 128),
    )


@triton.jit
def _bmm_128(
    a_ptr,
    b_ptr,
    out_ptr,
    A_ROWS: tl.constexpr,
    A_COLS: tl.constexpr,
    OUT_ROWS: tl.constexpr,
    RED: tl.constexpr,
    TRANS_A: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid = tl.program_id(0)
    bh = tl.program_id(1)
    tiles_n: tl.constexpr = tl.cdiv(128, BLOCK_N)
    pid_m = pid // tiles_n
    pid_n = pid % tiles_n
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    acc = tl.zeros((BLOCK_M, BLOCK_N), tl.float32)

    for start in range(0, RED, BLOCK_K):
        red = start + offs_k
        if TRANS_A:
            a_raw = tl.load(
                a_ptr + bh * A_ROWS * A_COLS + red[:, None] * A_COLS + offs_m[None, :],
                mask=(red[:, None] < RED) & (offs_m[None, :] < OUT_ROWS),
                other=0.0,
            )
            a = tl.trans(a_raw)
        else:
            a = tl.load(
                a_ptr + bh * A_ROWS * A_COLS + offs_m[:, None] * A_COLS + red[None, :],
                mask=(offs_m[:, None] < OUT_ROWS) & (red[None, :] < RED),
                other=0.0,
            )
        b = tl.load(
            b_ptr + bh * RED * 128 + red[:, None] * 128 + offs_n[None, :],
            mask=(red[:, None] < RED) & (offs_n[None, :] < 128),
            other=0.0,
        )
        acc += tl.dot(a, b)

    tl.store(
        out_ptr + bh * OUT_ROWS * 128 + offs_m[:, None] * 128 + offs_n[None, :],
        acc,
        mask=(offs_m[:, None] < OUT_ROWS) & (offs_n[None, :] < 128),
    )


@triton.jit
def _fused_square_stream(
    grad_ptr,
    weight_ptr,
    query_ptr,
    key_ptr,
    scaled_ptr,
    grad_query_ptr,
    grad_key_ptr,
    scaling,
    M: tl.constexpr,
    N: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_M: tl.constexpr,
):
    bh = tl.program_id(0)
    cols = tl.arange(0, BLOCK_N)
    dims = tl.arange(0, 128)
    local_rows = tl.arange(0, BLOCK_M)
    acc_k = tl.zeros((BLOCK_N, 128), tl.float32)

    for start in range(0, M, BLOCK_M):
        rows = start + local_rows
        offsets = bh * M * N + rows[:, None] * N + cols[None, :]
        mask = (rows[:, None] < M) & (cols[None, :] < N)
        grad = tl.load(grad_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        weight = tl.load(weight_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        row_sum = tl.sum(grad * weight, axis=1)
        logits = (weight * (grad - row_sum[:, None])).to(tl.bfloat16)
        scaled = (logits.to(tl.float32) * scaling).to(tl.bfloat16)
        tl.store(scaled_ptr + offsets, scaled, mask=mask)

        key = tl.load(
            key_ptr + bh * N * 128 + cols[:, None] * 128 + dims[None, :],
            mask=cols[:, None] < N,
            other=0.0,
        )
        acc_q = tl.dot(scaled, key)
        tl.store(
            grad_query_ptr + bh * M * 128 + rows[:, None] * 128 + dims[None, :],
            acc_q,
            mask=rows[:, None] < M,
        )

        query = tl.load(
            query_ptr + bh * M * 128 + rows[:, None] * 128 + dims[None, :],
            mask=rows[:, None] < M,
            other=0.0,
        )
        acc_k += tl.dot(tl.trans(scaled), query)

    tl.store(
        grad_key_ptr + bh * N * 128 + cols[:, None] * 128 + dims[None, :],
        acc_k,
        mask=cols[:, None] < N,
    )


@torch.no_grad()
def run(grad_output, query, key, attn_weights, scaling):
    batch, heads, m, n = grad_output.shape
    head_dim = query.shape[-1]
    bh = batch * heads

    grad_scaled = torch.empty_like(grad_output)
    if m == 128 and n == 128:
        grad_query = torch.empty_like(query)
        grad_key = torch.empty_like(key)
        _fused_small_square[(bh,)](
            grad_output,
            attn_weights,
            query,
            key,
            grad_scaled,
            grad_query,
            grad_key,
            scaling,
            M=m,
            N=n,
            BLOCK=128,
            num_warps=2,
        )
        return grad_query, grad_key
    elif n <= 224:
        grad_query = torch.empty_like(query)
        _fused_softmax_dq[(triton.cdiv(m, 16), bh)](
            grad_output,
            attn_weights,
            key,
            grad_scaled,
            grad_query,
            scaling,
            M=m,
            N=n,
            BLOCK_M=16,
            BLOCK_K=64,
            num_warps=8,
        )
    else:
        block_n = triton.next_power_of_2(n)
        total_rows = bh * m
        if n < 512 and total_rows >= 100000:
            _softmax_backward_scale_rows[(triton.cdiv(total_rows, 4),)](
                grad_output,
                attn_weights,
                grad_scaled,
                scaling,
                total_rows,
                n_cols=n,
                BLOCK_M=4,
                BLOCK_N=block_n,
                num_warps=4,
            )
        elif n < 512:
            num_warps = 1
        elif n <= 1024:
            num_warps = 2
        elif n <= 2048:
            num_warps = 4
        else:
            num_warps = 8
        if not (n < 512 and total_rows >= 100000):
            _softmax_backward_scale[(total_rows,)](
                grad_output,
                attn_weights,
                grad_scaled,
                scaling,
                n_cols=n,
                BLOCK_N=block_n,
                num_warps=num_warps,
            )
        grad_query = torch.bmm(
            grad_scaled.view(bh, m, n), key.view(bh, n, head_dim)
        ).view(batch, heads, m, head_dim)

    scaled_3d = grad_scaled.view(bh, m, n)
    grad_key = torch.bmm(scaled_3d.transpose(1, 2), query.view(bh, m, head_dim))
    return (
        grad_query,
        grad_key.view(batch, heads, n, head_dim),
    )

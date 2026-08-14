import torch
import triton
import triton.language as tl


@triton.jit
def _grad_weights_kernel(
    grad_out_ptr,
    value_ptr,
    mask_ptr,
    grad_weights_ptr,
    Q: tl.constexpr,
    K: tl.constexpr,
    KEEP_PROB,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    tile = tl.program_id(0)
    bh = tl.program_id(1)
    num_n = tl.cdiv(K, BLOCK_N)
    tile_m = tile // num_n
    tile_n = tile - tile_m * num_n

    b = bh // 80
    h = bh - b * 80
    kv_h = h // 10

    offs_m = tile_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tile_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, BLOCK_D)

    # grad_out is [B, Q, 80, 128], value is [B, 8, K, 128].
    g_ptrs = (
        grad_out_ptr
        + ((b * Q + offs_m[:, None]) * 80 + h) * 128
        + offs_d[None, :]
    )
    v_ptrs = (
        value_ptr
        + ((b * 8 + kv_h) * K + offs_n[:, None]) * 128
        + offs_d[None, :]
    )
    g = tl.load(g_ptrs, mask=offs_m[:, None] < Q, other=0.0)
    v = tl.load(v_ptrs, mask=offs_n[:, None] < K, other=0.0)
    acc = tl.dot(g, tl.trans(v))

    out_offsets = (bh * Q + offs_m[:, None]) * K + offs_n[None, :]
    valid = (offs_m[:, None] < Q) & (offs_n[None, :] < K)
    keep = tl.load(mask_ptr + out_offsets, mask=valid, other=0)
    # This is the reference's multiply-by-mask followed by division.
    dropped = tl.where(keep, acc / KEEP_PROB, 0.0)
    tl.store(grad_weights_ptr + out_offsets, dropped, mask=valid)


@triton.jit
def _grad_scores_fused_kernel(
    grad_out_ptr,
    value_ptr,
    mask_ptr,
    weights_ptr,
    grad_scores_ptr,
    Q: tl.constexpr,
    K: tl.constexpr,
    KEEP_PROB,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    tile_m = tl.program_id(0)
    bh = tl.program_id(1)
    b = bh // 80
    h = bh - b * 80
    kv_h = h // 10

    offs_m = tile_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, BLOCK_D)
    acc = tl.zeros((BLOCK_M, BLOCK_N), tl.float32)
    for d_start in range(0, 128, BLOCK_D):
        ds = d_start + offs_d
        g_ptrs = (
            grad_out_ptr
            + ((b * Q + offs_m[:, None]) * 80 + h) * 128
            + ds[None, :]
        )
        v_ptrs = (
            value_ptr
            + ((b * 8 + kv_h) * K + offs_n[:, None]) * 128
            + ds[None, :]
        )
        g = tl.load(g_ptrs, mask=offs_m[:, None] < Q, other=0.0)
        v = tl.load(v_ptrs, mask=offs_n[:, None] < K, other=0.0)
        acc += tl.dot(g, tl.trans(v))

    out_offsets = (bh * Q + offs_m[:, None]) * K + offs_n[None, :]
    valid = (offs_m[:, None] < Q) & (offs_n[None, :] < K)
    keep = tl.load(mask_ptr + out_offsets, mask=valid, other=0)
    x = tl.where(keep, acc / KEEP_PROB, 0.0)
    w = tl.load(weights_ptr + out_offsets, mask=valid, other=0.0).to(tl.float32)
    sum_term = tl.sum(x * w, axis=1)
    result = w * (x - sum_term[:, None])
    tl.store(grad_scores_ptr + out_offsets, result, mask=valid)


@triton.jit
def _softmax_backward_kernel(
    grad_weights_ptr,
    weights_ptr,
    grad_scores_ptr,
    K: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK_K)
    valid = offs < K
    x = tl.load(grad_weights_ptr + row * K + offs, mask=valid, other=0.0)
    w = tl.load(weights_ptr + row * K + offs, mask=valid, other=0.0).to(tl.float32)
    sum_term = tl.sum(x * w, axis=0)
    result = w * (x - sum_term)
    tl.store(grad_scores_ptr + row * K + offs, result, mask=valid)


@triton.jit
def _grad_value_kernel(
    dropped_weights_ptr,
    grad_out_ptr,
    grad_value_ptr,
    Q: tl.constexpr,
    K: tl.constexpr,
    BLOCK_K: tl.constexpr,
    BLOCK_D: tl.constexpr,
    BLOCK_Q: tl.constexpr,
):
    tile = tl.program_id(0)
    bkv = tl.program_id(1)
    num_d = tl.cdiv(128, BLOCK_D)
    tile_k = tile // num_d
    tile_d = tile - tile_k * num_d

    b = bkv // 8
    kv_h = bkv - b * 8
    offs_k = tile_k * BLOCK_K + tl.arange(0, BLOCK_K)
    offs_d = tile_d * BLOCK_D + tl.arange(0, BLOCK_D)

    total = tl.zeros((BLOCK_K, BLOCK_D), tl.float32)
    # Keeping one accumulator per attention head mirrors the two-stage
    # reference: Q reduction in each GEMM, followed by the 10-head GQA sum.
    for group in range(10):
        h = kv_h * 10 + group
        part = tl.zeros((BLOCK_K, BLOCK_D), tl.float32)
        for q_start in range(0, Q, BLOCK_Q):
            offs_q = q_start + tl.arange(0, BLOCK_Q)
            w_ptrs = (
                dropped_weights_ptr
                + ((b * 80 + h) * Q + offs_q[:, None]) * K
                + offs_k[None, :]
            )
            g_ptrs = (
                grad_out_ptr
                + ((b * Q + offs_q[:, None]) * 80 + h) * 128
                + offs_d[None, :]
            )
            w = tl.load(
                w_ptrs,
                mask=(offs_q[:, None] < Q) & (offs_k[None, :] < K),
                other=0.0,
            )
            g = tl.load(
                g_ptrs,
                mask=(offs_q[:, None] < Q) & (offs_d[None, :] < 128),
                other=0.0,
            )
            part += tl.dot(tl.trans(w), g)
        total += part

    out_ptrs = (
        grad_value_ptr
        + ((b * 8 + kv_h) * K + offs_k[:, None]) * 128
        + offs_d[None, :]
    )
    tl.store(
        out_ptrs,
        total,
        mask=(offs_k[:, None] < K) & (offs_d[None, :] < 128),
    )


@torch.no_grad()
def run(
    grad_attn_output,
    attn_weights,
    attn_weights_dropped,
    value_states,
    dropout_mask,
    attention_dropout,
):
    batch_size, q_len, _, _ = grad_attn_output.shape
    k_len = value_states.shape[2]

    grad_scores = torch.empty_like(attn_weights)
    grad_value = torch.empty_like(value_states)

    if k_len <= 1024:
        block_n = triton.next_power_of_2(k_len)
        # Keep the accumulator at 16K elements while maximizing reuse of V.
        block_m = 16384 // block_n
        _grad_scores_fused_kernel[(triton.cdiv(q_len, block_m), batch_size * 80)](
            grad_attn_output,
            value_states,
            dropout_mask,
            attn_weights,
            grad_scores,
            Q=q_len,
            K=k_len,
            KEEP_PROB=1.0 - attention_dropout,
            BLOCK_M=block_m,
            BLOCK_N=block_n,
            BLOCK_D=32 if block_n >= 1024 else 64,
            num_warps=8,
            num_stages=2,
        )
    else:
        grad_weights = torch.empty(
            attn_weights.shape, device=attn_weights.device, dtype=torch.float32
        )
        block_m = 64
        block_n = 128
        weight_grid = (
            triton.cdiv(q_len, block_m) * triton.cdiv(k_len, block_n),
            batch_size * 80,
        )
        _grad_weights_kernel[weight_grid](
            grad_attn_output,
            value_states,
            dropout_mask,
            grad_weights,
            Q=q_len,
            K=k_len,
            KEEP_PROB=1.0 - attention_dropout,
            BLOCK_M=block_m,
            BLOCK_N=block_n,
            BLOCK_D=128,
            num_warps=8,
            num_stages=2,
        )

        block_softmax = triton.next_power_of_2(k_len)
        _softmax_backward_kernel[(batch_size * 80 * q_len,)](
            grad_weights,
            attn_weights,
            grad_scores,
            K=k_len,
            BLOCK_K=block_softmax,
            num_warps=8 if block_softmax >= 2048 else 4,
            num_stages=1,
        )

    large_value_tile = batch_size * 8 * triton.cdiv(k_len, 128) >= 256
    block_k = 128 if large_value_tile else 64
    block_d = 128 if large_value_tile else 64
    value_grid = (
        triton.cdiv(k_len, block_k) * triton.cdiv(128, block_d),
        batch_size * 8,
    )
    _grad_value_kernel[value_grid](
        attn_weights_dropped,
        grad_attn_output,
        grad_value,
        Q=q_len,
        K=k_len,
        BLOCK_K=block_k,
        BLOCK_D=block_d,
        BLOCK_Q=32,
        num_warps=8,
        num_stages=2,
    )
    return grad_scores, grad_value

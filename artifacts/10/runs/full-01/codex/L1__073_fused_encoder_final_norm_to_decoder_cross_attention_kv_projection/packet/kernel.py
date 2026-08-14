import torch
import triton
import triton.language as tl


@triton.jit
def _fused_rms_kv_kernel(
    x_ptr,
    norm_ptr,
    k_weight_ptr,
    v_weight_ptr,
    keys_ptr,
    values_ptr,
    n_rows: tl.constexpr,
    seq_len: tl.constexpr,
    eps: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    SINGLE_PASS: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    rows = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    k_lane = tl.arange(0, BLOCK_K)
    row_mask = rows < n_rows

    # Columns 0:128 are K and columns 128:256 are V. Keeping both in one
    # logical GEMM lets a program reuse each normalized input tile.
    logical_cols = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    proj_cols = logical_cols & 127
    is_key = logical_cols < 128
    col_mask = logical_cols < 256
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    sum_sq = tl.zeros((BLOCK_M,), dtype=tl.float32)

    for k0 in range(0, 1024, BLOCK_K):
        ks = k0 + k_lane
        x = tl.load(
            x_ptr + rows[:, None] * 1024 + ks[None, :],
            mask=row_mask[:, None],
            other=0.0,
        ).to(tl.float32)
        sum_sq += tl.sum(x * x, axis=1)
        if SINGLE_PASS:
            nw = tl.load(norm_ptr + ks).to(tl.float32)
            normalized = (x * nw[None, :]).to(tl.float16)
            w_offsets = proj_cols[None, :] * 1024 + ks[:, None]
            wk = tl.load(
                k_weight_ptr + w_offsets,
                mask=is_key[None, :] & col_mask[None, :],
                other=0.0,
            )
            wv = tl.load(
                v_weight_ptr + w_offsets,
                mask=(~is_key[None, :]) & col_mask[None, :],
                other=0.0,
            )
            weights = tl.where(is_key[None, :], wk, wv)
            acc += tl.dot(normalized, weights)

    inv_rms = tl.rsqrt(sum_sq * (1.0 / 1024.0) + eps)
    if SINGLE_PASS:
        acc *= inv_rms[:, None]
    else:
        for k0 in range(0, 1024, BLOCK_K):
            ks = k0 + k_lane
            x = tl.load(
                x_ptr + rows[:, None] * 1024 + ks[None, :],
                mask=row_mask[:, None],
                other=0.0,
            ).to(tl.float32)
            nw = tl.load(norm_ptr + ks).to(tl.float32)

            # Match the reference's FP16 materialization before projection.
            normalized = (x * inv_rms[:, None] * nw[None, :]).to(tl.float16)
            w_offsets = proj_cols[None, :] * 1024 + ks[:, None]
            wk = tl.load(
                k_weight_ptr + w_offsets,
                mask=is_key[None, :] & col_mask[None, :],
                other=0.0,
            )
            wv = tl.load(
                v_weight_ptr + w_offsets,
                mask=(~is_key[None, :]) & col_mask[None, :],
                other=0.0,
            )
            weights = tl.where(is_key[None, :], wk, wv)
            acc += tl.dot(normalized, weights)

    # (batch, sequence, head, dim) -> contiguous (batch, head, sequence, dim)
    batch = rows // seq_len
    seq = rows - batch * seq_len
    head = proj_cols // 64
    dim = proj_cols & 63
    out_offsets = (
        batch[:, None] * (2 * seq_len * 64)
        + head[None, :] * (seq_len * 64)
        + seq[:, None] * 64
        + dim[None, :]
    )
    store_mask = row_mask[:, None] & col_mask[None, :]
    tl.store(keys_ptr + out_offsets, acc, mask=store_mask & is_key[None, :])
    tl.store(values_ptr + out_offsets, acc, mask=store_mask & (~is_key[None, :]))


@torch.no_grad()
def run(
    encoder_hidden_states: torch.Tensor,
    norm_weight: torch.Tensor,
    k_proj_weight: torch.Tensor,
    v_proj_weight: torch.Tensor,
    eps: float,
):
    batch_size, seq_len, _ = encoder_hidden_states.shape
    n_rows = batch_size * seq_len
    keys = torch.empty(
        (batch_size, 2, seq_len, 64),
        device=encoder_hidden_states.device,
        dtype=encoder_hidden_states.dtype,
    )
    values = torch.empty_like(keys)

    if n_rows < 512:
        block_m, block_n, num_warps = 16, 32, 4
    elif n_rows < 1536:
        block_m, block_n, num_warps = 16, 64, 4
    elif n_rows < 2500:
        block_m, block_n, num_warps = 32, 64, 8
    elif n_rows < 6144:
        block_m, block_n, num_warps = 64, 64, 8
    else:
        block_m, block_n, num_warps = 64, 128, 8

    grid = (triton.cdiv(n_rows, block_m), triton.cdiv(256, block_n))
    _fused_rms_kv_kernel[grid](
        encoder_hidden_states,
        norm_weight,
        k_proj_weight,
        v_proj_weight,
        keys,
        values,
        n_rows=n_rows,
        seq_len=seq_len,
        eps=eps,
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        BLOCK_K=32,
        SINGLE_PASS=n_rows < 1536,
        num_warps=num_warps,
        num_stages=2,
    )
    return keys, values

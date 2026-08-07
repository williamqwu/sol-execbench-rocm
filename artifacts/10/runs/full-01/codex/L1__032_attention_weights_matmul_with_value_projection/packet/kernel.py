import torch
import triton
import triton.language as tl


@triton.jit
def _attn_value_kernel(
    attn_ptr,
    value_ptr,
    out_ptr,
    SEQ_LEN: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    EVEN_M: tl.constexpr,
    EVEN_K: tl.constexpr,
):
    tile = tl.program_id(0)
    head = tl.program_id(1)
    batch = tl.program_id(2)
    bh = batch * 40 + head

    tiles_n = tl.cdiv(128, BLOCK_N)
    tile_m = tile // tiles_n
    tile_n = tile - tile_m * tiles_n

    offs_m = tile_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tile_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    attn = attn_ptr + bh * SEQ_LEN * SEQ_LEN
    value = value_ptr + bh * SEQ_LEN * 128

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k in range(0, SEQ_LEN, BLOCK_K):
        if EVEN_M and EVEN_K:
            a = tl.load(attn + offs_m[:, None] * SEQ_LEN + k + offs_k[None, :])
        else:
            a = tl.load(
                attn + offs_m[:, None] * SEQ_LEN + k + offs_k[None, :],
                mask=(offs_m[:, None] < SEQ_LEN)
                & (k + offs_k[None, :] < SEQ_LEN),
                other=0.0,
            )
        if EVEN_K:
            v = tl.load(value + (k + offs_k[:, None]) * 128 + offs_n[None, :])
        else:
            v = tl.load(
                value + (k + offs_k[:, None]) * 128 + offs_n[None, :],
                mask=k + offs_k[:, None] < SEQ_LEN,
                other=0.0,
            )
        acc += tl.dot(a, v)

    out = out_ptr + batch * SEQ_LEN * 5120
    out_offsets = offs_m[:, None] * 5120 + head * 128 + offs_n[None, :]
    if EVEN_M:
        tl.store(out + out_offsets, acc)
    else:
        tl.store(out + out_offsets, acc, mask=offs_m[:, None] < SEQ_LEN)


@torch.no_grad()
def run(attn_weights: torch.Tensor, value_states: torch.Tensor) -> torch.Tensor:
    batch, _, seq_len, _ = attn_weights.shape
    output = torch.empty(
        (batch, seq_len, 5120), dtype=torch.bfloat16, device=attn_weights.device
    )

    # The small matrices need more independent tiles, while the longer matrices
    # benefit from loading each attention tile only once with a 128-wide tile.
    # Ragged K dimensions use a shallower pipeline to avoid masked-tail pressure.
    block_m, block_n, block_k, num_warps, num_stages = 64, 128, 32, 4, 1
    if seq_len == 128:
        if batch >= 32:
            block_k, num_stages = 32, 4
        else:
            block_k = 64
    elif seq_len == 131:
        block_m, block_k, num_stages = 32, 64, 2
    elif seq_len == 256:
        block_m, block_n, block_k, num_warps, num_stages = 128, 64, 64, 4, 3
        if batch == 8:
            block_n, block_k, num_warps, num_stages = 128, 32, 8, 2
    elif seq_len == 293:
        block_k, num_warps, num_stages = 64, 8, 3
    elif seq_len == 512:
        block_k, num_stages = 64, 3
        if batch == 8:
            block_m, num_warps = 128, 8
    elif seq_len in (691, 853, 997):
        block_k = 64
    elif seq_len in (1024, 2048):
        block_k, num_warps, num_stages = 64, 8, 3

    grid = (
        triton.cdiv(seq_len, block_m) * triton.cdiv(128, block_n),
        40,
        batch,
    )
    _attn_value_kernel[grid](
        attn_weights,
        value_states,
        output,
        SEQ_LEN=seq_len,
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        BLOCK_K=block_k,
        EVEN_M=(seq_len % block_m == 0),
        EVEN_K=(seq_len % block_k == 0),
        num_warps=num_warps,
        num_stages=num_stages,
    )
    return output

import torch
import triton
import triton.language as tl


@triton.jit
def _fused_mask_kernel(
    full_ptr,
    swa_ptr,
    T,
    S,
    past,
    stride_bh,
    stride_t,
    BLOCK_T: tl.constexpr,
    BLOCK_S: tl.constexpr,
):
    pid_bh = tl.program_id(0).to(tl.int64)
    pid_t = tl.program_id(1)

    n_heads = 64
    pid_b = pid_bh // n_heads
    pid_h = pid_bh % n_heads

    base = pid_b * stride_bh + pid_h * stride_t
    t_offs = pid_t * BLOCK_T + tl.arange(0, BLOCK_T)
    t_mask = t_offs < T

    for s_start in range(0, S, BLOCK_S):
        s_offs = s_start + tl.arange(0, BLOCK_S)
        s_mask = s_offs < S

        split = t_offs[:, None] + past + 1
        split = tl.minimum(split, S)
        full_vals = s_offs[None, :] >= split

        m = t_mask[:, None] & s_mask[None, :]
        ptrs = base + t_offs[:, None] * S + s_offs[None, :]
        tl.store(full_ptr + ptrs, full_vals, mask=m)
        tl.store(swa_ptr + ptrs, tl.zeros([BLOCK_T, BLOCK_S], dtype=tl.int1), mask=m)


def _block_config(source_length):
    # Narrow rows: tile several rows per program to cut launch overhead.
    # Wide rows: one row per program, large S-tile for coalesced writes.
    if source_length <= 512:
        return 16, 128
    if source_length <= 1024:
        return 1, 1024
    return 1, 2048


@torch.no_grad()
def run(
    batch_size_scalar: int,
    seq_length_scalar: int,
    past_key_values_length_scalar: int,
):
    num_attention_heads = 64
    swa_num_attention_heads = 64
    device = torch.device('cuda')

    seq_length = int(seq_length_scalar)
    past_key_values_length = int(past_key_values_length_scalar)
    batch_size = int(batch_size_scalar)

    target_length = seq_length
    source_length = seq_length + past_key_values_length

    full_attention_mask = torch.empty(
        (batch_size, num_attention_heads, target_length, source_length),
        dtype=torch.bool, device=device,
    )
    sliding_window_attention_mask = torch.empty(
        (batch_size, swa_num_attention_heads, target_length, source_length),
        dtype=torch.bool, device=device,
    )

    BLOCK_T, BLOCK_S = _block_config(source_length)
    stride_bh = num_attention_heads * target_length * source_length
    stride_t = target_length * source_length

    _fused_mask_kernel[(batch_size * num_attention_heads,
                        triton.cdiv(target_length, BLOCK_T))](
        full_attention_mask,
        sliding_window_attention_mask,
        target_length, source_length,
        past_key_values_length,
        stride_bh, stride_t,
        BLOCK_T, BLOCK_S,
    )

    return full_attention_mask, sliding_window_attention_mask

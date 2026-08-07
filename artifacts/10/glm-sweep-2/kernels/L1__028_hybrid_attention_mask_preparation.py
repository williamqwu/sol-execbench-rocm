import torch
import triton
import triton.language as tl


@triton.jit
def _full_mask_kernel(out_ptr, T: tl.constexpr, S: tl.constexpr, pkv,
                      BLOCK_T: tl.constexpr, BLOCK_S: tl.constexpr):
    pid_t = tl.program_id(0)
    pid_s = tl.program_id(1)
    rows = pid_t * BLOCK_T + tl.arange(0, BLOCK_T)  # [BLOCK_T]
    cols = pid_s * BLOCK_S + tl.arange(0, BLOCK_S)  # [BLOCK_S]
    row_mask = rows < T
    col_mask = cols < S
    # value = row < col - pkv
    vals = (rows[:, None] < (cols[None, :] - pkv)) & row_mask[:, None] & col_mask[None, :]
    tl.store(out_ptr + rows[:, None] * S + cols[None, :], vals, mask=row_mask[:, None] & col_mask[None, :])


@torch.no_grad()
def run(
    batch_size_scalar: int,
    seq_length_scalar: int,
    past_key_values_length_scalar: int,
):
    num_attention_heads = 64
    swa_num_attention_heads = 64
    device = torch.device('cuda')

    batch_size = int(batch_size_scalar)
    seq_length = int(seq_length_scalar)
    past_key_values_length = int(past_key_values_length_scalar)

    target_length = seq_length
    source_length = seq_length + past_key_values_length

    # Full 4D mask: value depends only on (row, col). Build the [T,S] tile once
    # then the kernel fills the full [B,H,T,S] output by tiling over the 2D plane.
    full_attention_mask = torch.empty(
        (batch_size, num_attention_heads, target_length, source_length),
        dtype=torch.bool, device=device,
    )

    BLOCK_T = 128
    BLOCK_S = 128
    # Reshape so the last two dims are contiguous [T, S] tiled across B*H.
    total_panels = batch_size * num_attention_heads
    out_2d = full_attention_mask.view(total_panels, target_length, source_length)
    grid = (triton.cdiv(target_length, BLOCK_T), triton.cdiv(source_length, BLOCK_S), total_panels)
    _full_mask_kernel[grid](
        out_2d, target_length, source_length, past_key_values_length,
        BLOCK_T=BLOCK_T, BLOCK_S=BLOCK_S, num_warps=8,
    )

    # Reference SWA mask is always all-False.
    sliding_window_attention_mask = torch.zeros(
        (batch_size, swa_num_attention_heads, target_length, source_length),
        dtype=torch.bool, device=device,
    )

    return full_attention_mask, sliding_window_attention_mask

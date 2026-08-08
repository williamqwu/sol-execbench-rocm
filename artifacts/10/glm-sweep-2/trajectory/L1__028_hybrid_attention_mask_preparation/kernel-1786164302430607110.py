import torch
import triton
import triton.language as tl


@triton.jit
def _full_mask_kernel(
    out_ptr,
    T,
    S,
    past,
    stride_bh,
    stride_t,
    BLOCK_S: tl.constexpr,
):
    pid_bh = tl.program_id(0).to(tl.int64)
    pid_t = tl.program_id(1)

    n_heads = 64
    pid_b = pid_bh // n_heads
    pid_h = pid_bh % n_heads

    # s >= (t + past + 1)  -> True (masked-out upper region)
    split = pid_t + past + 1
    split = tl.minimum(split, S)
    split = tl.maximum(split, 0)

    base = pid_b * stride_bh + pid_h * stride_t + pid_t * S

    for s_start in range(0, S, BLOCK_S):
        s_offs = s_start + tl.arange(0, BLOCK_S)
        mask = s_offs < S
        vals = s_offs >= split
        tl.store(out_ptr + base + s_offs, vals, mask=mask)


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

    BLOCK_S = 2048
    stride_bh = num_attention_heads * target_length * source_length
    stride_t = target_length * source_length

    _full_mask_kernel[(batch_size * num_attention_heads, target_length)](
        full_attention_mask,
        target_length, source_length,
        past_key_values_length,
        stride_bh, stride_t,
        BLOCK_S,
    )

    sliding_window_attention_mask = torch.zeros(
        (batch_size, swa_num_attention_heads, target_length, source_length),
        dtype=torch.bool, device=device,
    )

    return full_attention_mask, sliding_window_attention_mask

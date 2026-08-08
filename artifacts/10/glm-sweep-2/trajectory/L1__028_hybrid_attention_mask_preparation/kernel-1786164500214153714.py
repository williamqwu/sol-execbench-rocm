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

    split = pid_t + past + 1
    split = tl.minimum(split, S)
    split = tl.maximum(split, 0)

    base = pid_b * stride_bh + pid_h * stride_t + pid_t * S

    for s_start in range(0, S, BLOCK_S):
        s_offs = s_start + tl.arange(0, BLOCK_S)
        mask = s_offs < S
        vals = s_offs >= split
        tl.store(out_ptr + base + s_offs, vals, mask=mask)


def _full_mask_torch(batch_size, target_length, source_length, past_key_values_length,
                     num_attention_heads, device):
    rows = torch.arange(target_length, device=device).unsqueeze(1)
    cols = torch.arange(source_length, device=device).unsqueeze(0)
    full_mask = cols > (rows + past_key_values_length)
    return full_mask[None, None, :, :].expand(
        batch_size, num_attention_heads, target_length, source_length
    ).contiguous()


def _full_mask_triton(batch_size, target_length, source_length, past_key_values_length,
                      num_attention_heads, device):
    out = torch.empty(
        (batch_size, num_attention_heads, target_length, source_length),
        dtype=torch.bool, device=device,
    )
    BLOCK_S = 2048
    stride_bh = num_attention_heads * target_length * source_length
    stride_t = target_length * source_length
    _full_mask_kernel[(batch_size * num_attention_heads, target_length)](
        out,
        target_length, source_length,
        past_key_values_length,
        stride_bh, stride_t,
        BLOCK_S,
    )
    return out


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

    # Wide rows amortize the Triton launch; many narrow-row programs favor the
    # fused PyTorch broadcast+contiguous path.
    n_programs = batch_size * num_attention_heads * target_length
    if source_length >= 512 or n_programs < 50000:
        full_attention_mask = _full_mask_triton(
            batch_size, target_length, source_length, past_key_values_length,
            num_attention_heads, device,
        )
    else:
        full_attention_mask = _full_mask_torch(
            batch_size, target_length, source_length, past_key_values_length,
            num_attention_heads, device,
        )

    sliding_window_attention_mask = torch.zeros(
        (batch_size, swa_num_attention_heads, target_length, source_length),
        dtype=torch.bool, device=device,
    )

    return full_attention_mask, sliding_window_attention_mask

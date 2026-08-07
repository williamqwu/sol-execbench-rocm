import torch

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

    # Full causal mask: True where row < col - past_key_values_length.
    # Reference builds ones then masked_fill(causal_cond, False); causal_cond
    # is (row >= col - pkv), so the surviving True region is the complement.
    target_indices = torch.arange(target_length, device=device)[:, None]
    source_indices = torch.arange(source_length, device=device)[None, :]
    full_mask = target_indices < (source_indices - past_key_values_length)

    full_attention_mask = full_mask[None, None, :, :].expand(
        batch_size, num_attention_heads, target_length, source_length
    ).contiguous()

    # The reference SWA mask is always all-False: it starts from zeros and
    # masked_fill(valid_positions, False) only ever writes False into False.
    sliding_window_attention_mask = torch.zeros(
        (batch_size, swa_num_attention_heads, target_length, source_length),
        dtype=torch.bool,
        device=device,
    )

    return full_attention_mask, sliding_window_attention_mask

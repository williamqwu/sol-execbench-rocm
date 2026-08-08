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

    seq_length = int(seq_length_scalar)
    past_key_values_length = int(past_key_values_length_scalar)
    batch_size = int(batch_size_scalar)

    target_length = seq_length
    source_length = seq_length + past_key_values_length

    # full_mask[t, s] = (s > t + past)  -> True where masked-out (upper triangle shifted)
    rows = torch.arange(target_length, device=device).unsqueeze(1)
    cols = torch.arange(source_length, device=device).unsqueeze(0)
    full_mask = cols > (rows + past_key_values_length)

    # Expand to 4D and materialize.
    full_attention_mask = full_mask[None, None, :, :].expand(
        batch_size, num_attention_heads, target_length, source_length
    ).contiguous()

    # swa mask is identically False under the reference semantics.
    sliding_window_attention_mask = torch.zeros(
        (batch_size, swa_num_attention_heads, target_length, source_length),
        dtype=torch.bool,
        device=device,
    )

    return full_attention_mask, sliding_window_attention_mask

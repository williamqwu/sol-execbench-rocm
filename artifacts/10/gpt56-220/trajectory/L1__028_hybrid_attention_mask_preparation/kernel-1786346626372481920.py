import torch


@torch.no_grad()
def run(batch_size_scalar: int, seq_length_scalar: int,
        past_key_values_length_scalar: int):
    batch_size = int(batch_size_scalar)
    seq_length = int(seq_length_scalar)
    past = int(past_key_values_length_scalar)
    source_length = seq_length + past

    base = torch.ones((seq_length, source_length), dtype=torch.bool,
                      device="cuda").triu_(past + 1)
    shape = (batch_size, 64, seq_length, source_length)
    full = base[None, None].expand(shape).contiguous()
    swa = torch.zeros(shape, dtype=torch.bool, device="cuda")
    return full, swa

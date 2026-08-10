import torch
import triton
import triton.language as tl


@triton.jit
def _make_masks(full, swa, source_len: tl.constexpr, seq_len: tl.constexpr,
                past: tl.constexpr, BLOCK: tl.constexpr):
    col = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    query = tl.program_id(1)
    flat_row = tl.program_id(2) * seq_len + query
    offset = flat_row * source_len + col
    active = col < source_len
    tl.store(full + offset, col > query + past, mask=active)
    tl.store(swa + offset, 0, mask=active)


@torch.no_grad()
def run(batch_size_scalar: int, seq_length_scalar: int,
        past_key_values_length_scalar: int):
    batch = int(batch_size_scalar)
    seq = int(seq_length_scalar)
    past = int(past_key_values_length_scalar)
    source = seq + past
    shape = (batch, 64, seq, source)
    full = torch.empty(shape, dtype=torch.bool, device="cuda")
    swa = torch.empty_like(full)
    _make_masks[(triton.cdiv(source, 256), seq, batch * 64)](
        full, swa, source, seq, past, BLOCK=256)
    return full, swa

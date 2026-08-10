import torch
import triton
import triton.language as tl


@triton.jit
def _causal_template(out, source: tl.constexpr, past: tl.constexpr,
                     BLOCK: tl.constexpr):
    col = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    row = tl.program_id(1)
    tl.store(out + row * source + col, col > row + past, mask=col < source)


@triton.jit
def _direct_masks(full, swa, source: tl.constexpr, seq: tl.constexpr,
                  past: tl.constexpr, BLOCK: tl.constexpr):
    col = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    row = tl.program_id(1)
    flat_row = tl.program_id(2) * seq + row
    offset = flat_row * source + col
    active = col < source
    tl.store(full + offset, col > row + past, mask=active)
    tl.store(swa + offset, 0, mask=active)


@torch.no_grad()
def run(batch_size_scalar: int, seq_length_scalar: int,
        past_key_values_length_scalar: int):
    batch_size = int(batch_size_scalar)
    seq_length = int(seq_length_scalar)
    past = int(past_key_values_length_scalar)
    source_length = seq_length + past

    shape = (batch_size, 64, seq_length, source_length)
    storage = torch.empty((2, *shape), dtype=torch.bool, device="cuda")
    full, swa = storage.unbind(0)
    if full.numel() <= 8 * 1024 * 1024:
        _direct_masks[(triton.cdiv(source_length, 512), seq_length,
                       batch_size * 64)](
            full, swa, source_length, seq_length, past, BLOCK=512,
            num_warps=8)
        return full, swa

    base = torch.empty((seq_length, source_length), dtype=torch.bool,
                       device="cuda")
    _causal_template[(triton.cdiv(source_length, 1024), seq_length)](
        base, source_length, past, BLOCK=1024, num_warps=8)
    full.copy_(base)
    swa.zero_()
    return full, swa

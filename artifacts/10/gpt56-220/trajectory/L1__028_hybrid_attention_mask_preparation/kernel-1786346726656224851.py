import torch
import triton
import triton.language as tl


@triton.jit
def _causal_template(out, source: tl.constexpr, past: tl.constexpr,
                     BLOCK: tl.constexpr):
    col = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    row = tl.program_id(1)
    tl.store(out + row * source + col, col > row + past, mask=col < source)


@torch.no_grad()
def run(batch_size_scalar: int, seq_length_scalar: int,
        past_key_values_length_scalar: int):
    batch_size = int(batch_size_scalar)
    seq_length = int(seq_length_scalar)
    past = int(past_key_values_length_scalar)
    source_length = seq_length + past

    base = torch.empty((seq_length, source_length), dtype=torch.bool,
                       device="cuda")
    _causal_template[(triton.cdiv(source_length, 256), seq_length)](
        base, source_length, past, BLOCK=256)
    shape = (batch_size, 64, seq_length, source_length)
    storage = torch.empty((2, *shape), dtype=torch.bool, device="cuda")
    full, swa = storage.unbind(0)
    full.copy_(base)
    swa.zero_()
    return full, swa

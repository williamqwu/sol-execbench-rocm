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


@triton.jit
def _flat_masks(full, swa, n_elements, source: tl.constexpr,
                seq: tl.constexpr, past: tl.constexpr, BLOCK: tl.constexpr):
    offset = (tl.program_id(0).to(tl.int64) * BLOCK
              + tl.arange(0, BLOCK).to(tl.int64))
    active = offset < n_elements
    col = offset % source
    row = (offset // source) % seq
    tl.store(full + offset, col > row + past, mask=active)
    tl.store(swa + offset, 0, mask=active)


@triton.jit
def _row_masks(full, swa, source: tl.constexpr, seq: tl.constexpr,
               past: tl.constexpr, BLOCK: tl.constexpr):
    col = tl.arange(0, BLOCK)
    row = tl.program_id(0)
    batch_head = tl.program_id(1)
    flat_row = batch_head.to(tl.int64) * seq + row
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
        _direct_masks[(triton.cdiv(source_length, 256), seq_length,
                       batch_size * 64)](
            full, swa, source_length, seq_length, past, BLOCK=256,
            num_warps=2)
        return full, swa

    if (source_length & (source_length - 1) == 0
            or full.numel() <= 256 * 1024 * 1024):
        flat_warps = 1 if full.numel() <= 256 * 1024 * 1024 else 2
        _flat_masks[(triton.cdiv(full.numel(), 1024),)](
            full, swa, full.numel(), source_length, seq_length, past,
            BLOCK=1024, num_warps=flat_warps)
        return full, swa

    row_block = triton.next_power_of_2(source_length)
    row_warps = 4 if row_block <= 1024 else 8
    _row_masks[(seq_length, batch_size * 64)](
        full, swa, source_length, seq_length, past, BLOCK=row_block,
        num_warps=row_warps)
    return full, swa

    base = torch.empty((seq_length, source_length), dtype=torch.bool,
                       device="cuda")
    _causal_template[(triton.cdiv(source_length, 1024), seq_length)](
        base, source_length, past, BLOCK=1024, num_warps=2)
    full.copy_(base)
    swa.zero_()
    return full, swa

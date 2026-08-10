import torch
import triton
import triton.language as tl


@triton.jit
def _hist_kernel(x, counts, n: tl.constexpr, BLOCK: tl.constexpr):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    vals = tl.load(x + offs, mask=mask, other=0)
    hist = tl.histogram(vals, 256)
    bins = tl.arange(0, 256)
    tl.atomic_add(counts + bins, hist)


@triton.jit
def _prefix_kernel(counts, offsets, low_offsets, high_offsets):
    bins = tl.arange(0, 256)
    c = tl.load(counts + bins)
    inclusive = tl.cumsum(c)
    tl.store(offsets, 0)
    tl.store(offsets + bins + 1, inclusive)

    digits = tl.arange(0, 16)
    low_count = tl.zeros((16,), tl.int32)
    high_count = tl.zeros((16,), tl.int32)
    for i in range(0, 16):
        low_count += tl.load(counts + digits + 16 * i)
        high_count += tl.load(counts + digits * 16 + i)
    tl.store(low_offsets, 0)
    tl.store(low_offsets + digits + 1, tl.cumsum(low_count))
    tl.store(high_offsets, 0)
    tl.store(high_offsets + digits + 1, tl.cumsum(high_count))


@triton.jit
def _radix_kernel(x, digit_offsets, out, n: tl.constexpr, SHIFT: tl.constexpr,
                  BLOCK: tl.constexpr):
    digit = tl.program_id(0)
    base = tl.load(digit_offsets + digit)
    running = 0
    for start in range(0, n, BLOCK):
        offs = start + tl.arange(0, BLOCK)
        valid = offs < n
        vals = tl.load(x + offs, mask=valid, other=-1)
        selected = valid & (((vals >> SHIFT) & 15) == digit)
        ranks = tl.cumsum(selected.to(tl.int32))
        tl.store(out + base + running + ranks - 1, offs, mask=selected)
        running += tl.sum(selected.to(tl.int32))


@torch.no_grad()
def run(topk_idx: torch.Tensor):
    flat = topk_idx.reshape(-1)
    n = flat.numel()
    counts = torch.zeros(256, dtype=torch.int32, device=flat.device)
    offsets = torch.empty(257, dtype=torch.int32, device=flat.device)
    low_offsets = torch.empty(17, dtype=torch.int32, device=flat.device)
    high_offsets = torch.empty(17, dtype=torch.int32, device=flat.device)
    temp = torch.empty(n, dtype=torch.int32, device=flat.device)
    out = torch.empty(n, dtype=torch.int32, device=flat.device)
    block = 1024
    _hist_kernel[(triton.cdiv(n, block),)](flat, counts, n=n, BLOCK=block)
    _prefix_kernel[(1,)](counts, offsets, low_offsets, high_offsets, num_warps=4)
    _radix_kernel[(16,)](flat, low_offsets, temp, n=n, SHIFT=0, BLOCK=block,
                         num_warps=4)
    _radix_kernel[(16,)](temp, high_offsets, out, n=n, SHIFT=4, BLOCK=block,
                         num_warps=4)
    return out, offsets

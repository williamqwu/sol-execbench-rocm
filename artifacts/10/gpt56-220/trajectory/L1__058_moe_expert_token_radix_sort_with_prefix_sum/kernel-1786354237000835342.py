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
    hist -= tl.where(bins == 0, BLOCK - tl.sum(mask.to(tl.int32)), 0)
    tl.atomic_add(counts + bins, hist)


@triton.jit
def _prefix_kernel(counts, offsets):
    bins = tl.arange(0, 256)
    c = tl.load(counts + bins)
    tl.store(offsets, 0)
    tl.store(offsets + bins + 1, tl.cumsum(c))


@triton.jit
def _scatter_kernel(x, offsets, out, n: tl.constexpr, BLOCK: tl.constexpr):
    expert = tl.program_id(0)
    base = tl.load(offsets + expert)
    running = 0
    for start in range(0, n, BLOCK):
        offs = start + tl.arange(0, BLOCK)
        valid = offs < n
        vals = tl.load(x + offs, mask=valid, other=-1)
        selected = valid & (vals == expert)
        ranks = tl.cumsum(selected.to(tl.int32))
        tl.store(out + base + running + ranks - 1, offs, mask=selected)
        running += tl.sum(selected.to(tl.int32))


@torch.no_grad()
def run(topk_idx: torch.Tensor):
    flat = topk_idx.reshape(-1)
    n = flat.numel()
    block = 2048
    hist_block = 2048
    use_compaction = n <= 65536
    if not use_compaction:
        out = torch.argsort(flat, stable=True).to(torch.int32)
    else:
        out = torch.empty(n, dtype=torch.int32, device=flat.device)
    offsets = torch.zeros(257, dtype=torch.int32, device=flat.device)
    counts = offsets[1:]
    _hist_kernel[(triton.cdiv(n, hist_block),)](
        flat, counts, n=n, BLOCK=hist_block, num_warps=4
    )
    _prefix_kernel[(1,)](counts, offsets, num_warps=8)
    if use_compaction:
        _scatter_kernel[(256,)](flat, offsets, out, n=n, BLOCK=block, num_warps=8)
    return out, offsets

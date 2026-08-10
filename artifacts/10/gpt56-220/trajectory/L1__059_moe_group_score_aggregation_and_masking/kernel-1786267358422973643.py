import torch
import triton
import triton.language as tl


@triton.jit
def _moe_mask_kernel(scores, masked, group_mask):
    row = tl.program_id(0)
    lane = tl.arange(0, 32)
    base = row * 256

    x0 = tl.load(scores + base + lane)
    x1 = tl.load(scores + base + 32 + lane)
    x2 = tl.load(scores + base + 64 + lane)
    x3 = tl.load(scores + base + 96 + lane)
    x4 = tl.load(scores + base + 128 + lane)
    x5 = tl.load(scores + base + 160 + lane)
    x6 = tl.load(scores + base + 192 + lane)
    x7 = tl.load(scores + base + 224 + lane)

    s0 = tl.sum(tl.topk(x0, 2), axis=0)
    s1 = tl.sum(tl.topk(x1, 2), axis=0)
    s2 = tl.sum(tl.topk(x2, 2), axis=0)
    s3 = tl.sum(tl.topk(x3, 2), axis=0)
    s4 = tl.sum(tl.topk(x4, 2), axis=0)
    s5 = tl.sum(tl.topk(x5, 2), axis=0)
    s6 = tl.sum(tl.topk(x6, 2), axis=0)
    s7 = tl.sum(tl.topk(x7, 2), axis=0)

    g = tl.arange(0, 8)
    gs = tl.where(g == 0, s0, tl.where(g == 1, s1, tl.where(g == 2, s2, tl.where(g == 3, s3, tl.where(g == 4, s4, tl.where(g == 5, s5, tl.where(g == 6, s6, s7)))))))
    cutoff = tl.min(tl.topk(gs, 4), axis=0)
    n_greater = tl.sum((gs > cutoff).to(tl.int32), axis=0)
    # Rank cutoff ties by group index, the same preference used by argmax.
    eq = gs == cutoff
    eq_row = tl.expand_dims(eq, 0)
    group_row = tl.expand_dims(g, 0)
    group_col = tl.expand_dims(g, 1)
    tie_rank = tl.sum(eq_row & (group_row < group_col), axis=1)
    gm = (gs > cutoff) | (eq & (tie_rank < 4 - n_greater))
    tl.store(group_mask + row * 8 + g, gm.to(tl.float32))

    off = tl.arange(0, 256)
    vals = tl.load(scores + base + off)
    eg = off // 32
    es = tl.where(eg == 0, s0, tl.where(eg == 1, s1,
         tl.where(eg == 2, s2, tl.where(eg == 3, s3,
         tl.where(eg == 4, s4, tl.where(eg == 5, s5,
         tl.where(eg == 6, s6, s7)))))))
    erank = ((eg > 0) & (s0 == cutoff)).to(tl.int32)
    erank += ((eg > 1) & (s1 == cutoff)).to(tl.int32)
    erank += ((eg > 2) & (s2 == cutoff)).to(tl.int32)
    erank += ((eg > 3) & (s3 == cutoff)).to(tl.int32)
    erank += ((eg > 4) & (s4 == cutoff)).to(tl.int32)
    erank += ((eg > 5) & (s5 == cutoff)).to(tl.int32)
    erank += ((eg > 6) & (s6 == cutoff)).to(tl.int32)
    selected = (es > cutoff) | ((es == cutoff) & (erank < 4 - n_greater))
    tl.store(masked + base + off, tl.where(selected, vals, -float("inf")))


@torch.no_grad()
def run(scores: torch.Tensor):
    rows = scores.shape[0]
    masked = torch.empty_like(scores)
    group_mask = torch.empty((rows, 8), device=scores.device, dtype=torch.float32)
    _moe_mask_kernel[(rows,)](scores, masked, group_mask, num_warps=1)
    return masked, group_mask

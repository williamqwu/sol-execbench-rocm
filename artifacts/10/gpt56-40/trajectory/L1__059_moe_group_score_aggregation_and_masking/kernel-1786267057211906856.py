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

    a0 = tl.argmax(x0, axis=0); m0 = tl.max(x0, axis=0); s0 = m0 + tl.max(tl.where(lane == a0, -float("inf"), x0), axis=0)
    a1 = tl.argmax(x1, axis=0); m1 = tl.max(x1, axis=0); s1 = m1 + tl.max(tl.where(lane == a1, -float("inf"), x1), axis=0)
    a2 = tl.argmax(x2, axis=0); m2 = tl.max(x2, axis=0); s2 = m2 + tl.max(tl.where(lane == a2, -float("inf"), x2), axis=0)
    a3 = tl.argmax(x3, axis=0); m3 = tl.max(x3, axis=0); s3 = m3 + tl.max(tl.where(lane == a3, -float("inf"), x3), axis=0)
    a4 = tl.argmax(x4, axis=0); m4 = tl.max(x4, axis=0); s4 = m4 + tl.max(tl.where(lane == a4, -float("inf"), x4), axis=0)
    a5 = tl.argmax(x5, axis=0); m5 = tl.max(x5, axis=0); s5 = m5 + tl.max(tl.where(lane == a5, -float("inf"), x5), axis=0)
    a6 = tl.argmax(x6, axis=0); m6 = tl.max(x6, axis=0); s6 = m6 + tl.max(tl.where(lane == a6, -float("inf"), x6), axis=0)
    a7 = tl.argmax(x7, axis=0); m7 = tl.max(x7, axis=0); s7 = m7 + tl.max(tl.where(lane == a7, -float("inf"), x7), axis=0)

    g = tl.arange(0, 8)
    gs = tl.where(g == 0, s0, tl.where(g == 1, s1, tl.where(g == 2, s2, tl.where(g == 3, s3, tl.where(g == 4, s4, tl.where(g == 5, s5, tl.where(g == 6, s6, s7)))))))
    i0 = tl.argmax(gs, axis=0); gs = tl.where(g == i0, -float("inf"), gs)
    i1 = tl.argmax(gs, axis=0); gs = tl.where(g == i1, -float("inf"), gs)
    i2 = tl.argmax(gs, axis=0); gs = tl.where(g == i2, -float("inf"), gs)
    i3 = tl.argmax(gs, axis=0)
    gm = (g == i0) | (g == i1) | (g == i2) | (g == i3)
    tl.store(group_mask + row * 8 + g, gm.to(tl.float32))

    off = tl.arange(0, 256)
    vals = tl.load(scores + base + off)
    selected = (off // 32 == i0) | (off // 32 == i1) | (off // 32 == i2) | (off // 32 == i3)
    tl.store(masked + base + off, tl.where(selected, vals, -float("inf")))


@torch.no_grad()
def run(scores: torch.Tensor):
    rows = scores.shape[0]
    masked = torch.empty_like(scores)
    group_mask = torch.empty((rows, 8), device=scores.device, dtype=torch.float32)
    _moe_mask_kernel[(rows,)](scores, masked, group_mask, num_warps=8)
    return masked, group_mask

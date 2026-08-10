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

    q0 = (i0 == 0) | (i1 == 0) | (i2 == 0) | (i3 == 0)
    q1 = (i0 == 1) | (i1 == 1) | (i2 == 1) | (i3 == 1)
    q2 = (i0 == 2) | (i1 == 2) | (i2 == 2) | (i3 == 2)
    q3 = (i0 == 3) | (i1 == 3) | (i2 == 3) | (i3 == 3)
    q4 = (i0 == 4) | (i1 == 4) | (i2 == 4) | (i3 == 4)
    q5 = (i0 == 5) | (i1 == 5) | (i2 == 5) | (i3 == 5)
    q6 = (i0 == 6) | (i1 == 6) | (i2 == 6) | (i3 == 6)
    q7 = (i0 == 7) | (i1 == 7) | (i2 == 7) | (i3 == 7)
    tl.store(masked + base + lane, tl.where(q0, x0, -float("inf")))
    tl.store(masked + base + 32 + lane, tl.where(q1, x1, -float("inf")))
    tl.store(masked + base + 64 + lane, tl.where(q2, x2, -float("inf")))
    tl.store(masked + base + 96 + lane, tl.where(q3, x3, -float("inf")))
    tl.store(masked + base + 128 + lane, tl.where(q4, x4, -float("inf")))
    tl.store(masked + base + 160 + lane, tl.where(q5, x5, -float("inf")))
    tl.store(masked + base + 192 + lane, tl.where(q6, x6, -float("inf")))
    tl.store(masked + base + 224 + lane, tl.where(q7, x7, -float("inf")))


@torch.no_grad()
def run(scores: torch.Tensor):
    rows = scores.shape[0]
    masked = torch.empty_like(scores)
    group_mask = torch.empty((rows, 8), device=scores.device, dtype=torch.float32)
    _moe_mask_kernel[(rows,)](scores, masked, group_mask, num_warps=4)
    return masked, group_mask

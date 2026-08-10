import torch
import triton
import triton.language as tl


@triton.jit
def _patch_kernel(grid, weight, out, nimg: tl.constexpr, hidden: tl.constexpr,
                  BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    tok = offs // hidden
    d = offs - tok * hidden

    # Locate the image containing this flattened output token.  The number of
    # images is tiny (<=16), so scalar predication is cheaper than constructing
    # several temporary indexing tensors and synchronizing them through Python.
    start = tl.zeros((BLOCK,), tl.int64)
    chosen_start = tl.zeros((BLOCK,), tl.int64)
    tt = tl.full((BLOCK,), 1, tl.int64)
    hh = tl.full((BLOCK,), 2, tl.int64)
    ww = tl.full((BLOCK,), 2, tl.int64)
    found = tl.zeros((BLOCK,), tl.int1)
    for i in tl.static_range(nimg):
        t = tl.load(grid + i * 3)
        h = tl.load(grid + i * 3 + 1)
        w = tl.load(grid + i * 3 + 2)
        count = t * h * w
        take = (~found) & (tok < start + count)
        chosen_start = tl.where(take, start, chosen_start)
        tt = tl.where(take, t, tt)
        hh = tl.where(take, h, hh)
        ww = tl.where(take, w, ww)
        found = found | take
        start += count

    local = tok - chosen_start
    spatial = hh * ww
    p = local % spatial
    mw = ww // 2
    block = p // 4
    intra = p % 4
    y = (block // mw) * 2 + intra // 2
    x = (block % mw) * 2 + intra % 2

    yf = y.to(tl.float32) * 34.0 / (hh - 1).to(tl.float32)
    xf = x.to(tl.float32) * 34.0 / (ww - 1).to(tl.float32)
    y0 = yf.to(tl.int32)
    x0 = xf.to(tl.int32)
    y1 = tl.minimum(y0 + 1, 34)
    x1 = tl.minimum(x0 + 1, 34)
    dy = yf - y0.to(tl.float32)
    dx = xf - x0.to(tl.float32)
    i00 = (y0 * 35 + x0) * hidden + d
    i01 = (y0 * 35 + x1) * hidden + d
    i10 = (y1 * 35 + x0) * hidden + d
    i11 = (y1 * 35 + x1) * hidden + d
    v = tl.load(weight + i00) * ((1.0-dy)*(1.0-dx))
    v += tl.load(weight + i01) * ((1.0-dy)*dx)
    v += tl.load(weight + i10) * (dy*(1.0-dx))
    v += tl.load(weight + i11) * (dy*dx)
    tl.store(out + offs, v)


@triton.jit
def _rope_kernel(grid, inv, cosout, sinout, nimg: tl.constexpr,
                 head: tl.constexpr, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    tok = offs // head
    d = offs % head
    start = tl.zeros((BLOCK,), tl.int64)
    chosen_start = tl.zeros((BLOCK,), tl.int64)
    hh = tl.full((BLOCK,), 2, tl.int64)
    ww = tl.full((BLOCK,), 2, tl.int64)
    found = tl.zeros((BLOCK,), tl.int1)
    for i in tl.static_range(nimg):
        t = tl.load(grid + i * 3)
        h = tl.load(grid + i * 3 + 1)
        w = tl.load(grid + i * 3 + 2)
        count = t * h * w
        take = (~found) & (tok < start + count)
        chosen_start = tl.where(take, start, chosen_start)
        hh = tl.where(take, h, hh)
        ww = tl.where(take, w, ww)
        found = found | take
        start += count
    p = (tok - chosen_start) % (hh * ww)
    mw = ww // 2
    block = p // 4
    intra = p % 4
    row = (block // mw) * 2 + intra // 2
    col = (block % mw) * 2 + intra % 2
    q = d % 64
    coord = tl.where(q < 32, row, col).to(tl.float32)
    f = tl.load(inv + (q % 32))
    a = coord * f
    tl.store(cosout + offs, tl.cos(a))
    tl.store(sinout + offs, tl.sin(a))


@torch.no_grad()
def run(grid_thw: torch.Tensor, pos_embed_weight: torch.Tensor, inv_freq: torch.Tensor):
    # Obtaining the output size is the sole host synchronization.  The reference
    # performs many such synchronizations; all coordinate work stays on device.
    total = int(torch.prod(grid_thw, dim=1).sum().item())
    hidden = pos_embed_weight.shape[1]
    head = inv_freq.shape[0] * 4
    patch = torch.empty((total, hidden), device=grid_thw.device, dtype=pos_embed_weight.dtype)
    cos = torch.empty((total, head), device=grid_thw.device, dtype=torch.float32)
    sin = torch.empty_like(cos)
    nimg = grid_thw.shape[0]
    n_patch = total * hidden
    n_rope = total * head
    _patch_kernel[(triton.cdiv(n_patch, 256),)](grid_thw, pos_embed_weight, patch,
        nimg=nimg, hidden=hidden, BLOCK=256)
    _rope_kernel[(triton.cdiv(n_rope, 256),)](grid_thw, inv_freq, cos, sin,
        nimg=nimg, head=head, BLOCK=256)
    return patch, cos, sin

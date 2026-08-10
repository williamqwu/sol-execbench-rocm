import torch
import triton
import triton.language as tl


@triton.jit
def _rmsnorm(x, w, out, H: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < H
    xv = tl.load(x + row * H + cols, mask=mask, other=0.0).to(tl.float32)
    variance = tl.sum(xv * xv, axis=0) / H
    scale = tl.rsqrt(variance + 1.0e-6)
    weight = tl.load(w + cols, mask=mask, other=0.0).to(tl.float32)
    tl.store(out + row * H + cols, xv * scale * weight, mask=mask)


@triton.jit
def _partial_sums(x, partial, H: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    part = tl.program_id(1)
    cols = part * BLOCK + tl.arange(0, BLOCK)
    xv = tl.load(x + row * H + cols, mask=cols < H, other=0.0).to(tl.float32)
    tl.store(partial + row * 2 + part, tl.sum(xv * xv, axis=0))


@triton.jit
def _rmsnorm_from_sums(x, w, partial, out, H: tl.constexpr,
                       BLOCK: tl.constexpr):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < H
    total = tl.load(partial + row * 2) + tl.load(partial + row * 2 + 1)
    scale = tl.rsqrt(total / H + 1.0e-6)
    xv = tl.load(x + row * H + cols, mask=mask, other=0.0).to(tl.float32)
    weight = tl.load(w + cols, mask=mask, other=0.0).to(tl.float32)
    tl.store(out + row * H + cols, xv * scale * weight, mask=mask)


@torch.no_grad()
def run(hidden_states, weight):
    rows, hidden = hidden_states.shape
    assert hidden == 7168
    out = torch.empty_like(hidden_states)
    if rows >= 1000:
        partial = torch.empty((rows, 2), device=hidden_states.device,
                              dtype=torch.float32)
        _partial_sums[(rows, 2)](hidden_states, partial, hidden,
                                 BLOCK=4096, num_warps=8)
        _rmsnorm_from_sums[(rows,)](hidden_states, weight, partial, out, hidden,
                                    BLOCK=8192, num_warps=8)
    else:
        _rmsnorm[(rows,)](hidden_states, weight, out, hidden,
                          BLOCK=8192, num_warps=8)
    return out

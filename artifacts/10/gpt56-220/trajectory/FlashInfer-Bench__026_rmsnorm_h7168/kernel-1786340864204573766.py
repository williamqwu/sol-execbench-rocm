import torch
import triton
import triton.language as tl


@triton.jit
def _rmsnorm(x, w, out, H: tl.constexpr):
    row = tl.program_id(0)
    row_start = row * H
    c0 = tl.arange(0, 4096)
    c1 = 4096 + tl.arange(0, 2048)
    c2 = 6144 + tl.arange(0, 1024)
    x0 = tl.load(x + row_start + c0).to(tl.float32)
    x1 = tl.load(x + row_start + c1).to(tl.float32)
    x2 = tl.load(x + row_start + c2).to(tl.float32)
    total = tl.sum(x0 * x0, axis=0)
    total += tl.sum(x1 * x1, axis=0)
    total += tl.sum(x2 * x2, axis=0)
    variance = total / H
    scale = tl.rsqrt(variance + 1.0e-6)
    w0 = tl.load(w + c0).to(tl.float32)
    w1 = tl.load(w + c1).to(tl.float32)
    w2 = tl.load(w + c2).to(tl.float32)
    tl.store(out + row_start + c0, x0 * scale * w0)
    tl.store(out + row_start + c1, x1 * scale * w1)
    tl.store(out + row_start + c2, x2 * scale * w2)


@torch.no_grad()
def run(hidden_states, weight):
    rows, hidden = hidden_states.shape
    assert hidden == 7168
    out = torch.empty_like(hidden_states)
    _rmsnorm[(rows,)](hidden_states, weight, out, hidden,
                      num_warps=8, waves_per_eu=2)
    return out

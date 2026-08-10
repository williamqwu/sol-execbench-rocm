import torch
import triton
import triton.language as tl


@triton.jit
def _rmsnorm(x, w, out, H: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < H
    xv = tl.load(x + row * H + cols, mask=mask, other=0.0,
                 cache_modifier=".cg",
                 eviction_policy="evict_first").to(tl.float32)
    variance = tl.sum(xv * xv, axis=0) / H
    scale = tl.rsqrt(variance + 1.0e-6)
    weight = tl.load(w + cols, mask=mask, other=0.0,
                     cache_modifier=".ca",
                     eviction_policy="evict_last").to(tl.float32)
    tl.store(out + row * H + cols, xv * scale * weight, mask=mask)


@torch.no_grad()
def run(hidden_states, weight):
    rows, hidden = hidden_states.shape
    assert hidden == 7168
    out = torch.empty_like(hidden_states)
    _rmsnorm[(rows,)](hidden_states, weight, out, hidden,
                      BLOCK=8192, num_warps=8, waves_per_eu=2)
    return out

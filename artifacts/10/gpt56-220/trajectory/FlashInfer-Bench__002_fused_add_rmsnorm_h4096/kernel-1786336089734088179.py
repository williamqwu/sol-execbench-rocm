import torch
import triton
import triton.language as tl


@triton.jit
def _add_rmsnorm(x_ptr, r_ptr, w_ptr, y_ptr, n_rows: tl.constexpr,
                 BLOCK: tl.constexpr):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    off = row * 4096 + cols
    x = tl.load(x_ptr + off).to(tl.float32)
    r = tl.load(r_ptr + off).to(tl.float32)
    z = x + r
    mean_sq = tl.sum(z * z, axis=0) * (1.0 / 4096.0)
    inv = tl.rsqrt(mean_sq + 1.0e-5)
    w = tl.load(w_ptr + cols).to(tl.float32)
    tl.store(y_ptr + off, z * inv * w)


@torch.no_grad()
def run(hidden_states, residual, weight):
    rows, hidden = hidden_states.shape
    assert hidden == 4096
    out = torch.empty_like(hidden_states)
    _add_rmsnorm[(rows,)](
        hidden_states, residual, weight, out, rows,
        BLOCK=4096, num_warps=8,
    )
    return out

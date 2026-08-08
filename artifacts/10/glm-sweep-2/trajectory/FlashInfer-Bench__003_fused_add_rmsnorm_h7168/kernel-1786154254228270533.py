import torch
import triton
import triton.language as tl


@triton.jit
def _add_rmsnorm_fwd(out, hidden, residual, weight, eps, N, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < N
    off = row * N + cols
    h = tl.load(hidden + off, mask=mask, other=0.0).to(tl.float32)
    r = tl.load(residual + off, mask=mask, other=0.0).to(tl.float32)
    x = h + r
    mean_x2 = tl.sum(x * x, axis=0) / N
    inv_rms = 1.0 / tl.sqrt(mean_x2 + eps)
    w = tl.load(weight + cols, mask=mask, other=0.0).to(tl.float32)
    y = (x * inv_rms) * w
    tl.store(out + off, y.to(tl.bfloat16), mask=mask)


@torch.no_grad()
def run(hidden_states, residual, weight):
    _, hidden_size = hidden_states.shape
    assert hidden_size == 7168
    EPS = 1e-6
    N = hidden_size
    out = torch.empty_like(hidden_states)
    BLOCK = 8192
    grid = (hidden_states.shape[0],)
    _add_rmsnorm_fwd[grid](out, hidden_states, residual, weight, EPS, N, BLOCK=BLOCK, num_warps=8)
    return out

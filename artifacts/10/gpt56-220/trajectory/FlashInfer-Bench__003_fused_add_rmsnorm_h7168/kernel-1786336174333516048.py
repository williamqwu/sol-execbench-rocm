import torch
import triton
import triton.language as tl


@triton.jit
def _add_rmsnorm_kernel(hidden, residual, weight, output, n_cols: tl.constexpr,
                        BLOCK: tl.constexpr):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < n_cols
    offsets = row * n_cols + cols
    x = tl.load(hidden + offsets, mask=mask, other=0.0).to(tl.float32)
    x += tl.load(residual + offsets, mask=mask, other=0.0).to(tl.float32)
    mean_square = tl.sum(x * x, axis=0) / n_cols
    inv_rms = tl.rsqrt(mean_square + 1.0e-6)
    w = tl.load(weight + cols, mask=mask, other=0.0).to(tl.float32)
    tl.store(output + offsets, x * inv_rms * w, mask=mask)


@torch.no_grad()
def run(hidden_states, residual, weight):
    rows, hidden_size = hidden_states.shape
    assert hidden_size == 7168
    output = torch.empty_like(hidden_states)
    num_warps = 4 if rows <= 64 else 8
    _add_rmsnorm_kernel[(rows,)](
        hidden_states, residual, weight, output,
        n_cols=7168, BLOCK=8192, num_warps=num_warps,
    )
    return output

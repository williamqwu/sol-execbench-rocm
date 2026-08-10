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


@triton.jit
def _inv_rms_kernel(hidden, residual, inv_rms_out, n_cols: tl.constexpr,
                    BLOCK: tl.constexpr):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < n_cols
    offsets = row * n_cols + cols
    x = tl.load(hidden + offsets, mask=mask, other=0.0).to(tl.float32)
    x += tl.load(residual + offsets, mask=mask, other=0.0).to(tl.float32)
    mean_square = tl.sum(x * x, axis=0) / n_cols
    tl.store(inv_rms_out + row, tl.rsqrt(mean_square + 1.0e-6))


@triton.jit
def _apply_kernel(hidden, residual, weight, inv_rms, output,
                  n_elements: tl.constexpr, n_cols: tl.constexpr,
                  BLOCK: tl.constexpr):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < n_elements
    x = tl.load(hidden + offsets, mask=mask).to(tl.float32)
    x += tl.load(residual + offsets, mask=mask).to(tl.float32)
    w = tl.load(weight + offsets % n_cols, mask=mask).to(tl.float32)
    scale = tl.load(inv_rms + offsets // n_cols, mask=mask)
    tl.store(output + offsets, x * scale * w, mask=mask)


@torch.no_grad()
def run(hidden_states, residual, weight):
    rows, hidden_size = hidden_states.shape
    assert hidden_size == 7168
    output = torch.empty_like(hidden_states)
    if rows > 64:
        inv_rms = torch.empty((rows,), dtype=torch.float32,
                              device=hidden_states.device)
        _inv_rms_kernel[(rows,)](
            hidden_states, residual, inv_rms,
            n_cols=7168, BLOCK=8192, num_warps=8,
        )
        n_elements = rows * hidden_size
        _apply_kernel[(triton.cdiv(n_elements, 256),)](
            hidden_states, residual, weight, inv_rms, output,
            n_elements=n_elements, n_cols=7168, BLOCK=256, num_warps=4,
        )
        return output
    num_warps = 4 if rows <= 64 else 8
    _add_rmsnorm_kernel[(rows,)](
        hidden_states, residual, weight, output,
        n_cols=7168, BLOCK=8192, num_warps=num_warps,
    )
    return output

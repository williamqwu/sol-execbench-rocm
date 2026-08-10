import torch
import triton
import triton.language as tl


@triton.jit
def _add_rmsnorm_kernel(hidden, residual, weight, output,
                        BLOCK: tl.constexpr):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    offsets = row * 2048 + cols
    x = tl.load(hidden + offsets).to(tl.float32)
    x += tl.load(residual + offsets).to(tl.float32)
    sum_sq = tl.sum(x * x, axis=0)
    inv_rms = tl.rsqrt(sum_sq * (1.0 / 2048.0) + 1.0e-6)
    w = tl.load(weight + cols).to(tl.float32)
    tl.store(output + offsets, x * inv_rms * w)


@triton.jit
def _add_rmsnorm_2rows_kernel(hidden, residual, weight, output, n_rows,
                              BLOCK: tl.constexpr):
    first_row = tl.program_id(0) * 2
    cols = tl.arange(0, BLOCK)
    w = tl.load(weight + cols).to(tl.float32)
    for row_offset in tl.static_range(2):
        row = first_row + row_offset
        offsets = row * 2048 + cols
        active = row < n_rows
        x = tl.load(hidden + offsets, mask=active, other=0.0).to(tl.float32)
        x += tl.load(residual + offsets, mask=active, other=0.0).to(tl.float32)
        sum_sq = tl.sum(x * x, axis=0)
        inv_rms = tl.rsqrt(sum_sq * (1.0 / 2048.0) + 1.0e-6)
        tl.store(output + offsets, x * inv_rms * w, mask=active)


@torch.no_grad()
def run(hidden_states, residual, weight):
    n_rows, hidden_size = hidden_states.shape
    assert hidden_size == 2048
    output = torch.empty_like(hidden_states)
    if n_rows >= 256:
        _add_rmsnorm_2rows_kernel[(triton.cdiv(n_rows, 2),)](
            hidden_states, residual, weight, output, n_rows,
            BLOCK=2048, num_warps=8, num_stages=1,
        )
    else:
        _add_rmsnorm_kernel[(n_rows,)](
            hidden_states, residual, weight, output,
            BLOCK=2048, num_warps=8, num_stages=1,
        )
    return output

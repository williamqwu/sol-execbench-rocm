import torch
import triton
import triton.language as tl


@triton.jit
def _rmsnorm(x_ptr, w_ptr, out_ptr, n_rows, x_stride: tl.constexpr,
             BLOCK: tl.constexpr):
    row = tl.program_id(0) * 2
    cols = tl.arange(0, BLOCK)
    x = tl.load(x_ptr + row * x_stride + cols).to(tl.float32)
    w = tl.load(w_ptr + cols).to(tl.float32)
    mean_square = tl.sum(x * x, axis=0) * (1.0 / 512.0)
    inv_rms = tl.rsqrt(mean_square + 1.0e-6)
    tl.store(out_ptr + row * x_stride + cols, x * inv_rms * w)

    x1 = tl.load(x_ptr + (row + 1) * x_stride + cols,
                 mask=row + 1 < n_rows, other=0.0).to(tl.float32)
    mean_square1 = tl.sum(x1 * x1, axis=0) * (1.0 / 512.0)
    inv_rms1 = tl.rsqrt(mean_square1 + 1.0e-6)
    tl.store(out_ptr + (row + 1) * x_stride + cols, x1 * inv_rms1 * w,
             mask=row + 1 < n_rows)


@torch.no_grad()
def run(hidden_states, weight):
    assert hidden_states.shape[1] == 512
    output = torch.empty_like(hidden_states)
    n_rows = hidden_states.shape[0]
    _rmsnorm[(triton.cdiv(n_rows, 2),)](
        hidden_states, weight, output, n_rows, hidden_states.stride(0),
        BLOCK=512, num_warps=1,
    )
    return output

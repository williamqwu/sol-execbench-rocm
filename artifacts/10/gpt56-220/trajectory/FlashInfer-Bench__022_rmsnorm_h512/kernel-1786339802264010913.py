import torch
import triton
import triton.language as tl


@triton.jit
def _rmsnorm(x_ptr, w_ptr, out_ptr, x_stride: tl.constexpr,
             BLOCK: tl.constexpr):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    x = tl.load(x_ptr + row * x_stride + cols).to(tl.float32)
    w = tl.load(w_ptr + cols).to(tl.float32)
    mean_square = tl.sum(x * x, axis=0) * (1.0 / 512.0)
    inv_rms = tl.rsqrt(mean_square + 1.0e-6)
    tl.store(out_ptr + row * x_stride + cols, x * inv_rms * w)


@torch.no_grad()
def run(hidden_states, weight):
    assert hidden_states.shape[1] == 512
    output = torch.empty_like(hidden_states)
    _rmsnorm[(hidden_states.shape[0],)](
        hidden_states, weight, output, hidden_states.stride(0),
        BLOCK=512, num_warps=8,
    )
    return output

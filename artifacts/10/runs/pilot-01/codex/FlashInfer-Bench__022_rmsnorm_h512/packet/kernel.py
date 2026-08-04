import torch
import triton
import triton.language as tl


@triton.jit
def _rmsnorm_h512_kernel(x_ptr, w_ptr, y_ptr, n_rows: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    x = tl.load(x_ptr + row * BLOCK + offs).to(tl.float32)
    w = tl.load(w_ptr + offs).to(tl.float32)
    ss = tl.sum(x * x, axis=0)
    inv_rms = tl.rsqrt(ss * (1.0 / 512.0) + 1.0e-6)
    y = x * inv_rms * w
    tl.store(y_ptr + row * BLOCK + offs, y)


@torch.no_grad()
def run(hidden_states, weight):
    batch_size = hidden_states.shape[0]
    output = torch.empty_like(hidden_states)
    _rmsnorm_h512_kernel[(batch_size,)](
        hidden_states,
        weight,
        output,
        batch_size,
        BLOCK=512,
        num_warps=1,
    )
    return output

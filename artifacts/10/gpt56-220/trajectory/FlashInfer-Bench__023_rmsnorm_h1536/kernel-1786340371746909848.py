import torch
import triton
import triton.language as tl


@triton.jit
def _rmsnorm_kernel(x_ptr, w_ptr, y_ptr, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < 1536
    x = tl.load(x_ptr + row * 1536 + cols, mask=mask, other=0.0,
                cache_modifier=".cg").to(tl.float32)
    variance = tl.sum(x * x, axis=0) * (1.0 / 1536.0)
    inv_rms = tl.rsqrt(variance + 1.0e-6)
    w = tl.load(w_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    tl.store(y_ptr + row * 1536 + cols, x * inv_rms * w, mask=mask)


@torch.no_grad()
def run(hidden_states, weight):
    output = torch.empty_like(hidden_states)
    rows = hidden_states.shape[0]
    _rmsnorm_kernel[(rows,)](hidden_states, weight, output,
                             BLOCK=2048, num_warps=4, waves_per_eu=2)
    return output

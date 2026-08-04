import torch
import triton
import triton.language as tl


@triton.jit
def _rmsnorm_kernel(hidden_states, residual, weight, output, BLOCK:tl.constexpr):
    row = tl.program_id(0)
    offsets = tl.arange(0, BLOCK)
    row_offsets = row * BLOCK + offsets

    h = tl.load(hidden_states + row_offsets, eviction_policy="evict_last").to(tl.float32)
    r = tl.load(residual + row_offsets, eviction_policy="evict_first").to(tl.float32)
    w = tl.load(weight + offsets, eviction_policy="evict_last").to(tl.float32)

    x = h + r
    ss = tl.sum(x * x, axis=0)
    inv_rms = tl.rsqrt(ss / BLOCK + 1.0e-6)
    y = x * inv_rms * w
    tl.store(output + row_offsets, y)


@torch.no_grad()
def run(hidden_states, residual, weight):
    output = hidden_states
    n_rows = hidden_states.shape[0]
    _rmsnorm_kernel[(n_rows,)](
        hidden_states,
        residual,
        weight,
        output,
        BLOCK=2048,
        num_warps=8,
    )
    return output

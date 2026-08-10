import torch
import triton
import triton.language as tl


@triton.jit
def _add_rmsnorm_kernel(hidden, residual, weight, output,
                        BLOCK: tl.constexpr, BF16_ADD: tl.constexpr):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    offsets = row * 2048 + cols
    if BF16_ADD:
        x = (tl.load(hidden + offsets) + tl.load(residual + offsets)).to(tl.float32)
    else:
        x = tl.load(hidden + offsets).to(tl.float32)
        x += tl.load(residual + offsets).to(tl.float32)
    sum_sq = tl.sum(x * x, axis=0)
    inv_rms = tl.rsqrt(sum_sq * (1.0 / 2048.0) + 1.0e-6)
    w = tl.load(weight + cols).to(tl.float32)
    tl.store(output + offsets, x * inv_rms * w)


@torch.no_grad()
def run(hidden_states, residual, weight):
    n_rows, hidden_size = hidden_states.shape
    assert hidden_size == 2048
    output = torch.empty_like(hidden_states)
    _add_rmsnorm_kernel[(n_rows,)](
        hidden_states, residual, weight, output,
        BLOCK=2048, BF16_ADD=n_rows >= 256,
        num_warps=8, num_stages=1, waves_per_eu=1,
    )
    return output

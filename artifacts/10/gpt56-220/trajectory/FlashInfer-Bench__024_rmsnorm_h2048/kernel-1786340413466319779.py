import torch
import triton
import triton.language as tl


@triton.jit
def _rmsnorm_kernel(x, weight, output, stride, eps: tl.constexpr,
                    BLOCK: tl.constexpr):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    x_row = x + row * stride
    values = tl.load(x_row + cols).to(tl.float32)
    mean_square = tl.sum(values * values, axis=0) / BLOCK
    inv_rms = tl.rsqrt(mean_square + eps)
    scales = tl.load(weight + cols).to(tl.float32)
    tl.store(output + row * stride + cols, values * inv_rms * scales)


@torch.no_grad()
def run(hidden_states, weight):
    batch_size, hidden_size = hidden_states.shape
    assert hidden_size == 2048
    output = torch.empty_like(hidden_states)
    num_warps = 4 if batch_size >= 256 else 8
    _rmsnorm_kernel[(batch_size,)](
        hidden_states, weight, output, hidden_states.stride(0),
        eps=1e-6, BLOCK=2048, num_warps=num_warps, waves_per_eu=1,
        num_stages=1,
    )
    return output

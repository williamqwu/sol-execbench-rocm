import torch
import triton
import triton.language as tl


@triton.jit
def _rms_kernel(hidden, residual, weight, output, n_rows: tl.constexpr,
                eps: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    offsets = row * BLOCK + cols
    x = tl.load(hidden + offsets).to(tl.float32) + tl.load(residual + offsets).to(tl.float32)
    variance = tl.sum(x * x, axis=0) * (1.0 / BLOCK)
    inv_rms = tl.rsqrt(variance + eps)
    w = tl.load(weight + cols).to(tl.float32)
    tl.store(output + offsets, x * inv_rms * w)


@torch.no_grad()
def run(hidden_states: torch.Tensor, residual: torch.Tensor,
        weight: torch.Tensor, eps: float) -> torch.Tensor:
    output = torch.empty_like(hidden_states)
    rows = hidden_states.numel() // 8192
    _rms_kernel[(rows,)](
        hidden_states, residual, weight, output, rows, eps,
        BLOCK=8192, num_warps=4,
    )
    return output

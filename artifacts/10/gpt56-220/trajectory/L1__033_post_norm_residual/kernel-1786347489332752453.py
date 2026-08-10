import torch
import triton
import triton.language as tl


@triton.jit
def _rms_residual(x, residual, weight, out, eps: tl.constexpr,
                  BLOCK: tl.constexpr):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    offsets = row * BLOCK + cols
    values = tl.load(x + offsets).to(tl.float32)
    variance = tl.sum(values * values, axis=0) * (1.0 / BLOCK)
    scale = tl.rsqrt(variance + eps)
    w = tl.load(weight + cols).to(tl.float32)
    normalized = (values * scale * w).to(tl.bfloat16)
    r = tl.load(residual + offsets)
    tl.store(out + offsets, r + normalized)


@torch.no_grad()
def run(sublayer_output: torch.Tensor, residual: torch.Tensor,
        weight: torch.Tensor, eps: float) -> torch.Tensor:
    out = torch.empty_like(sublayer_output)
    rows = sublayer_output.numel() // 4096
    _rms_residual[(rows,)](
        sublayer_output, residual, weight, out, eps,
        BLOCK=4096, num_warps=4,
    )
    return out

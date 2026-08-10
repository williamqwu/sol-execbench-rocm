import torch
import triton
import triton.language as tl


@triton.jit
def _rms_residual(x, residual, weight, out, eps: tl.constexpr,
                  BLOCK: tl.constexpr):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    row_start = row * 4096
    offsets0 = row_start + cols
    offsets1 = offsets0 + BLOCK
    offsets2 = offsets1 + BLOCK
    offsets3 = offsets2 + BLOCK
    values0 = tl.load(x + offsets0).to(tl.float32)
    values1 = tl.load(x + offsets1).to(tl.float32)
    values2 = tl.load(x + offsets2).to(tl.float32)
    values3 = tl.load(x + offsets3).to(tl.float32)
    squares = (values0 * values0 + values1 * values1
               + values2 * values2 + values3 * values3)
    variance = tl.sum(squares, axis=0) * (1.0 / 4096.0)
    scale = tl.rsqrt(variance + eps)
    w0 = tl.load(weight + cols).to(tl.float32)
    w1 = tl.load(weight + cols + BLOCK).to(tl.float32)
    w2 = tl.load(weight + cols + 2 * BLOCK).to(tl.float32)
    w3 = tl.load(weight + cols + 3 * BLOCK).to(tl.float32)
    normalized0 = (values0 * scale * w0).to(tl.bfloat16)
    normalized1 = (values1 * scale * w1).to(tl.bfloat16)
    normalized2 = (values2 * scale * w2).to(tl.bfloat16)
    normalized3 = (values3 * scale * w3).to(tl.bfloat16)
    r0 = tl.load(residual + offsets0)
    r1 = tl.load(residual + offsets1)
    r2 = tl.load(residual + offsets2)
    r3 = tl.load(residual + offsets3)
    tl.store(out + offsets0, r0 + normalized0)
    tl.store(out + offsets1, r1 + normalized1)
    tl.store(out + offsets2, r2 + normalized2)
    tl.store(out + offsets3, r3 + normalized3)


@torch.no_grad()
def run(sublayer_output: torch.Tensor, residual: torch.Tensor,
        weight: torch.Tensor, eps: float) -> torch.Tensor:
    out = torch.empty_like(sublayer_output)
    rows = sublayer_output.numel() // 4096
    _rms_residual[(rows,)](
        sublayer_output, residual, weight, out, eps,
        BLOCK=1024, num_warps=8, num_stages=1, waves_per_eu=8,
    )
    return out

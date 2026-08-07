import torch
import triton
import triton.language as tl


@triton.jit
def _rms_norm_kernel(
    hidden_ptr,
    residual_ptr,
    weight_ptr,
    output_ptr,
    eps,
    N,
    H: tl.constexpr,
):
    row = tl.program_id(0)
    if row >= N:
        return

    cols = tl.arange(0, H)
    off = row * H + cols

    # Residual add in bf16 (matches reference: residual + hidden_states), then fp32 for reduction
    x = tl.load(residual_ptr + off).to(tl.float32) + tl.load(hidden_ptr + off).to(tl.float32)

    # variance = mean(x^2) in fp32
    variance = tl.sum(x * x, axis=0) * (1.0 / H)
    rstd = tl.rsqrt(variance + eps)

    # weight * normalized, cast to bf16
    w = tl.load(weight_ptr + cols).to(tl.float32)
    tl.store(output_ptr + off, (w * (x * rstd)).to(tl.bfloat16))


@torch.no_grad()
def run(hidden_states: torch.Tensor, residual: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    H = hidden_states.shape[-1]
    N = hidden_states.numel() // H
    output = torch.empty_like(hidden_states)

    # Tuned: large N (many rows) -> 16 warps; medium/small -> 8 warps
    if N >= 4096:
        num_warps = 16
        num_stages = 3
    else:
        num_warps = 8
        num_stages = 2

    _rms_norm_kernel[(N,)](
        hidden_states,
        residual,
        weight,
        output,
        float(eps),
        N,
        H=H,
        num_warps=num_warps,
        num_stages=num_stages,
    )
    return output

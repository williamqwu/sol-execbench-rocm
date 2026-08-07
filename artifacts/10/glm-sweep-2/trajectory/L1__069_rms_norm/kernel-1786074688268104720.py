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
    N,  # number of rows to process (batch*seq)
    H: tl.constexpr,  # hidden_size = 8192
):
    row = tl.program_id(0)
    if row >= N:
        return

    cols = tl.arange(0, H)
    off = row * H + cols

    # Load both inputs in bf16, add in bf16 (matches reference: residual + hidden_states)
    x = tl.load(residual_ptr + off).to(tl.float32) + tl.load(hidden_ptr + off).to(tl.float32)

    # variance = mean(x^2) in fp32
    x2 = x * x
    variance = tl.sum(x2, axis=0) * (1.0 / H)

    # normalize
    rstd = tl.rsqrt(variance + eps)
    x_norm = x * rstd

    # weight * x_norm, cast to bf16 output
    w = tl.load(weight_ptr + cols).to(tl.float32)
    out = (w * x_norm).to(tl.bfloat16)

    tl.store(output_ptr + off, out)


@torch.no_grad()
def run(hidden_states: torch.Tensor, residual: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    H = hidden_states.shape[-1]
    N = hidden_states.numel() // H
    output = torch.empty_like(hidden_states)
    assert hidden_states.is_contiguous() and residual.is_contiguous()
    assert H == 8192

    _rms_norm_kernel[(N,)](
        hidden_states,
        residual,
        weight,
        output,
        float(eps),
        N,
        H=H,
        num_warps=16,
        num_stages=2,
    )
    return output

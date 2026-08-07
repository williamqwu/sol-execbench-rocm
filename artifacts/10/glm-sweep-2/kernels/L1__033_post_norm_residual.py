import torch
import triton
import triton.language as tl


@triton.jit
def _rmsnorm_residual_kernel(
    sublayer_ptr,
    residual_ptr,
    weight_ptr,
    out_ptr,
    eps,
    H: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < H

    x = tl.load(sublayer_ptr + row * H + cols, mask=mask, other=0.0).to(tl.float32)
    x2 = x * x
    var = tl.sum(x2, axis=0) / H
    rstd = 1.0 / tl.sqrt(var + eps)

    w = tl.load(weight_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    normed = x * rstd * w

    res = tl.load(residual_ptr + row * H + cols, mask=mask, other=0.0).to(tl.float32)
    out = (res + normed).to(tl.bfloat16)
    tl.store(out_ptr + row * H + cols, out, mask=mask)


@torch.no_grad()
def run(sublayer_output: torch.Tensor, residual: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    N = sublayer_output.shape[0] * sublayer_output.shape[1]
    H = sublayer_output.shape[2]
    sub = sublayer_output.reshape(N, H)
    res = residual.reshape(N, H)
    out = torch.empty_like(sub)

    BLOCK = triton.next_power_of_2(H)
    _rmsnorm_residual_kernel[(N,)](sub, res, weight, out, eps, H, BLOCK, num_warps=8)

    return out.view_as(sublayer_output)

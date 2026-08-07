import torch
import triton
import triton.language as tl


@triton.jit
def _rmsnorm_residual_kernel(
    sublayer_ptr,        # [N, H] bf16
    residual_ptr,        # [N, H] bf16
    weight_ptr,          # [H]   bf16
    out_ptr,             # [N, H] bf16
    eps,                 # float
    H: tl.constexpr,     # hidden size
    BLOCK: tl.constexpr, # >= H
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < H

    # Load sublayer row in fp32
    x = tl.load(sublayer_ptr + row * H + cols, mask=mask, other=0.0).to(tl.float32)

    # variance = mean(x^2) over hidden
    x2 = x * x
    # sum then divide
    var = tl.sum(x2, axis=0) / H
    rstd = 1.0 / tl.sqrt(var + eps)

    # normalize + scale
    w = tl.load(weight_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    normed = x * rstd * w

    # add residual, write bf16
    res = tl.load(residual_ptr + row * H + cols, mask=mask, other=0.0).to(tl.float32)
    out = (res + normed).to(tl.bfloat16)
    tl.store(out_ptr + row * H + cols, out, mask=mask)


@torch.no_grad()
def run(sublayer_output: torch.Tensor, residual: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    N = sublayer_output.shape[0] * sublayer_output.shape[1]
    H = sublayer_output.shape[2]
    sub = sublayer_output.reshape(N, H).contiguous()
    res = residual.reshape(N, H).contiguous()
    out = torch.empty_like(sub)

    BLOCK = triton.next_power_of_2(H)
    num_warps = 8
    _rmsnorm_residual_kernel[(N,)](sub, res, weight, out, eps, H, BLOCK, num_warps=num_warps)

    return out.view_as(sublayer_output)

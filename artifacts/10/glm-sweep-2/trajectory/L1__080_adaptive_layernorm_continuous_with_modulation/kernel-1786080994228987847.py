import torch
import triton
import triton.language as tl


@triton.jit
def _fuse_kernel(
    hs_ptr, mean_ptr, rstd_ptr, scale_ptr, shift_ptr, out_ptr,
    D, S,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    batch = pid // S
    offs = tl.arange(0, BLOCK)
    mask = offs < D
    x = tl.load(hs_ptr + pid * D + offs, mask=mask, other=0.0)
    m = tl.load(mean_ptr + pid)
    rstd = tl.load(rstd_ptr + pid)
    s = tl.load(scale_ptr + batch * D + offs, mask=mask, other=0.0)
    t = tl.load(shift_ptr + batch * D + offs, mask=mask, other=0.0)
    norm = (x - m) / rstd
    out = norm * (1.0 + s) + t
    tl.store(out_ptr + pid * D + offs, out, mask=mask)


@torch.no_grad()
def run(
    hidden_states: torch.Tensor,
    temb: torch.Tensor,
    linear_weight: torch.Tensor,
    linear_bias: torch.Tensor,
    eps: float,
):
    B, S, D = hidden_states.shape
    mean = hidden_states.mean(dim=-1, keepdim=True)
    variance = hidden_states.var(dim=-1, keepdim=True, unbiased=False)
    rstd = torch.sqrt(variance + eps)
    modulation = torch.nn.functional.linear(temb, linear_weight, linear_bias)
    scale = modulation[:, :D].contiguous()
    shift = modulation[:, D:].contiguous()
    out = torch.empty_like(hidden_states)
    N = B * S
    BLOCK = triton.next_power_of_2(D)
    _fuse_kernel[(N,)](
        hidden_states, mean, rstd, scale, shift, out,
        D, S, BLOCK, enable_fp_fusion=False,
    )
    return out

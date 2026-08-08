import torch
import torch.nn.functional as F
import triton
import triton.language as tl

@triton.jit
def _norm_kernel(hs_ptr, mean_ptr, rstd_ptr, out_ptr, n_total,
                 D: tl.constexpr, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n_total
    hs = tl.load(hs_ptr + offs, mask=mask, other=0.0)
    row = offs // D
    mean = tl.load(mean_ptr + row, mask=mask, other=0.0)
    rstd = tl.load(rstd_ptr + row, mask=mask, other=0.0)
    out = (hs - mean) / rstd
    tl.store(out_ptr + offs, out, mask=mask)


@triton.jit
def _mod_mul_kernel(hn_ptr, scale_ptr, out_ptr, n_total, sd,
                    D: tl.constexpr, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n_total
    hn = tl.load(hn_ptr + offs, mask=mask, other=0.0)
    batch = offs // sd
    dim = offs % D
    sidx = batch * D + dim
    scale = tl.load(scale_ptr + sidx, mask=mask, other=0.0)
    out = hn * (1.0 + scale)
    tl.store(out_ptr + offs, out, mask=mask)


@triton.jit
def _mod_add_kernel(in_ptr, shift_ptr, out_ptr, n_total, sd,
                    D: tl.constexpr, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n_total
    val = tl.load(in_ptr + offs, mask=mask, other=0.0)
    batch = offs // sd
    dim = offs % D
    sidx = batch * D + dim
    shift = tl.load(shift_ptr + sidx, mask=mask, other=0.0)
    out = val + shift
    tl.store(out_ptr + offs, out, mask=mask)


@torch.no_grad()
def run(
    hidden_states: torch.Tensor,
    temb: torch.Tensor,
    linear_weight: torch.Tensor,
    linear_bias: torch.Tensor,
    proj_out_weight: torch.Tensor,
    proj_out_bias: torch.Tensor,
    eps: float,
):
    B, S, D = hidden_states.shape
    N = B * S
    n_total = N * D
    sd = S * D
    BLOCK = 4096
    grid = (triton.cdiv(n_total, BLOCK),)

    mean = hidden_states.mean(dim=-1, keepdim=True)
    var = hidden_states.var(dim=-1, keepdim=True, unbiased=False)
    rstd = torch.sqrt(var + eps)

    hidden_states_norm = torch.empty_like(hidden_states)
    _norm_kernel[grid](hidden_states, mean, rstd, hidden_states_norm, n_total,
                       D=D, BLOCK=BLOCK)

    temb_silu = temb * torch.sigmoid(temb)
    modulation = F.linear(temb_silu, linear_weight, linear_bias)
    shift = modulation[:, :D].unsqueeze(1).contiguous()
    scale = modulation[:, D:].unsqueeze(1).contiguous()

    tmp = torch.empty_like(hidden_states)
    _mod_mul_kernel[grid](hidden_states_norm, scale, tmp, n_total, sd,
                          D=D, BLOCK=BLOCK)
    hidden_states_mod = torch.empty_like(hidden_states)
    _mod_add_kernel[grid](tmp, shift, hidden_states_mod, n_total, sd,
                          D=D, BLOCK=BLOCK)

    output = F.linear(hidden_states_mod, proj_out_weight, proj_out_bias)
    return output

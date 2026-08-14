import torch
import triton
import triton.language as tl
import torch.nn.functional as F


@triton.jit
def _sincos(ts_ptr, fr_ptr, out_ptr, H: tl.constexpr, BH: tl.constexpr):
    m = tl.program_id(0)
    o = tl.arange(0, BH)
    mk = o < H
    t = tl.load(ts_ptr + m) * 1000.0
    f = tl.load(fr_ptr + o, mask=mk, other=0.0)
    a = t * f
    tl.store(out_ptr + m * 2 * H + o, tl.cos(a), mask=mk)
    tl.store(out_ptr + m * 2 * H + H + o, tl.sin(a), mask=mk)


@triton.jit
def _silu(x_ptr, o_ptr, n, BL: tl.constexpr):
    p = tl.program_id(0) * BL + tl.arange(0, BL)
    m = p < n
    v = tl.load(x_ptr + p, mask=m)
    tl.store(o_ptr + p, v * tl.sigmoid(v), mask=m)


@torch.no_grad()
def run(
    timestep,
    pooled_projections,
    freqs,
    timestep_linear1_weight,
    timestep_linear1_bias,
    timestep_linear2_weight,
    timestep_linear2_bias,
    text_embedder_weight,
    text_embedder_bias,
):
    B = timestep.shape[0]
    H = freqs.shape[0]

    te = torch.empty((B, 2 * H), dtype=torch.float32, device=timestep.device)
    _sincos[(B,)](timestep, freqs, te, H, triton.next_power_of_2(H))

    x = F.linear(te, timestep_linear1_weight, timestep_linear1_bias)
    x = x * torch.sigmoid(x)
    out = F.linear(x, timestep_linear2_weight, timestep_linear2_bias)
    out += F.linear(pooled_projections, text_embedder_weight, text_embedder_bias)
    return out

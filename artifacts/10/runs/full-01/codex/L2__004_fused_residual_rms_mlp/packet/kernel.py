import torch
import torch.nn.functional as F
import aiter
import triton
import triton.language as tl
from triton.language.extra import libdevice


@triton.jit
def _silu_mul_kernel(gate, up, output, n_elements: tl.constexpr,
                     BLOCK: tl.constexpr):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < n_elements
    gate_f32 = tl.load(gate + offsets, mask=mask).to(tl.float32)
    up_f32 = tl.load(up + offsets, mask=mask).to(tl.float32)
    # F.silu returns bf16 before the subsequent bf16 multiply in the spec.
    activated = (gate_f32 / (1.0 + libdevice.exp(-gate_f32))).to(tl.bfloat16)
    result = (activated.to(tl.float32) * up_f32).to(tl.bfloat16)
    tl.store(output + offsets, result, mask=mask)


def _silu_mul(gate, up):
    output = torch.empty_like(gate)
    n = gate.numel()
    _silu_mul_kernel[(triton.cdiv(n, 4096),)](
        gate, up, output, n_elements=n, BLOCK=4096, num_warps=4
    )
    return output


@triton.jit
def _square_kernel(x, squared, n_elements: tl.constexpr,
                   BLOCK: tl.constexpr):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < n_elements
    value = tl.load(x + offsets, mask=mask).to(tl.float32)
    tl.store(squared + offsets, value * value, mask=mask)


@triton.jit
def _rms_finish_kernel(x, variance, weight, output, n_elements: tl.constexpr,
                       hidden_size: tl.constexpr, eps: tl.constexpr,
                       BLOCK: tl.constexpr):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < n_elements
    value = tl.load(x + offsets, mask=mask).to(tl.float32)
    var = tl.load(variance + offsets // hidden_size, mask=mask)
    w = tl.load(weight + offsets % hidden_size, mask=mask).to(tl.float32)
    normalized = value * libdevice.rsqrt(var + eps)
    tl.store(output + offsets, (w * normalized).to(tl.bfloat16), mask=mask)


def _rms_norm(hidden_states, residual, weight, eps):
    x = hidden_states + residual
    squared = torch.empty(x.shape, device=x.device, dtype=torch.float32)
    n = x.numel()
    _square_kernel[(triton.cdiv(n, 8192),)](
        x, squared, n_elements=n, BLOCK=8192, num_warps=8
    )
    variance = squared.mean(dim=-1, keepdim=True)
    output = torch.empty_like(x)
    _rms_finish_kernel[(triton.cdiv(n, 8192),)](
        x, variance, weight, output, n_elements=n, hidden_size=16384,
        eps=eps, BLOCK=8192, num_warps=8
    )
    return output


@triton.jit
def _rms_fused_kernel(hidden_states, residual, weight, output,
                      hidden_size: tl.constexpr, eps: tl.constexpr,
                      BLOCK: tl.constexpr):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    offsets = row * hidden_size + cols
    h = tl.load(hidden_states + offsets).to(tl.float32)
    r = tl.load(residual + offsets).to(tl.float32)
    # The standalone residual add in the reference materializes bf16.
    x = (h + r).to(tl.bfloat16).to(tl.float32)
    variance = tl.sum(x * x, axis=0) / hidden_size
    normalized = x * libdevice.rsqrt(variance + eps)
    w = tl.load(weight + cols).to(tl.float32)
    tl.store(output + offsets, (w * normalized).to(tl.bfloat16))


def _rms_norm_fused(hidden_states, residual, weight, eps, num_warps):
    output = torch.empty_like(hidden_states)
    rows = hidden_states.numel() // hidden_states.shape[-1]
    _rms_fused_kernel[(rows,)](
        hidden_states, residual, weight, output, hidden_size=16384,
        eps=eps, BLOCK=16384, num_warps=num_warps
    )
    return output


def _wide_linear(x, weight):
    m = x.shape[0]
    # The gfx950 assembly kernel is markedly better for the medium-M, very
    # wide projections; hipBLASLt wins at either end of the range.
    if 512 <= m < 3000:
        output = torch.empty(
            (m, weight.shape[0]), device=x.device, dtype=x.dtype
        )
        aiter.gemm_a16w16_asm(x, weight, output)
        return output
    return F.linear(x, weight)


def _down_linear(x, weight):
    m = x.shape[0]
    if (1536 <= m < 1900) or m >= 3000:
        output = torch.empty(
            (m, weight.shape[0]), device=x.device, dtype=x.dtype
        )
        aiter.gemm_a16w16_asm(x, weight, output)
        return output
    return F.linear(x, weight)


@torch.no_grad()
def run(
    hidden_states,
    residual,
    norm_weight,
    gate_proj_weight,
    up_proj_weight,
    down_proj_weight,
    eps,
):
    output_shape = hidden_states.shape
    rows = hidden_states.numel() // hidden_states.shape[-1]
    hidden_states = _rms_norm_fused(
        hidden_states, residual, norm_weight, eps, 16 if rows <= 256 else 8
    )
    hidden_states = hidden_states.reshape(-1, hidden_states.shape[-1])
    gate_output = _wide_linear(hidden_states, gate_proj_weight)
    up_output = _wide_linear(hidden_states, up_proj_weight)
    intermediate = _silu_mul(gate_output, up_output)
    output = _down_linear(intermediate, down_proj_weight)
    return output.reshape(output_shape)

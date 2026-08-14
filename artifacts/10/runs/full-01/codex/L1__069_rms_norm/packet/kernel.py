import torch
import triton
import triton.language as tl


@triton.jit
def _packed_bf16_cast(x):
    return tl.inline_asm_elementwise(
        "v_cvt_pk_bf16_f32 $0, $1, $2",
        "=v,v,v",
        [x],
        dtype=tl.bfloat16,
        is_pure=True,
        pack=2,
    )


@triton.jit
def _rms_norm_kernel(hidden_ptr, residual_ptr, weight_ptr, output_ptr,
                     eps, N: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    offsets = row * N + cols
    mask = cols < N

    # torch's BF16 residual add rounds before the FP32 RMS calculation.
    hidden = tl.load(hidden_ptr + offsets, mask=mask, other=0.0,
                     cache_modifier=".cg", eviction_policy="evict_first")
    residual = tl.load(residual_ptr + offsets, mask=mask, other=0.0,
                       cache_modifier=".cg", eviction_policy="evict_first")
    x = (hidden + residual).to(tl.bfloat16)
    x_fp32 = x.to(tl.float32)
    variance = tl.sum(x_fp32 * x_fp32, axis=0) / N
    inv_rms = tl.rsqrt(variance + eps)

    # The reference rounds normalized values to BF16 before multiplying weight.
    normalized = (x_fp32 * inv_rms).to(tl.bfloat16)
    weight = tl.load(weight_ptr + cols, mask=mask, other=0.0,
                     eviction_policy="evict_last")
    output = _packed_bf16_cast(weight.to(tl.float32) *
                               normalized.to(tl.float32))
    tl.store(output_ptr + offsets, output, mask=mask,
             cache_modifier=".wt")


def run(hidden_states: torch.Tensor, residual: torch.Tensor,
        weight: torch.Tensor, eps: float) -> torch.Tensor:
    n = hidden_states.shape[-1]
    output = torch.empty_like(hidden_states)
    rows = hidden_states.numel() // n
    # More waves shorten the 8192-wide reduction for the very smallest grids;
    # eight waves provide better occupancy once there are more rows.
    num_warps = 16 if rows < 192 else 8
    _rms_norm_kernel[(rows,)](
        hidden_states, residual, weight, output,
        eps, N=n, BLOCK=triton.next_power_of_2(n), num_warps=num_warps,
    )
    return output

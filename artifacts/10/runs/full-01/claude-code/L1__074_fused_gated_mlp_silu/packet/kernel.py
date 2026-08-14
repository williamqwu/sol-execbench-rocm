import torch
import triton
import triton.language as tl


# ---------------------------------------------------------------------------
# Numerics note (why this kernel looks the way it does)
#
# The workload tolerance is rtol = 1.1920929e-07 == 1 ulp of float32, with
# required_matched_ratio = 0.99.  That is far tighter than the intrinsic error
# of an fp32 dot product over K = 1024/4096, so the reference's *exact*
# reduction order has to be reproduced -- a hand-written Triton GEMM (even a
# more accurate one) matches only ~29% of elements.  Both GEMMs are therefore
# left to rocBLAS/hipBLASLt via torch.mm, which is bit-identical to the
# reference's F.linear.
#
# Likewise torch's fp32 exp() is not reproducible by Triton's exp (fast or
# libdevice): using tl.exp for the sigmoid drops the match ratio to 0.9865,
# below the 0.99 threshold.  So sigmoid stays in torch, and only the two
# exactly-rounded fp32 multiplies are fused into Triton -- which is bit-exact
# by construction and removes a full round trip of the (M, I) intermediates.
# ---------------------------------------------------------------------------


@triton.jit
def _gated_mul_kernel(
    UP,             # *f32 (M, 2*I)  row = [gate | up]
    S,              # *f32 (M, I)    sigmoid(gate), computed by torch
    Y,              # *f32 (M, I)    out
    I,
    stride_um,
    stride_sm,
    stride_ym,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    """Y = up * (gate * S), matching the reference's association exactly."""
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    mask_n = rn < I

    up_ptr = UP + rm[:, None] * stride_um
    g = tl.load(up_ptr + rn[None, :], mask=mask_n[None, :], other=0.0)
    u = tl.load(up_ptr + I + rn[None, :], mask=mask_n[None, :], other=0.0)
    s = tl.load(S + rm[:, None] * stride_sm + rn[None, :],
                mask=mask_n[None, :], other=0.0)

    # fp32 multiply is exactly rounded -> identical bits to torch's g*s then u*t
    y = u * (g * s)

    tl.store(Y + rm[:, None] * stride_ym + rn[None, :], y, mask=mask_n[None, :])


def _gated_mul(up_states: torch.Tensor, s: torch.Tensor, I: int) -> torch.Tensor:
    M = up_states.shape[0]
    y = torch.empty((M, I), device=up_states.device, dtype=up_states.dtype)

    BLOCK_N = 1024 if I >= 1024 else max(64, triton.next_power_of_2(I))
    BLOCK_M = 1

    grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(I, BLOCK_N))
    _gated_mul_kernel[grid](
        up_states, s, y, I,
        up_states.stride(0), s.stride(0), y.stride(0),
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N,
        num_warps=4, num_stages=1,
    )
    return y


@torch.no_grad()
def run(hidden_states: torch.Tensor,
        gate_up_weight: torch.Tensor,
        down_weight: torch.Tensor) -> torch.Tensor:
    # hidden_states:  (batch_size, seq_len, hidden_size)
    # gate_up_weight: (2 * intermediate_size, hidden_size)
    # down_weight:    (hidden_size, intermediate_size)
    H = hidden_states.shape[-1]
    lead = hidden_states.shape[:-1]

    x = hidden_states.reshape(-1, H)
    if not x.is_contiguous():
        x = x.contiguous()

    # gate/up projection -- bit-identical to F.linear(hidden_states, gate_up_weight)
    up_states = torch.mm(x, gate_up_weight.t())

    I = up_states.shape[1] // 2
    gate = up_states[:, :I]

    if up_states.is_contiguous():
        s = torch.sigmoid(gate)
        gated = _gated_mul(up_states, s, I)
    else:  # defensive fallback, exactly the reference expression
        up = up_states[:, I:]
        gated = up * (gate * torch.sigmoid(gate))

    output = torch.mm(gated, down_weight.t())
    return output.view(*lead, down_weight.shape[0])

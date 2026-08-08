import torch
import triton
import triton.language as tl


@triton.jit
def _fused_add_rmsnorm_kernel(
    hidden_states_ptr,
    residual_ptr,
    weight_ptr,
    output_ptr,
    n_rows,
    hidden_size,
    stride_hs_row,
    stride_res_row,
    stride_out_row,
    EPS: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    row = tl.program_id(0)
    if row >= n_rows:
        return

    cols = tl.arange(0, BLOCK_SIZE)
    mask = cols < hidden_size

    hs = tl.load(hidden_states_ptr + row * stride_hs_row + cols, mask=mask, other=0.0).to(tl.float32)
    res = tl.load(residual_ptr + row * stride_res_row + cols, mask=mask, other=0.0).to(tl.float32)

    x = hs + res
    # mean of squares
    sum_sq = tl.sum(x * x, axis=0)
    inv_rms = 1.0 / tl.sqrt(sum_sq / hidden_size + EPS)

    w = tl.load(weight_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    y = (x * inv_rms) * w

    tl.store(output_ptr + row * stride_out_row + cols, y.to(tl.bfloat16), mask=mask)


@torch.no_grad()
def run(hidden_states, residual, weight):
    _, hidden_size = hidden_states.shape
    assert hidden_size == 2048

    EPS = 1e-6
    batch_size, _ = hidden_states.shape

    output = torch.empty_like(hidden_states)

    BLOCK_SIZE = triton.next_power_of_2(hidden_size)
    num_warps = 8
    grid = (batch_size,)

    _fused_add_rmsnorm_kernel[grid](
        hidden_states,
        residual,
        weight,
        output,
        batch_size,
        hidden_size,
        hidden_states.stride(0),
        residual.stride(0),
        output.stride(0),
        EPS=EPS,
        BLOCK_SIZE=BLOCK_SIZE,
        num_warps=num_warps,
    )
    return output

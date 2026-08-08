import torch
import triton
import triton.language as tl


@triton.jit
def _fused_add_rmsnorm_kernel(
    hidden_ptr, residual_ptr, weight_ptr, output_ptr,
    stride_h, stride_r, stride_o,
    H: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)

    h = tl.load(hidden_ptr + row * stride_h + cols, mask=cols < H, other=0.0).to(tl.float32)
    r = tl.load(residual_ptr + row * stride_r + cols, mask=cols < H, other=0.0).to(tl.float32)
    x = h + r

    sum_sq = tl.sum(x * x, axis=0)
    inv_rms = 1.0 / tl.sqrt(sum_sq / H + 1e-5)

    w = tl.load(weight_ptr + cols, mask=cols < H, other=0.0).to(tl.float32)
    y = (x * inv_rms) * w
    tl.store(output_ptr + row * stride_o + cols, y.to(tl.bfloat16), mask=cols < H)


@torch.no_grad()
def run(hidden_states, residual, weight):
    _, hidden_size = hidden_states.shape
    assert hidden_size == 4096

    out = torch.empty_like(hidden_states)
    rows = hidden_states.shape[0]
    BLOCK = triton.next_power_of_2(hidden_size)

    grid = (rows,)
    _fused_add_rmsnorm_kernel[grid](
        hidden_states, residual, weight, out,
        hidden_states.stride(0), residual.stride(0), out.stride(0),
        H=hidden_size, BLOCK=BLOCK,
        num_warps=8, num_stages=1,
    )
    return out

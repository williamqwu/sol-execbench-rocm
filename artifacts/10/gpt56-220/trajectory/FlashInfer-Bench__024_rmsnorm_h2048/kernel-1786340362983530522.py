import torch
import triton
import triton.language as tl


@triton.jit
def _rmsnorm_kernel(x, weight, output, n_rows, stride, eps: tl.constexpr,
                    BLOCK: tl.constexpr, ROWS: tl.constexpr):
    first_row = tl.program_id(0) * ROWS
    cols = tl.arange(0, BLOCK)
    scales = tl.load(weight + cols).to(tl.float32)
    for offset in tl.static_range(ROWS):
        row = first_row + offset
        mask = row < n_rows
        values = tl.load(x + row * stride + cols, mask=mask).to(tl.float32)
        mean_square = tl.sum(values * values, axis=0) / BLOCK
        inv_rms = tl.rsqrt(mean_square + eps)
        tl.store(output + row * stride + cols, values * inv_rms * scales,
                 mask=mask)


@torch.no_grad()
def run(hidden_states, weight):
    batch_size, hidden_size = hidden_states.shape
    assert hidden_size == 2048
    output = torch.empty_like(hidden_states)
    num_warps = 4 if batch_size >= 256 else 8
    rows_per_program = 2 if batch_size >= 256 else 1
    grid = (triton.cdiv(batch_size, rows_per_program),)
    _rmsnorm_kernel[grid](
        hidden_states, weight, output, batch_size, hidden_states.stride(0),
        eps=1e-6, BLOCK=2048, ROWS=rows_per_program, num_warps=num_warps,
        waves_per_eu=1, num_stages=1,
    )
    return output

import torch
import triton
import triton.language as tl


@triton.jit
def _rmsnorm_2d_kernel(
    x_ptr,
    w_ptr,
    out_ptr,
    n_rows: tl.constexpr,
    rows_per_program: tl.constexpr,
):
    rows = tl.program_id(0) * rows_per_program + tl.arange(0, rows_per_program)
    cols = tl.arange(0, 512)
    valid = rows[:, None] < n_rows
    offsets = rows[:, None] * 512 + cols[None, :]

    x = tl.load(x_ptr + offsets, mask=valid, other=0.0).to(tl.float32)
    w = tl.load(w_ptr + cols).to(tl.float32)
    mean_square = tl.sum(x * x, axis=1) * (1.0 / 512.0)
    inv_rms = tl.rsqrt(mean_square + 1.0e-6)
    y = x * inv_rms[:, None] * w[None, :]
    tl.store(out_ptr + offsets, y, mask=valid)


@triton.jit
def _rmsnorm_2d_replicated_kernel(
    x_ptr,
    w_ptr,
    out_ptr,
    n_rows: tl.constexpr,
    rows_per_program: tl.constexpr,
):
    rows = tl.program_id(0) * rows_per_program + tl.arange(0, rows_per_program)
    cols = tl.arange(0, 512)
    valid = rows[:, None] < n_rows
    offsets = rows[:, None] * 512 + cols[None, :]

    x = tl.load(x_ptr + offsets, mask=valid, other=0.0).to(tl.float32)
    mean_square = tl.sum(x * x, axis=1) * (1.0 / 512.0)
    inv_rms = tl.rsqrt(mean_square + 1.0e-6)
    weight_offsets = cols[None, :] + rows[:, None] * 0
    w = tl.load(w_ptr + weight_offsets, mask=valid, other=0.0).to(tl.float32)
    tl.store(out_ptr + offsets, x * inv_rms[:, None] * w, mask=valid)


@torch.no_grad()
def run(hidden_states, weight):
    n_rows = hidden_states.shape[0]
    output = torch.empty_like(hidden_states)
    if n_rows >= 8192:
        rows_per_program = 8
        n_programs = triton.cdiv(n_rows, rows_per_program)
        num_warps = 4 if n_rows < 13000 else 8
        _rmsnorm_2d_replicated_kernel[(n_programs,)](
            hidden_states,
            weight,
            output,
            n_rows=n_rows,
            rows_per_program=rows_per_program,
            num_warps=num_warps,
        )
    else:
        rows_per_program = 4
        n_programs = triton.cdiv(n_rows, rows_per_program)
        _rmsnorm_2d_kernel[(n_programs,)](
            hidden_states,
            weight,
            output,
            n_rows=n_rows,
            rows_per_program=rows_per_program,
            num_warps=4,
        )
    return output

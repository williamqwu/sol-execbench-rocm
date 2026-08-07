import torch
import triton
import triton.language as tl


@triton.jit
def _rmsnorm_rows_kernel(x_ptr, w_ptr, out_ptr, n_rows,
                         H: tl.constexpr, ROWS: tl.constexpr,
                         GROUP_STRIDE: tl.constexpr):
    group = tl.program_id(0)
    row_in_group = tl.arange(0, ROWS)
    cols0 = tl.arange(0, 1024)
    cols1 = tl.arange(0, 512)
    weight0 = tl.load(w_ptr + cols0)[None, :].to(tl.float32)
    weight1 = tl.load(w_ptr + 1024 + cols1)[None, :].to(tl.float32)

    while group * ROWS < n_rows:
        rows = group * ROWS + row_in_group
        row_mask = rows < n_rows
        offs0 = rows[:, None] * H + cols0[None, :]
        offs1 = rows[:, None] * H + 1024 + cols1[None, :]
        x0 = tl.load(x_ptr + offs0, mask=row_mask[:, None], other=0.0).to(tl.float32)
        x1 = tl.load(x_ptr + offs1, mask=row_mask[:, None], other=0.0).to(tl.float32)
        square_sum = tl.sum(x0 * x0, axis=1) + tl.sum(x1 * x1, axis=1)
        inv_rms = tl.rsqrt(square_sum / H + 1.0e-6)
        tl.store(out_ptr + offs0, x0 * inv_rms[:, None] * weight0,
                 mask=row_mask[:, None])
        tl.store(out_ptr + offs1, x1 * inv_rms[:, None] * weight1,
                 mask=row_mask[:, None])
        group += GROUP_STRIDE


def run(hidden_states: torch.Tensor, weight: torch.Tensor):
    assert hidden_states.shape[1] == 1536
    output = torch.empty_like(hidden_states)
    n_rows = hidden_states.shape[0]
    n_programs = min(triton.cdiv(n_rows, 2), 1792)
    _rmsnorm_rows_kernel[(n_programs,)](
        hidden_states, weight, output,
        n_rows=n_rows, H=1536, ROWS=2,
        GROUP_STRIDE=n_programs,
        num_warps=1,
        num_stages=2,
        waves_per_eu=4,
    )
    return output

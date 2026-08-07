import torch
import triton
import triton.language as tl


@triton.jit
def _rmsnorm_kernel(
    x_ptr, w_ptr, y_ptr, n_rows, ROWS: tl.constexpr, STREAM: tl.constexpr
):
    pid = tl.program_id(0)
    cols = tl.arange(0, 4096)
    weight = tl.load(w_ptr + cols).to(tl.float32)
    for r in tl.static_range(ROWS):
        row = pid * ROWS + r
        offsets = row * 4096 + cols
        valid = row < n_rows
        if STREAM:
            x = tl.load(
                x_ptr + offsets,
                mask=valid,
                other=0.0,
                cache_modifier=".cg",
            ).to(tl.float32)
        else:
            x = tl.load(x_ptr + offsets, mask=valid, other=0.0).to(
                tl.float32
            )
        sum_sq = tl.sum(x * x, axis=0)
        inv_rms = tl.rsqrt(sum_sq * (1.0 / 4096.0) + 1.0e-5)
        y = (x * inv_rms) * weight
        if STREAM:
            tl.store(
                y_ptr + offsets, y, mask=valid, cache_modifier=".cs"
            )
        else:
            tl.store(y_ptr + offsets, y, mask=valid)


def run(hidden_states, weight):
    output = torch.empty_like(hidden_states)
    batch_size = hidden_states.shape[0]
    if batch_size >= 14000:
        rows = 32 if batch_size >= 14500 else 28
        num_warps, waves_per_eu = 1, 2
    elif batch_size >= 10000:
        rows, num_warps, waves_per_eu = 16, 4, 2
    elif batch_size >= 1000:
        rows, num_warps, waves_per_eu = 8, 2, 4
    else:
        rows, num_warps, waves_per_eu = 1, 8, 1
    _rmsnorm_kernel[(triton.cdiv(batch_size, rows),)](
        hidden_states,
        weight,
        output,
        batch_size,
        rows,
        batch_size >= 1000,
        num_warps=num_warps,
        waves_per_eu=waves_per_eu,
    )
    return output

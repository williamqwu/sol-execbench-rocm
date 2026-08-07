import torch
import triton
import triton.language as tl


@triton.jit
def _rmsnorm_kernel(
    x_ptr,
    w_ptr,
    out_ptr,
    n_rows,
    ROWS_PER_PROGRAM: tl.constexpr,
    FULL_GROUPS: tl.constexpr,
):
    first_row = tl.program_id(0) * ROWS_PER_PROGRAM
    cols = tl.arange(0, 2048)

    # The weights fit in the per-CU cache and stay live across paired rows.
    weight = tl.load(w_ptr + cols, cache_modifier=".ca").to(tl.float32)

    for row_offset in tl.static_range(ROWS_PER_PROGRAM):
        row = first_row + row_offset
        if FULL_GROUPS:
            x = tl.load(
                x_ptr + row * 2048 + cols,
                cache_modifier=".cg",
            ).to(tl.float32)
        else:
            valid = row < n_rows
            x = tl.load(
                x_ptr + row * 2048 + cols,
                mask=valid,
                other=0.0,
                cache_modifier=".cg",
            ).to(tl.float32)

        variance = tl.sum(x * x, axis=0) * (1.0 / 2048.0)
        inv_rms = tl.rsqrt(variance + 1.0e-6)
        out = (x * inv_rms) * weight

        if FULL_GROUPS:
            tl.store(
                out_ptr + row * 2048 + cols,
                out,
                cache_modifier=".wt",
            )
        else:
            tl.store(
                out_ptr + row * 2048 + cols,
                out,
                mask=valid,
                cache_modifier=".wt",
            )


def run(hidden_states, weight):
    output = torch.empty_like(hidden_states)
    rows = hidden_states.shape[0]

    rows_per_program = 1 if rows < 128 else 2
    num_warps = 4 if rows >= 15000 else 2
    _rmsnorm_kernel[((rows + rows_per_program - 1) // rows_per_program,)](
        hidden_states,
        weight,
        output,
        rows,
        ROWS_PER_PROGRAM=rows_per_program,
        FULL_GROUPS=(rows % rows_per_program == 0),
        num_warps=num_warps,
    )
    return output

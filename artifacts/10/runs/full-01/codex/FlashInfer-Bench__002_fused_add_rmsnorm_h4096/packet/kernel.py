import torch
import triton
import triton.language as tl


HIDDEN_SIZE = 4096


@triton.jit
def _fused_add_rmsnorm_kernel(
    hidden_states,
    residual,
    weight,
    output,
    BLOCK_SIZE: tl.constexpr,
):
    row = tl.program_id(0)
    columns = tl.arange(0, BLOCK_SIZE)
    offsets = row * BLOCK_SIZE + columns

    # The two row inputs are streaming. Bypassing the per-CU cache prevents
    # them from displacing the 8 KiB weight vector shared by every row.
    hidden = tl.load(
        hidden_states + offsets, cache_modifier=".cg"
    ).to(tl.float32)
    res = tl.load(
        residual + offsets, cache_modifier=".cg"
    ).to(tl.float32)

    x = hidden + res
    mean_square = tl.sum(x * x, axis=0) * (1.0 / BLOCK_SIZE)
    inv_rms = tl.rsqrt(mean_square + 1.0e-5)
    scale = tl.load(weight + columns, cache_modifier=".ca").to(tl.float32)
    y = (x * inv_rms) * scale

    # Write-through avoids allocating this one-use output stream in cache.
    tl.store(output + offsets, y, cache_modifier=".wt")


@torch.no_grad()
def run(hidden_states, residual, weight):
    output = torch.empty_like(hidden_states)
    rows = hidden_states.numel() // HIDDEN_SIZE

    if rows >= 1024:
        _fused_add_rmsnorm_kernel[(rows,)](
            hidden_states,
            residual,
            weight,
            output,
            BLOCK_SIZE=HIDDEN_SIZE,
            num_warps=4,
            waves_per_eu=8,
        )
    else:
        _fused_add_rmsnorm_kernel[(rows,)](
            hidden_states,
            residual,
            weight,
            output,
            BLOCK_SIZE=HIDDEN_SIZE,
            num_warps=8,
        )
    return output

import torch
import triton
import triton.language as tl


HIDDEN = 2560


@triton.jit
def _fused_rows_kernel(
    grad_output,
    normalized,
    rstd,
    weight,
    grad_hidden,
    grad_residual,
    partials,
    n_rows: tl.constexpr,
    GROUP: tl.constexpr,
    BLOCK: tl.constexpr,
):
    group_id = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    col_mask = cols < 2560
    w = tl.load(weight + cols, mask=col_mask, other=0.0)
    dw = tl.zeros((BLOCK,), dtype=tl.float32)

    for i in tl.static_range(0, GROUP):
        row = group_id * GROUP + i
        mask = col_mask & (row < n_rows)
        offsets = row * 2560 + cols
        go = tl.load(grad_output + offsets, mask=mask, other=0.0).to(tl.float32)
        norm = tl.load(normalized + offsets, mask=mask, other=0.0)
        dw += go * norm

        grad_norm = go * w
        dot = tl.sum(grad_norm * norm, axis=0)
        mean = dot * (1.0 / 2560)
        rs = tl.load(rstd + row, mask=row < n_rows, other=0.0)
        dx = rs * (grad_norm - mean * norm)
        tl.store(grad_hidden + offsets, dx, mask=mask)
        tl.store(grad_residual + offsets, dx, mask=mask)

    tl.store(partials + group_id * 2560 + cols, dw, mask=col_mask)


@triton.jit
def _reduce_partials_kernel(
    partials,
    reduced,
    n_partials: tl.constexpr,
    ROW_BLOCK: tl.constexpr,
    COL_BLOCK: tl.constexpr,
):
    out_row = tl.program_id(0)
    col_block = tl.program_id(1)
    rows = out_row * ROW_BLOCK + tl.arange(0, ROW_BLOCK)
    cols = col_block * COL_BLOCK + tl.arange(0, COL_BLOCK)
    offsets = rows[:, None] * 2560 + cols[None, :]
    mask = (rows[:, None] < n_partials) & (cols[None, :] < 2560)
    values = tl.load(partials + offsets, mask=mask, other=0.0)
    result = tl.sum(values, axis=0)
    tl.store(reduced + out_row * 2560 + cols, result, mask=cols < 2560)


@triton.jit
def _dw_finish_kernel(
    reduced,
    grad_weight,
    n_reduced: tl.constexpr,
    COL_BLOCK: tl.constexpr,
    BLOCK: tl.constexpr,
):
    col_block = tl.program_id(0)
    cols = col_block * COL_BLOCK + tl.arange(0, COL_BLOCK)
    rows = tl.arange(0, BLOCK)
    values = tl.load(
        reduced + rows[:, None] * 2560 + cols[None, :],
        mask=(rows[:, None] < n_reduced) & (cols[None, :] < 2560),
        other=0.0,
    )
    tl.store(
        grad_weight + cols,
        tl.sum(values, axis=0),
        mask=cols < 2560,
    )


def run(
    grad_output: torch.Tensor,
    x: torch.Tensor,
    normalized: torch.Tensor,
    rstd: torch.Tensor,
    weight: torch.Tensor,
):
    n_rows = grad_output.numel() // HIDDEN
    grad_hidden = torch.empty_like(grad_output)
    grad_residual = torch.empty_like(grad_output)
    grad_weight = torch.empty((HIDDEN,), device=grad_output.device, dtype=torch.float32)

    if n_rows <= 512:
        group = 2
        row_warps = 8
    elif n_rows <= 4096:
        group = 4
        row_warps = 8
    else:
        group = 4
        row_warps = 4

    n_partials = triton.cdiv(n_rows, group)
    partials = torch.empty(
        (n_partials, HIDDEN), device=grad_output.device, dtype=torch.float32
    )
    _fused_rows_kernel[(n_partials,)](
        grad_output,
        normalized,
        rstd,
        weight,
        grad_hidden,
        grad_residual,
        partials,
        n_rows=n_rows,
        GROUP=group,
        BLOCK=4096,
        num_warps=row_warps,
    )

    if n_partials >= 8192:
        reduce_rows = 512
        reduce_cols = 64
    else:
        reduce_rows = 256
        reduce_cols = 32
    n_reduced = triton.cdiv(n_partials, reduce_rows)
    reduced = torch.empty(
        (n_reduced, HIDDEN), device=grad_output.device, dtype=torch.float32
    )
    _reduce_partials_kernel[(n_reduced, triton.cdiv(HIDDEN, reduce_cols))](
        partials,
        reduced,
        n_partials=n_partials,
        ROW_BLOCK=reduce_rows,
        COL_BLOCK=reduce_cols,
        num_warps=4,
    )
    _dw_finish_kernel[(triton.cdiv(HIDDEN, 32),)](
        reduced,
        grad_weight,
        n_reduced=n_reduced,
        COL_BLOCK=32,
        BLOCK=triton.next_power_of_2(n_reduced),
        num_warps=4,
    )
    return grad_hidden, grad_residual, grad_weight

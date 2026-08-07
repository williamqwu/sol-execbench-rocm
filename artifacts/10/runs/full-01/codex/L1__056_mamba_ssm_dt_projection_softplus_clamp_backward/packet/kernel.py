import torch
import triton
import triton.language as tl


@triton.jit
def _grad_and_partial_kernel(
    grad_output,
    dt_with_bias,
    dt_activated,
    grad_dt,
    partial,
    n_rows,
    time_step_min,
    time_step_max,
    BLOCK_R: tl.constexpr,
    N_HEADS: tl.constexpr,
    BLOCK_H: tl.constexpr,
):
    pid = tl.program_id(0)
    pid_h = tl.program_id(1)
    rows = pid * BLOCK_R + tl.arange(0, BLOCK_R)
    heads = pid_h * BLOCK_H + tl.arange(0, BLOCK_H)
    offsets = rows[:, None] * N_HEADS + heads[None, :]
    mask = (rows[:, None] < n_rows) & (heads[None, :] < N_HEADS)

    go = tl.load(grad_output + offsets, mask=mask, other=0.0).to(tl.float32)
    x = tl.load(dt_with_bias + offsets, mask=mask, other=0.0).to(tl.float32)
    activated = tl.load(dt_activated + offsets, mask=mask, other=0.0).to(tl.float32)

    inside = (activated > time_step_min) & (activated < time_step_max)
    sigmoid = 1.0 / (1.0 + tl.exp(-x))
    grad = tl.where(inside & mask, go * sigmoid, 0.0)

    tl.store(grad_dt + offsets, grad, mask=mask)
    sums = tl.sum(grad, axis=0)
    tl.store(partial + pid * N_HEADS + heads, sums, mask=heads < N_HEADS)


@triton.jit
def _grad_and_partial_32_8_kernel(
    grad_output,
    dt_with_bias,
    dt_activated,
    grad_dt,
    partial,
    n_rows,
    time_step_min,
    time_step_max,
    BLOCK_R: tl.constexpr,
    ATOMIC: tl.constexpr,
):
    pid = tl.program_id(0)
    rows = pid * BLOCK_R + tl.arange(0, BLOCK_R)

    heads0 = tl.arange(0, 32)
    offsets0 = rows[:, None] * 40 + heads0[None, :]
    mask0 = rows[:, None] < n_rows
    go0 = tl.load(grad_output + offsets0, mask=mask0, other=0.0).to(tl.float32)
    x0 = tl.load(dt_with_bias + offsets0, mask=mask0, other=0.0).to(tl.float32)
    activated0 = tl.load(dt_activated + offsets0, mask=mask0, other=0.0).to(tl.float32)
    inside0 = (activated0 > time_step_min) & (activated0 < time_step_max)
    sigmoid0 = 1.0 / (1.0 + tl.exp(-x0))
    grad0 = tl.where(inside0 & mask0, go0 * sigmoid0, 0.0)
    tl.store(grad_dt + offsets0, grad0, mask=mask0)
    sums0 = tl.sum(grad0, axis=0)
    if ATOMIC:
        tl.atomic_add(partial + heads0, sums0)
    else:
        tl.store(partial + pid * 40 + heads0, sums0)

    heads1 = 32 + tl.arange(0, 8)
    offsets1 = rows[:, None] * 40 + heads1[None, :]
    mask1 = rows[:, None] < n_rows
    go1 = tl.load(grad_output + offsets1, mask=mask1, other=0.0).to(tl.float32)
    x1 = tl.load(dt_with_bias + offsets1, mask=mask1, other=0.0).to(tl.float32)
    activated1 = tl.load(dt_activated + offsets1, mask=mask1, other=0.0).to(tl.float32)
    inside1 = (activated1 > time_step_min) & (activated1 < time_step_max)
    sigmoid1 = 1.0 / (1.0 + tl.exp(-x1))
    grad1 = tl.where(inside1 & mask1, go1 * sigmoid1, 0.0)
    tl.store(grad_dt + offsets1, grad1, mask=mask1)
    sums1 = tl.sum(grad1, axis=0)
    if ATOMIC:
        tl.atomic_add(partial + heads1, sums1)
    else:
        tl.store(partial + pid * 40 + heads1, sums1)


@triton.jit
def _grad_and_partial_8x5_kernel(
    grad_output,
    dt_with_bias,
    dt_activated,
    grad_dt,
    partial,
    n_rows,
    time_step_min,
    time_step_max,
    BLOCK_R: tl.constexpr,
):
    pid = tl.program_id(0)
    rows = pid * BLOCK_R + tl.arange(0, BLOCK_R)
    for head_start in tl.static_range(0, 40, 8):
        heads = head_start + tl.arange(0, 8)
        offsets = rows[:, None] * 40 + heads[None, :]
        mask = rows[:, None] < n_rows
        go = tl.load(grad_output + offsets, mask=mask, other=0.0).to(tl.float32)
        x = tl.load(dt_with_bias + offsets, mask=mask, other=0.0).to(tl.float32)
        activated = tl.load(dt_activated + offsets, mask=mask, other=0.0).to(tl.float32)
        inside = (activated > time_step_min) & (activated < time_step_max)
        grad = tl.where(inside & mask, go / (1.0 + tl.exp(-x)), 0.0)
        tl.store(grad_dt + offsets, grad, mask=mask)
        tl.store(partial + pid * 40 + heads, tl.sum(grad, axis=0))


@triton.jit
def _grad_and_partial_16_16_8_kernel(
    grad_output,
    dt_with_bias,
    dt_activated,
    grad_dt,
    partial,
    n_rows,
    time_step_min,
    time_step_max,
    BLOCK_R: tl.constexpr,
):
    pid = tl.program_id(0)
    rows = pid * BLOCK_R + tl.arange(0, BLOCK_R)
    for head_start in tl.static_range(0, 32, 16):
        heads = head_start + tl.arange(0, 16)
        offsets = rows[:, None] * 40 + heads[None, :]
        mask = rows[:, None] < n_rows
        go = tl.load(grad_output + offsets, mask=mask, other=0.0).to(tl.float32)
        x = tl.load(dt_with_bias + offsets, mask=mask, other=0.0).to(tl.float32)
        activated = tl.load(dt_activated + offsets, mask=mask, other=0.0).to(tl.float32)
        inside = (activated > time_step_min) & (activated < time_step_max)
        grad = tl.where(inside & mask, go / (1.0 + tl.exp(-x)), 0.0)
        tl.store(grad_dt + offsets, grad, mask=mask)
        tl.store(partial + pid * 40 + heads, tl.sum(grad, axis=0))

    heads = 32 + tl.arange(0, 8)
    offsets = rows[:, None] * 40 + heads[None, :]
    mask = rows[:, None] < n_rows
    go = tl.load(grad_output + offsets, mask=mask, other=0.0).to(tl.float32)
    x = tl.load(dt_with_bias + offsets, mask=mask, other=0.0).to(tl.float32)
    activated = tl.load(dt_activated + offsets, mask=mask, other=0.0).to(tl.float32)
    inside = (activated > time_step_min) & (activated < time_step_max)
    grad = tl.where(inside & mask, go / (1.0 + tl.exp(-x)), 0.0)
    tl.store(grad_dt + offsets, grad, mask=mask)
    tl.store(partial + pid * 40 + heads, tl.sum(grad, axis=0))


@triton.jit
def _finish_bias_kernel(partial, grad_dt_bias, n_parts, BLOCK_P: tl.constexpr):
    head = tl.program_id(0)
    parts = tl.arange(0, BLOCK_P)
    values = tl.load(partial + parts * 40 + head, mask=parts < n_parts, other=0.0)
    total = tl.sum(values, axis=0)
    tl.store(grad_dt_bias + head, total)


@triton.jit
def _finish_bias_tiled_kernel(
    partial,
    grad_dt_bias,
    n_parts,
    BLOCK_P: tl.constexpr,
    BLOCK_H: tl.constexpr,
):
    pid_h = tl.program_id(0)
    parts = tl.arange(0, BLOCK_P)
    heads = pid_h * BLOCK_H + tl.arange(0, BLOCK_H)
    mask = (parts[:, None] < n_parts) & (heads[None, :] < 40)
    values = tl.load(partial + parts[:, None] * 40 + heads[None, :], mask=mask, other=0.0)
    total = tl.sum(values, axis=0)
    tl.store(grad_dt_bias + heads, total, mask=heads < 40)


def run(
    grad_output: torch.Tensor,
    dt_with_bias: torch.Tensor,
    dt_activated: torch.Tensor,
    time_step_min: float,
    time_step_max: float,
):
    n_rows = grad_output.numel() // 40
    grad_dt = torch.empty_like(grad_output)
    grad_dt_bias = torch.empty((40,), device=grad_output.device, dtype=torch.bfloat16)

    # For short inputs, one tile covers every row.  Its partial sum is already
    # final, so write it straight to the bf16 bias output and avoid a second
    # kernel launch.
    if n_rows <= 256:
        block_r, block_h, num_warps = 256, 8, 4
    elif n_rows <= 512:
        block_r, block_h, num_warps = 512, 4, 4
    elif n_rows <= 1024:
        block_r, block_h, num_warps = 1024, 2, 8
    elif n_rows <= 1536:
        block_r, block_h, num_warps = 2048, 1, 4
    else:
        block_r = 0

    if block_r:
        _grad_and_partial_kernel[(1, triton.cdiv(40, block_h))](
            grad_output,
            dt_with_bias,
            dt_activated,
            grad_dt,
            grad_dt_bias,
            n_rows,
            time_step_min,
            time_step_max,
            BLOCK_R=block_r,
            N_HEADS=40,
            BLOCK_H=block_h,
            num_warps=num_warps,
        )
        return grad_dt, grad_dt_bias

    # Narrow head tiles avoid padded sigmoid work at sizes where they still
    # provide enough coalescing.  The row sizes keep the grid near full CU
    # waves.  This path also wins at 64K rows, where it makes exactly 768 CTAs.
    if n_rows <= 2500:
        block_r, block_h, num_warps = 512, 8, 8
    elif n_rows <= 4096:
        block_r, block_h, num_warps = 128, 32, 4
    elif n_rows <= 18000:
        block_r, block_h, num_warps = 256, 16, 4
    elif n_rows <= 25000:
        block_r, block_h, num_warps = 512, 8, 8
    elif n_rows <= 45000:
        block_r, block_h, num_warps = 256, 16, 4
    elif n_rows <= 60000:
        block_r, block_h, num_warps = 128, 64, 4
    elif n_rows <= 70000:
        block_r, block_h, num_warps = 256, 16, 4
    else:
        block_r, block_h, num_warps = 128, 64, 4

    n_parts = triton.cdiv(n_rows, block_r)
    partial = torch.empty((n_parts, 40), device=grad_output.device, dtype=torch.float32)

    _grad_and_partial_kernel[(n_parts, triton.cdiv(40, block_h))](
        grad_output,
        dt_with_bias,
        dt_activated,
        grad_dt,
        partial,
        n_rows,
        time_step_min,
        time_step_max,
        BLOCK_R=block_r,
        N_HEADS=40,
        BLOCK_H=block_h,
        num_warps=num_warps,
    )
    _finish_bias_kernel[(40,)](
        partial,
        grad_dt_bias,
        n_parts,
        BLOCK_P=triton.next_power_of_2(n_parts),
        num_warps=1 if n_parts <= 512 else 8,
    )
    return grad_dt, grad_dt_bias

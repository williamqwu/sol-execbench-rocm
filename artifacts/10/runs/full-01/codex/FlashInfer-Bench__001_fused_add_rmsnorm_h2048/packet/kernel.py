import torch
import triton
import triton.language as tl


@triton.jit
def _fused_add_rmsnorm_kernel(
    hidden_states,
    residual,
    weight,
    output,
    stride_row: tl.constexpr,
    H: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, H)
    offsets = row * stride_row + cols

    x = tl.load(hidden_states + offsets).to(tl.float32)
    x += tl.load(residual + offsets).to(tl.float32)
    w = tl.load(weight + cols).to(tl.float32)

    mean_square = tl.sum(x * x, axis=0) * (1.0 / H)
    inv_rms = tl.rsqrt(mean_square + 1.0e-6)
    tl.store(output + offsets, x * inv_rms * w)


@triton.jit
def _fused_add_rmsnorm_rows_kernel(
    hidden_states,
    residual,
    weight,
    output,
    n_rows,
    H: tl.constexpr,
    BLOCK_ROWS: tl.constexpr,
):
    row_ids = tl.program_id(0) * BLOCK_ROWS + tl.arange(0, BLOCK_ROWS)
    cols = tl.arange(0, H)
    offsets = row_ids[:, None] * H + cols[None, :]
    mask = row_ids[:, None] < n_rows

    x = tl.load(hidden_states + offsets, mask=mask, other=0.0).to(tl.float32)
    x += tl.load(residual + offsets, mask=mask, other=0.0).to(tl.float32)
    mean_square = tl.sum(x * x, axis=1) * (1.0 / H)
    inv_rms = tl.rsqrt(mean_square + 1.0e-6)
    w = tl.load(weight + cols).to(tl.float32)
    tl.store(output + offsets, x * inv_rms[:, None] * w[None, :], mask=mask)


@triton.jit
def _fused_add_rmsnorm_persistent_kernel(
    hidden_states,
    residual,
    weight,
    output,
    n_rows,
    H: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, H)
    step = tl.num_programs(0)
    while row < n_rows:
        offsets = row * H + cols
        x = tl.load(hidden_states + offsets).to(tl.float32)
        x += tl.load(residual + offsets).to(tl.float32)
        mean_square = tl.sum(x * x, axis=0) * (1.0 / H)
        inv_rms = tl.rsqrt(mean_square + 1.0e-6)
        w = tl.load(weight + cols).to(tl.float32)
        tl.store(output + offsets, x * inv_rms * w)
        row += step


@triton.jit
def _fused_add_rmsnorm_persistent_cached_kernel(
    hidden_states,
    residual,
    weight,
    output,
    n_rows,
    H: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, H)
    step = tl.num_programs(0)
    w = tl.load(weight + cols).to(tl.float32)
    while row < n_rows:
        offsets = row * H + cols
        x = tl.load(hidden_states + offsets).to(tl.float32)
        x += tl.load(residual + offsets).to(tl.float32)
        mean_square = tl.sum(x * x, axis=0) * (1.0 / H)
        inv_rms = tl.rsqrt(mean_square + 1.0e-6)
        tl.store(output + offsets, x * inv_rms * w)
        row += step


@triton.jit
def _fused_add_rmsnorm_persistent_cg_kernel(
    hidden_states,
    residual,
    weight,
    output,
    n_rows,
    H: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, H)
    step = tl.num_programs(0)
    while row < n_rows:
        offsets = row * H + cols
        x = tl.load(hidden_states + offsets, cache_modifier=".cg").to(tl.float32)
        x += tl.load(residual + offsets, cache_modifier=".cg").to(tl.float32)
        mean_square = tl.sum(x * x, axis=0) * (1.0 / H)
        inv_rms = tl.rsqrt(mean_square + 1.0e-6)
        w = tl.load(weight + cols, eviction_policy="evict_last").to(tl.float32)
        tl.store(output + offsets, x * inv_rms * w)
        row += step


@triton.jit
def _fused_add_rmsnorm_persistent_rows_kernel(
    hidden_states,
    residual,
    weight,
    output,
    n_rows,
    H: tl.constexpr,
    BLOCK_ROWS: tl.constexpr,
):
    row_base = tl.program_id(0) * BLOCK_ROWS
    row_delta = tl.arange(0, BLOCK_ROWS)
    cols = tl.arange(0, H)
    step = tl.num_programs(0) * BLOCK_ROWS
    while row_base < n_rows:
        row_ids = row_base + row_delta
        offsets = row_ids[:, None] * H + cols[None, :]
        mask = row_ids[:, None] < n_rows
        x = tl.load(hidden_states + offsets, mask=mask, other=0.0).to(tl.float32)
        x += tl.load(residual + offsets, mask=mask, other=0.0).to(tl.float32)
        mean_square = tl.sum(x * x, axis=1) * (1.0 / H)
        inv_rms = tl.rsqrt(mean_square + 1.0e-6)
        w = tl.load(weight + cols).to(tl.float32)
        tl.store(output + offsets, x * inv_rms[:, None] * w[None, :], mask=mask)
        row_base += step


@triton.jit
def _fused_add_rmsnorm_pipelined_rows_kernel(
    hidden_states,
    residual,
    weight,
    output,
    n_rows,
    H: tl.constexpr,
    BLOCK_ROWS: tl.constexpr,
    PIPE_STAGES: tl.constexpr,
    UNROLL: tl.constexpr,
):
    row_delta = tl.arange(0, BLOCK_ROWS)
    cols = tl.arange(0, H)
    start = tl.program_id(0) * BLOCK_ROWS
    step = tl.num_programs(0) * BLOCK_ROWS
    for row_base in tl.range(
        start,
        n_rows,
        step,
        num_stages=PIPE_STAGES,
        loop_unroll_factor=UNROLL,
    ):
        row_ids = row_base + row_delta
        offsets = row_ids[:, None] * H + cols[None, :]
        mask = row_ids[:, None] < n_rows
        x = tl.load(hidden_states + offsets, mask=mask, other=0.0).to(tl.float32)
        x += tl.load(residual + offsets, mask=mask, other=0.0).to(tl.float32)
        mean_square = tl.sum(x * x, axis=1) * (1.0 / H)
        inv_rms = tl.rsqrt(mean_square + 1.0e-6)
        w = tl.load(weight + cols).to(tl.float32)
        tl.store(output + offsets, x * inv_rms[:, None] * w[None, :], mask=mask)


@triton.jit
def _fused_add_rmsnorm_persistent_rows_cg_kernel(
    hidden_states,
    residual,
    weight,
    output,
    n_rows,
    H: tl.constexpr,
    BLOCK_ROWS: tl.constexpr,
):
    row_base = tl.program_id(0) * BLOCK_ROWS
    row_delta = tl.arange(0, BLOCK_ROWS)
    cols = tl.arange(0, H)
    step = tl.num_programs(0) * BLOCK_ROWS
    while row_base < n_rows:
        row_ids = row_base + row_delta
        offsets = row_ids[:, None] * H + cols[None, :]
        mask = row_ids[:, None] < n_rows
        x = tl.load(
            hidden_states + offsets,
            mask=mask,
            other=0.0,
            cache_modifier=".cg",
            eviction_policy="evict_first",
        ).to(tl.float32)
        x += tl.load(
            residual + offsets,
            mask=mask,
            other=0.0,
            cache_modifier=".cg",
            eviction_policy="evict_first",
        ).to(tl.float32)
        mean_square = tl.sum(x * x, axis=1) * (1.0 / H)
        inv_rms = tl.rsqrt(mean_square + 1.0e-6)
        w = tl.load(weight + cols, eviction_policy="evict_last").to(tl.float32)
        tl.store(output + offsets, x * inv_rms[:, None] * w[None, :], mask=mask)
        row_base += step


@torch.no_grad()
def run(hidden_states, residual, weight):
    batch_size, hidden_size = hidden_states.shape
    assert hidden_size == 2048
    output = torch.empty_like(hidden_states)
    if batch_size >= 1000:
        _fused_add_rmsnorm_persistent_rows_cg_kernel[(384,)](
            hidden_states,
            residual,
            weight,
            output,
            batch_size,
            H=2048,
            BLOCK_ROWS=4,
            num_warps=4,
            waves_per_eu=4,
        )
    else:
        _fused_add_rmsnorm_kernel[(batch_size,)](
            hidden_states,
            residual,
            weight,
            output,
            hidden_states.stride(0),
            H=2048,
            num_warps=1,
            waves_per_eu=4,
        )
    return output

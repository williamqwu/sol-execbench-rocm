import torch
import triton
import triton.language as tl


@triton.jit
def _fused_add_rmsnorm_kernel(
    hidden_states,
    residual,
    weight,
    output,
    BLOCK_SIZE: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK_SIZE)
    mask = cols < 7168
    offsets = row * 7168 + cols

    hidden = tl.load(hidden_states + offsets, mask=mask, other=0.0).to(tl.float32)
    skip = tl.load(residual + offsets, mask=mask, other=0.0).to(tl.float32)
    x = hidden + skip
    mean_square = tl.sum(x * x, axis=0) * (1.0 / 7168.0)
    inv_rms = tl.rsqrt(mean_square + 1.0e-6)
    scale = tl.load(weight + cols, mask=mask, other=0.0).to(tl.float32)
    result = (x * inv_rms) * scale
    tl.store(output + offsets, result, mask=mask)


@triton.jit
def _load_added(hidden_states, residual, offsets):
    return (
        tl.load(hidden_states + offsets, cache_modifier=".cg",
                eviction_policy="evict_first").to(tl.float32)
        + tl.load(residual + offsets, cache_modifier=".cg",
                  eviction_policy="evict_first").to(tl.float32)
    )


@triton.jit
def _fused_add_rmsnorm_exact_kernel(hidden_states, residual, weight, output):
    row_base = tl.program_id(0) * 7168
    cols = tl.arange(0, 512)

    x0 = _load_added(hidden_states, residual, row_base + cols)
    x1 = _load_added(hidden_states, residual, row_base + cols + 512)
    x2 = _load_added(hidden_states, residual, row_base + cols + 1024)
    x3 = _load_added(hidden_states, residual, row_base + cols + 1536)
    x4 = _load_added(hidden_states, residual, row_base + cols + 2048)
    x5 = _load_added(hidden_states, residual, row_base + cols + 2560)
    x6 = _load_added(hidden_states, residual, row_base + cols + 3072)
    x7 = _load_added(hidden_states, residual, row_base + cols + 3584)
    x8 = _load_added(hidden_states, residual, row_base + cols + 4096)
    x9 = _load_added(hidden_states, residual, row_base + cols + 4608)
    x10 = _load_added(hidden_states, residual, row_base + cols + 5120)
    x11 = _load_added(hidden_states, residual, row_base + cols + 5632)
    x12 = _load_added(hidden_states, residual, row_base + cols + 6144)
    x13 = _load_added(hidden_states, residual, row_base + cols + 6656)

    squares = x0 * x0
    squares += x1 * x1
    squares += x2 * x2
    squares += x3 * x3
    squares += x4 * x4
    squares += x5 * x5
    squares += x6 * x6
    squares += x7 * x7
    squares += x8 * x8
    squares += x9 * x9
    squares += x10 * x10
    squares += x11 * x11
    squares += x12 * x12
    squares += x13 * x13
    inv_rms = tl.rsqrt(tl.sum(squares, axis=0) * (1.0 / 7168.0) + 1.0e-6)

    w0 = tl.load(weight + cols).to(tl.float32)
    tl.store(output + row_base + cols, (x0 * inv_rms) * w0)
    w1 = tl.load(weight + cols + 512).to(tl.float32)
    tl.store(output + row_base + cols + 512, (x1 * inv_rms) * w1)
    w2 = tl.load(weight + cols + 1024).to(tl.float32)
    tl.store(output + row_base + cols + 1024, (x2 * inv_rms) * w2)
    w3 = tl.load(weight + cols + 1536).to(tl.float32)
    tl.store(output + row_base + cols + 1536, (x3 * inv_rms) * w3)
    w4 = tl.load(weight + cols + 2048).to(tl.float32)
    tl.store(output + row_base + cols + 2048, (x4 * inv_rms) * w4)
    w5 = tl.load(weight + cols + 2560).to(tl.float32)
    tl.store(output + row_base + cols + 2560, (x5 * inv_rms) * w5)
    w6 = tl.load(weight + cols + 3072).to(tl.float32)
    tl.store(output + row_base + cols + 3072, (x6 * inv_rms) * w6)
    w7 = tl.load(weight + cols + 3584).to(tl.float32)
    tl.store(output + row_base + cols + 3584, (x7 * inv_rms) * w7)
    w8 = tl.load(weight + cols + 4096).to(tl.float32)
    tl.store(output + row_base + cols + 4096, (x8 * inv_rms) * w8)
    w9 = tl.load(weight + cols + 4608).to(tl.float32)
    tl.store(output + row_base + cols + 4608, (x9 * inv_rms) * w9)
    w10 = tl.load(weight + cols + 5120).to(tl.float32)
    tl.store(output + row_base + cols + 5120, (x10 * inv_rms) * w10)
    w11 = tl.load(weight + cols + 5632).to(tl.float32)
    tl.store(output + row_base + cols + 5632, (x11 * inv_rms) * w11)
    w12 = tl.load(weight + cols + 6144).to(tl.float32)
    tl.store(output + row_base + cols + 6144, (x12 * inv_rms) * w12)
    w13 = tl.load(weight + cols + 6656).to(tl.float32)
    tl.store(output + row_base + cols + 6656, (x13 * inv_rms) * w13)


@triton.jit
def _fused_add_rmsnorm_fp16x_kernel(
    hidden_states,
    residual,
    weight,
    output,
    BLOCK_SIZE: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK_SIZE)
    mask = cols < 7168
    offsets = row * 7168 + cols
    hidden = tl.load(hidden_states + offsets, mask=mask, other=0.0).to(tl.float32)
    skip = tl.load(residual + offsets, mask=mask, other=0.0).to(tl.float32)
    x_half = (hidden + skip).to(tl.float16)
    x = x_half.to(tl.float32)
    mean_square = tl.sum(x * x, axis=0) * (1.0 / 7168.0)
    inv_rms = tl.rsqrt(mean_square + 1.0e-6)
    scale = tl.load(weight + cols, mask=mask, other=0.0).to(tl.float32)
    tl.store(output + offsets, (x * inv_rms) * scale, mask=mask)


@triton.jit
def _fused_add_rmsnorm_cache_kernel(
    hidden_states,
    residual,
    weight,
    output,
    BLOCK_SIZE: tl.constexpr,
    FP16_X: tl.constexpr,
    LOAD_MODE: tl.constexpr,
    WEIGHT_MODE: tl.constexpr,
    STORE_MODE: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK_SIZE)
    mask = cols < 7168
    offsets = row * 7168 + cols
    if LOAD_MODE == 0:
        hidden = tl.load(hidden_states + offsets, mask=mask, other=0.0,
                         cache_modifier=".cg", eviction_policy="evict_first")
        skip = tl.load(residual + offsets, mask=mask, other=0.0,
                       cache_modifier=".cg", eviction_policy="evict_first")
    elif LOAD_MODE == 1:
        hidden = tl.load(hidden_states + offsets, mask=mask, other=0.0,
                         cache_modifier=".cg")
        skip = tl.load(residual + offsets, mask=mask, other=0.0,
                       cache_modifier=".cg")
    elif LOAD_MODE == 2:
        hidden = tl.load(hidden_states + offsets, mask=mask, other=0.0,
                         cache_modifier=".ca", eviction_policy="evict_first")
        skip = tl.load(residual + offsets, mask=mask, other=0.0,
                       cache_modifier=".ca", eviction_policy="evict_first")
    elif LOAD_MODE == 3:
        hidden = tl.load(hidden_states + offsets, mask=mask, other=0.0,
                         cache_modifier=".ca")
        skip = tl.load(residual + offsets, mask=mask, other=0.0,
                       cache_modifier=".ca")
    elif LOAD_MODE == 4:
        hidden = tl.load(hidden_states + offsets, mask=mask, other=0.0,
                         cache_modifier=".cv", eviction_policy="evict_first")
        skip = tl.load(residual + offsets, mask=mask, other=0.0,
                       cache_modifier=".cv", eviction_policy="evict_first")
    elif LOAD_MODE == 5:
        hidden = tl.load(hidden_states + offsets, mask=mask, other=0.0,
                         eviction_policy="evict_first")
        skip = tl.load(residual + offsets, mask=mask, other=0.0,
                       eviction_policy="evict_first")
    else:
        hidden = tl.load(hidden_states + offsets, mask=mask, other=0.0)
        skip = tl.load(residual + offsets, mask=mask, other=0.0)
    hidden = hidden.to(tl.float32)
    skip = skip.to(tl.float32)
    x = hidden + skip
    if FP16_X:
        x = x.to(tl.float16).to(tl.float32)
    mean_square = tl.sum(x * x, axis=0) * (1.0 / 7168.0)
    inv_rms = tl.rsqrt(mean_square + 1.0e-6)
    if WEIGHT_MODE == 0:
        scale = tl.load(weight + cols, mask=mask, other=0.0,
                        eviction_policy="evict_last")
    elif WEIGHT_MODE == 1:
        scale = tl.load(weight + cols, mask=mask, other=0.0)
    elif WEIGHT_MODE == 2:
        scale = tl.load(weight + cols, mask=mask, other=0.0,
                        cache_modifier=".ca", eviction_policy="evict_last")
    elif WEIGHT_MODE == 3:
        scale = tl.load(weight + cols, mask=mask, other=0.0,
                        cache_modifier=".ca")
    else:
        scale = tl.load(weight + cols, mask=mask, other=0.0,
                        cache_modifier=".cg", eviction_policy="evict_last")
    scale = scale.to(tl.float32)
    result = (x * inv_rms) * scale
    if STORE_MODE == 1:
        tl.store(
            output + offsets,
            result,
            mask=mask,
            cache_modifier=".cg",
            eviction_policy="evict_first",
        )
    elif STORE_MODE == 2:
        tl.store(
            output + offsets,
            result,
            mask=mask,
            cache_modifier=".cs",
            eviction_policy="evict_first",
        )
    elif STORE_MODE == 3:
        tl.store(
            output + offsets,
            result,
            mask=mask,
            cache_modifier=".wt",
            eviction_policy="evict_first",
        )
    elif STORE_MODE == 4:
        tl.store(
            output + offsets,
            result,
            mask=mask,
            cache_modifier=".wb",
            eviction_policy="evict_first",
        )
    elif STORE_MODE == 5:
        tl.store(output + offsets, result, mask=mask, cache_modifier=".cs")
    elif STORE_MODE == 6:
        tl.store(output + offsets, result, mask=mask, cache_modifier=".wt")
    else:
        tl.store(output + offsets, result, mask=mask)


@triton.jit
def _one_streaming_row(hidden_states, residual, weight, output, row, row_valid):
    cols = tl.arange(0, 8192)
    mask = (cols < 7168) & row_valid
    offsets = row * 7168 + cols
    hidden = tl.load(hidden_states + offsets, mask=mask, other=0.0,
                     cache_modifier=".cg", eviction_policy="evict_first").to(tl.float32)
    skip = tl.load(residual + offsets, mask=mask, other=0.0,
                   cache_modifier=".cg", eviction_policy="evict_first").to(tl.float32)
    x = hidden + skip
    mean_square = tl.sum(x * x, axis=0) * (1.0 / 7168.0)
    inv_rms = tl.rsqrt(mean_square + 1.0e-6)
    scale = tl.load(weight + cols, mask=mask, other=0.0,
                    eviction_policy="evict_last").to(tl.float32)
    tl.store(output + offsets, (x * inv_rms) * scale, mask=mask,
             cache_modifier=".cs", eviction_policy="evict_first")


@triton.jit
def _grouped_streaming_kernel(
    hidden_states,
    residual,
    weight,
    output,
    rows,
    ROWS_PER_PROGRAM: tl.constexpr,
):
    first_row = tl.program_id(0) * ROWS_PER_PROGRAM
    for i in tl.static_range(ROWS_PER_PROGRAM):
        row = first_row + i
        _one_streaming_row(hidden_states, residual, weight, output, row, row < rows)


@triton.jit
def _exact1024_streaming_kernel(hidden_states, residual, weight, output):
    row_base = tl.program_id(0) * 7168
    cols = tl.arange(0, 1024)
    x0 = _load_added(hidden_states, residual, row_base + cols)
    x1 = _load_added(hidden_states, residual, row_base + cols + 1024)
    x2 = _load_added(hidden_states, residual, row_base + cols + 2048)
    x3 = _load_added(hidden_states, residual, row_base + cols + 3072)
    x4 = _load_added(hidden_states, residual, row_base + cols + 4096)
    x5 = _load_added(hidden_states, residual, row_base + cols + 5120)
    x6 = _load_added(hidden_states, residual, row_base + cols + 6144)
    squares = x0 * x0 + x1 * x1 + x2 * x2 + x3 * x3
    squares += x4 * x4 + x5 * x5 + x6 * x6
    inv_rms = tl.rsqrt(tl.sum(squares, axis=0) * (1.0 / 7168.0) + 1.0e-6)
    w0 = tl.load(weight + cols, eviction_policy="evict_last").to(tl.float32)
    w1 = tl.load(weight + cols + 1024, eviction_policy="evict_last").to(tl.float32)
    w2 = tl.load(weight + cols + 2048, eviction_policy="evict_last").to(tl.float32)
    w3 = tl.load(weight + cols + 3072, eviction_policy="evict_last").to(tl.float32)
    w4 = tl.load(weight + cols + 4096, eviction_policy="evict_last").to(tl.float32)
    w5 = tl.load(weight + cols + 5120, eviction_policy="evict_last").to(tl.float32)
    w6 = tl.load(weight + cols + 6144, eviction_policy="evict_last").to(tl.float32)
    tl.store(output + row_base + cols, (x0 * inv_rms) * w0,
             cache_modifier=".cs", eviction_policy="evict_first")
    tl.store(output + row_base + cols + 1024, (x1 * inv_rms) * w1,
             cache_modifier=".cs", eviction_policy="evict_first")
    tl.store(output + row_base + cols + 2048, (x2 * inv_rms) * w2,
             cache_modifier=".cs", eviction_policy="evict_first")
    tl.store(output + row_base + cols + 3072, (x3 * inv_rms) * w3,
             cache_modifier=".cs", eviction_policy="evict_first")
    tl.store(output + row_base + cols + 4096, (x4 * inv_rms) * w4,
             cache_modifier=".cs", eviction_policy="evict_first")
    tl.store(output + row_base + cols + 5120, (x5 * inv_rms) * w5,
             cache_modifier=".cs", eviction_policy="evict_first")
    tl.store(output + row_base + cols + 6144, (x6 * inv_rms) * w6,
             cache_modifier=".cs", eviction_policy="evict_first")


def run(hidden_states, residual, weight):
    output = torch.empty_like(hidden_states)
    rows = hidden_states.shape[0]
    store_mode = 2 if rows > 1024 else 3
    _fused_add_rmsnorm_cache_kernel[(rows,)](
        hidden_states,
        residual,
        weight,
        output,
        BLOCK_SIZE=8192,
        FP16_X=False,
        LOAD_MODE=0,
        WEIGHT_MODE=0,
        STORE_MODE=store_mode,
        num_warps=8,
    )
    return output

import torch
import triton
import triton.language as tl


HIDDEN_SIZE = 4096


@triton.jit
def _post_norm_residual_kernel(
    x_ptr,
    residual_ptr,
    weight_ptr,
    output_ptr,
    eps,
    BLOCK_SIZE: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK_SIZE)
    offsets = row * BLOCK_SIZE + cols

    x = tl.load(x_ptr + offsets).to(tl.float32)
    variance = tl.sum(x * x, axis=0) * (1.0 / BLOCK_SIZE)
    inv_rms = tl.rsqrt(variance + eps)

    weight = tl.load(weight_ptr + cols).to(tl.float32)
    normalized = (x * inv_rms * weight).to(tl.bfloat16)
    residual = tl.load(residual_ptr + offsets)
    output = residual + normalized
    tl.store(output_ptr + offsets, output)


@triton.jit
def _post_norm_residual_dot_kernel(
    x_ptr,
    residual_ptr,
    weight_ptr,
    output_ptr,
    eps,
    BLOCK_SIZE: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK_SIZE)
    offsets = row * BLOCK_SIZE + cols

    x_bf16 = tl.load(x_ptr + offsets)
    x_matrix = tl.reshape(x_bf16, (16, 256))
    gram = tl.dot(x_matrix, tl.trans(x_matrix), out_dtype=tl.float32)
    diag_idx = tl.arange(0, 16)
    diagonal = tl.where(
        diag_idx[:, None] == diag_idx[None, :], gram, 0.0
    )
    variance = tl.sum(tl.sum(diagonal, axis=1), axis=0) * (1.0 / BLOCK_SIZE)
    inv_rms = tl.rsqrt(variance + eps)

    x = x_bf16.to(tl.float32)
    weight = tl.load(weight_ptr + cols).to(tl.float32)
    normalized = (x * inv_rms * weight).to(tl.bfloat16)
    residual = tl.load(residual_ptr + offsets)
    tl.store(output_ptr + offsets, residual + normalized)


@triton.jit
def _post_norm_residual_stream_kernel(
    x_ptr,
    residual_ptr,
    weight_ptr,
    output_ptr,
    eps,
    BLOCK_SIZE: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK_SIZE)
    offsets = row * BLOCK_SIZE + cols

    x = tl.load(
        x_ptr + offsets, cache_modifier=".cg", eviction_policy="evict_first"
    ).to(tl.float32)
    variance = tl.sum(x * x, axis=0) * (1.0 / BLOCK_SIZE)
    inv_rms = tl.rsqrt(variance + eps)

    weight = tl.load(
        weight_ptr + cols, cache_modifier=".ca", eviction_policy="evict_last"
    ).to(tl.float32)
    normalized = (x * inv_rms * weight).to(tl.bfloat16)
    residual = tl.load(
        residual_ptr + offsets,
        cache_modifier=".cg",
        eviction_policy="evict_first",
    )
    tl.store(output_ptr + offsets, residual + normalized)


@triton.jit
def _post_norm_residual_all_stream_kernel(
    x_ptr,
    residual_ptr,
    weight_ptr,
    output_ptr,
    eps,
    BLOCK_SIZE: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK_SIZE)
    offsets = row * BLOCK_SIZE + cols

    x = tl.load(
        x_ptr + offsets, cache_modifier=".cg", eviction_policy="evict_first"
    ).to(tl.float32)
    variance = tl.sum(x * x, axis=0) * (1.0 / BLOCK_SIZE)
    inv_rms = tl.rsqrt(variance + eps)

    weight = tl.load(
        weight_ptr + cols, cache_modifier=".ca", eviction_policy="evict_last"
    ).to(tl.float32)
    normalized = (x * inv_rms * weight).to(tl.bfloat16)
    residual = tl.load(
        residual_ptr + offsets,
        cache_modifier=".cg",
        eviction_policy="evict_first",
    )
    tl.store(
        output_ptr + offsets,
        residual + normalized,
        cache_modifier=".cs",
        eviction_policy="evict_first",
    )


@triton.jit
def _post_norm_residual_prefetch_kernel(
    x_ptr,
    residual_ptr,
    weight_ptr,
    output_ptr,
    eps,
    BLOCK_SIZE: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK_SIZE)
    offsets = row * BLOCK_SIZE + cols

    x = tl.load(
        x_ptr + offsets, cache_modifier=".cg", eviction_policy="evict_first"
    ).to(tl.float32)
    residual = tl.load(
        residual_ptr + offsets,
        cache_modifier=".cg",
        eviction_policy="evict_first",
    )
    weight = tl.load(
        weight_ptr + cols, cache_modifier=".ca", eviction_policy="evict_last"
    ).to(tl.float32)

    variance = tl.sum(x * x, axis=0) * (1.0 / BLOCK_SIZE)
    inv_rms = tl.rsqrt(variance + eps)
    normalized = (x * inv_rms * weight).to(tl.bfloat16)
    tl.store(
        output_ptr + offsets,
        residual + normalized,
        cache_modifier=".cs",
        eviction_policy="evict_first",
    )


@triton.jit
def _packed_bf16_add(a, b):
    return tl.inline_asm_elementwise(
        "v_pk_add_bf16 $0, $1, $2",
        "=v,v,v",
        [a, b],
        dtype=tl.bfloat16,
        is_pure=True,
        pack=2,
    )


@triton.jit
def _post_norm_residual_packed_kernel(
    x_ptr,
    residual_ptr,
    weight_ptr,
    output_ptr,
    eps,
    BLOCK_SIZE: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK_SIZE)
    offsets = row * BLOCK_SIZE + cols

    x = tl.load(
        x_ptr + offsets, cache_modifier=".cg", eviction_policy="evict_first"
    ).to(tl.float32)
    residual = tl.load(
        residual_ptr + offsets,
        cache_modifier=".cg",
        eviction_policy="evict_first",
    )
    weight = tl.load(
        weight_ptr + cols, cache_modifier=".ca", eviction_policy="evict_last"
    ).to(tl.float32)

    variance = tl.sum(x * x, axis=0) * (1.0 / BLOCK_SIZE)
    inv_rms = tl.rsqrt(variance + eps)
    normalized = (x * inv_rms * weight).to(tl.bfloat16)
    output = _packed_bf16_add(residual, normalized)
    tl.store(
        output_ptr + offsets,
        output,
        cache_modifier=".cs",
        eviction_policy="evict_first",
    )


@triton.jit
def _post_norm_residual_cache_test_kernel(
    x_ptr,
    residual_ptr,
    weight_ptr,
    output_ptr,
    eps,
    BLOCK_SIZE: tl.constexpr,
    CACHE_MODE: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK_SIZE)
    offsets = row * BLOCK_SIZE + cols

    if CACHE_MODE == 0:
        x_raw = tl.load(x_ptr + offsets)
        residual = tl.load(residual_ptr + offsets)
    elif CACHE_MODE == 1:
        x_raw = tl.load(x_ptr + offsets, cache_modifier=".cg")
        residual = tl.load(residual_ptr + offsets, cache_modifier=".cg")
    elif CACHE_MODE == 2:
        x_raw = tl.load(x_ptr + offsets, eviction_policy="evict_first")
        residual = tl.load(residual_ptr + offsets, eviction_policy="evict_first")
    else:
        x_raw = tl.load(
            x_ptr + offsets, cache_modifier=".cg", eviction_policy="evict_first"
        )
        residual = tl.load(
            residual_ptr + offsets,
            cache_modifier=".cg",
            eviction_policy="evict_first",
        )
    x = x_raw.to(tl.float32)
    weight = tl.load(
        weight_ptr + cols, cache_modifier=".ca", eviction_policy="evict_last"
    ).to(tl.float32)
    variance = tl.sum(x * x, axis=0) * (1.0 / BLOCK_SIZE)
    inv_rms = tl.rsqrt(variance + eps)
    normalized = (x * inv_rms * weight).to(tl.bfloat16)
    tl.store(
        output_ptr + offsets,
        residual + normalized,
        cache_modifier=".cs",
        eviction_policy="evict_first",
    )


@triton.jit
def _post_norm_residual_pair_kernel(
    x_ptr,
    residual_ptr,
    weight_ptr,
    output_ptr,
    eps,
    BLOCK_SIZE: tl.constexpr,
):
    pair = tl.program_id(0)
    row_in_pair = tl.arange(0, 2)[:, None]
    cols = tl.arange(0, BLOCK_SIZE)
    offsets = (pair * 2 + row_in_pair) * BLOCK_SIZE + cols[None, :]

    x = tl.load(
        x_ptr + offsets, cache_modifier=".cg", eviction_policy="evict_first"
    ).to(tl.float32)
    residual = tl.load(
        residual_ptr + offsets,
        cache_modifier=".cg",
        eviction_policy="evict_first",
    )
    weight = tl.load(
        weight_ptr + cols, cache_modifier=".ca", eviction_policy="evict_last"
    ).to(tl.float32)

    variance = tl.sum(x * x, axis=1) * (1.0 / BLOCK_SIZE)
    inv_rms = tl.rsqrt(variance + eps)
    normalized = (x * inv_rms[:, None] * weight[None, :]).to(tl.bfloat16)
    tl.store(
        output_ptr + offsets,
        residual + normalized,
        cache_modifier=".cs",
        eviction_policy="evict_first",
    )


@triton.jit
def _post_norm_residual_quad_kernel(
    x_ptr,
    residual_ptr,
    weight_ptr,
    output_ptr,
    eps,
    BLOCK_SIZE: tl.constexpr,
):
    group = tl.program_id(0)
    row_in_group = tl.arange(0, 4)[:, None]
    cols = tl.arange(0, BLOCK_SIZE)
    offsets = (group * 4 + row_in_group) * BLOCK_SIZE + cols[None, :]
    x = tl.load(x_ptr + offsets, cache_modifier=".cg").to(tl.float32)
    residual = tl.load(residual_ptr + offsets, cache_modifier=".cg")
    weight = tl.load(weight_ptr + cols).to(tl.float32)
    variance = tl.sum(x * x, axis=1) * (1.0 / BLOCK_SIZE)
    inv_rms = tl.rsqrt(variance + eps)
    normalized = (x * inv_rms[:, None] * weight[None, :]).to(tl.bfloat16)
    tl.store(
        output_ptr + offsets,
        residual + normalized,
        cache_modifier=".cs",
    )


def run(
    sublayer_output: torch.Tensor,
    residual: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
) -> torch.Tensor:
    output = torch.empty_like(sublayer_output)
    rows = sublayer_output.numel() // HIDDEN_SIZE
    if rows % 2 == 0 and rows < 12000:
        _post_norm_residual_pair_kernel[(rows // 2,)](
            sublayer_output,
            residual,
            weight,
            output,
            eps,
            BLOCK_SIZE=HIDDEN_SIZE,
            num_warps=8,
        )
    else:
        num_warps = 4 if 1500 <= rows < 6000 else 8
        _post_norm_residual_prefetch_kernel[(rows,)](
            sublayer_output,
            residual,
            weight,
            output,
            eps,
            BLOCK_SIZE=HIDDEN_SIZE,
            num_warps=num_warps,
        )
    return output

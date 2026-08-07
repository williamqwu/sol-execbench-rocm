import torch
import triton
import triton.language as tl


@triton.jit
def _kv_cache_rope_kernel(
    key_states,
    value_states,
    cos,
    sin,
    key_cache,
    value_cache,
    updated_key_cache,
    updated_value_cache,
    CURRENT_SEQ_LEN: tl.constexpr,
    NEW_SEQ_LEN: tl.constexpr,
    CACHE_TILES: tl.constexpr,
    CACHE_PROGRAMS: tl.constexpr,
    NEW_TILES: tl.constexpr,
    CACHE_BLOCK: tl.constexpr,
    NEW_BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    total_seq_len: tl.constexpr = CURRENT_SEQ_LEN + NEW_SEQ_LEN
    row_elements: tl.constexpr = total_seq_len * 128

    # Both classes share one launch, but cache copying uses much larger tiles
    # than RoPE.  This is especially important for decode, where the new region
    # is only 128 elements per head.
    if pid < CACHE_PROGRAMS:
        cache_bh = pid // CACHE_TILES
        cache_tile = pid - cache_bh * CACHE_TILES
        cache_offsets = (
            cache_tile * CACHE_BLOCK + tl.arange(0, CACHE_BLOCK)
        )
        cache_valid = cache_offsets < CURRENT_SEQ_LEN * 128
        cache_input_offset = (
            cache_bh * (CURRENT_SEQ_LEN * 128) + cache_offsets
        )
        cache_output_offset = cache_bh * row_elements + cache_offsets
        cached_key = tl.load(
            key_cache + cache_input_offset, mask=cache_valid
        )
        tl.store(
            updated_key_cache + cache_output_offset,
            cached_key,
            mask=cache_valid,
        )
        cached_value = tl.load(
            value_cache + cache_input_offset, mask=cache_valid
        )
        tl.store(
            updated_value_cache + cache_output_offset,
            cached_value,
            mask=cache_valid,
        )
    else:
        new_pid = pid - CACHE_PROGRAMS
        new_bh = new_pid // NEW_TILES
        new_tile = new_pid - new_bh * NEW_TILES
        new_offsets = new_tile * NEW_BLOCK + tl.arange(0, NEW_BLOCK)
        new_valid = new_offsets < NEW_SEQ_LEN * 128
        new_dim = new_offsets % 128
        new_input_offset = (
            new_bh * (NEW_SEQ_LEN * 128) + new_offsets
        )
        new_batch = new_bh // 10
        new_rope_offset = (
            new_batch * (NEW_SEQ_LEN * 128) + new_offsets
        )
        new_peer_offset = new_input_offset + tl.where(
            new_dim < 64, 64, -64
        )

        new_key = tl.load(
            key_states + new_input_offset, mask=new_valid, other=0.0
        )
        new_value = tl.load(
            value_states + new_input_offset, mask=new_valid, other=0.0
        )
        peer_key = tl.load(
            key_states + new_peer_offset, mask=new_valid, other=0.0
        )
        cosine = tl.load(
            cos + new_rope_offset, mask=new_valid, other=0.0
        )
        sine = tl.load(
            sin + new_rope_offset, mask=new_valid, other=0.0
        )

        rotated_half = tl.where(new_dim < 64, -peer_key, peer_key)
        product_1 = (
            new_key.to(tl.float32) * cosine.to(tl.float32)
        ).to(tl.bfloat16)
        product_2 = (
            rotated_half.to(tl.float32) * sine.to(tl.float32)
        ).to(tl.bfloat16)
        rotated_key = (
            product_1.to(tl.float32) + product_2.to(tl.float32)
        ).to(tl.bfloat16)

        new_output_offset = (
            new_bh * row_elements + CURRENT_SEQ_LEN * 128 + new_offsets
        )
        tl.store(
            updated_key_cache + new_output_offset,
            rotated_key,
            mask=new_valid,
        )
        tl.store(
            updated_value_cache + new_output_offset,
            new_value,
            mask=new_valid,
        )


@torch.no_grad()
def run(
    key_states: torch.Tensor,
    value_states: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
):
    batch_size, num_heads, new_seq_len, head_dim = key_states.shape
    current_seq_len = key_cache.shape[2]
    total_seq_len = current_seq_len + new_seq_len

    updated_key_cache = torch.empty(
        (batch_size, num_heads, total_seq_len, head_dim),
        device=key_states.device,
        dtype=key_states.dtype,
    )
    updated_value_cache = torch.empty_like(updated_key_cache)

    cache_block = 8192
    new_block = 128 if current_seq_len else 1024
    real_cache_tiles = triton.cdiv(current_seq_len * head_dim, cache_block)
    # A nonzero divisor keeps the compile-time-dead cache branch well-formed
    # for prefill workloads, where CACHE_PROGRAMS is zero.
    cache_tiles = max(1, real_cache_tiles)
    new_tiles = triton.cdiv(new_seq_len * head_dim, new_block)
    cache_programs = batch_size * num_heads * real_cache_tiles
    grid = (cache_programs + batch_size * num_heads * new_tiles,)
    _kv_cache_rope_kernel[grid](
        key_states,
        value_states,
        cos,
        sin,
        key_cache,
        value_cache,
        updated_key_cache,
        updated_value_cache,
        CURRENT_SEQ_LEN=current_seq_len,
        NEW_SEQ_LEN=new_seq_len,
        CACHE_TILES=cache_tiles,
        CACHE_PROGRAMS=cache_programs,
        NEW_TILES=new_tiles,
        CACHE_BLOCK=cache_block,
        NEW_BLOCK=new_block,
        num_warps=8,
    )
    return updated_key_cache, updated_value_cache

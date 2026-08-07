import torch
import triton
import triton.language as tl


@triton.jit
def _rope_kernel(
    query,
    key,
    cos,
    sin,
    query_out,
    key_out,
    NUM_QUERY_HEADS: tl.constexpr,
    NUM_KEY_HEADS: tl.constexpr,
    NUM_QUERY_GROUPS: tl.constexpr,
    GROUP_SIZE: tl.constexpr,
    SEQ_LEN: tl.constexpr,
):
    batch_pos = tl.program_id(0)
    head_group = tl.program_id(1)
    d = tl.arange(0, 64)
    batch = batch_pos // SEQ_LEN
    pos = batch_pos % SEQ_LEN
    rope_base = pos * 128
    c0 = tl.load(cos + rope_base + d)
    c1 = tl.load(cos + rope_base + 64 + d)
    s0 = tl.load(sin + rope_base + d)
    s1 = tl.load(sin + rope_base + 64 + d)

    if head_group < NUM_QUERY_GROUPS:
        first_head = head_group * GROUP_SIZE
        for i in tl.static_range(0, GROUP_SIZE):
            head = first_head + i
            valid = head < NUM_QUERY_HEADS
            row = (batch * NUM_QUERY_HEADS + head) * SEQ_LEN + pos
            x_base = query + row * 128
            out_base = query_out + row * 128
            x0 = tl.load(x_base + d, mask=valid)
            x1 = tl.load(x_base + 64 + d, mask=valid)
            y0 = x0 * c0 + (-x1) * s0
            y1 = x1 * c1 + x0 * s1
            tl.store(out_base + d, y0, mask=valid)
            tl.store(out_base + 64 + d, y1, mask=valid)
    else:
        first_key_head = (head_group - NUM_QUERY_GROUPS) * GROUP_SIZE
        for i in tl.static_range(0, GROUP_SIZE):
            key_head = first_key_head + i
            valid = key_head < NUM_KEY_HEADS
            row = (batch * NUM_KEY_HEADS + key_head) * SEQ_LEN + pos
            x_base = key + row * 128
            out_base = key_out + row * 128
            x0 = tl.load(x_base + d, mask=valid)
            x1 = tl.load(x_base + 64 + d, mask=valid)
            y0 = x0 * c0 + (-x1) * s0
            y1 = x1 * c1 + x0 * s1
            tl.store(out_base + d, y0, mask=valid)
            tl.store(out_base + 64 + d, y1, mask=valid)


@triton.jit
def _rope_flat_kernel(
    query,
    key,
    cos,
    sin,
    query_out,
    key_out,
    NUM_QUERY_GROUPS: tl.constexpr,
    SEQ_LEN: tl.constexpr,
    INPUT_CACHE: tl.constexpr,
    STORE_CACHE: tl.constexpr,
    INPUT_EVICTION: tl.constexpr,
    ROPE_EVICTION: tl.constexpr,
):
    pos = tl.program_id(0)
    row_group = tl.program_id(1)
    d = tl.arange(0, 64)

    rope_base = pos * 128
    c0 = tl.load(cos + rope_base + d, eviction_policy=ROPE_EVICTION)
    c1 = tl.load(cos + rope_base + 64 + d, eviction_policy=ROPE_EVICTION)
    s0 = tl.load(sin + rope_base + d, eviction_policy=ROPE_EVICTION)
    s1 = tl.load(sin + rope_base + 64 + d, eviction_policy=ROPE_EVICTION)

    if row_group < NUM_QUERY_GROUPS:
        first_row = row_group * 4
        for i in tl.static_range(0, 4):
            row = first_row + i
            offset = (row * SEQ_LEN + pos) * 128
            x0 = tl.load(query + offset + d, cache_modifier=INPUT_CACHE, eviction_policy=INPUT_EVICTION)
            x1 = tl.load(query + offset + 64 + d, cache_modifier=INPUT_CACHE, eviction_policy=INPUT_EVICTION)
            y0 = x0 * c0 + (-x1) * s0
            y1 = x1 * c1 + x0 * s1
            tl.store(query_out + offset + d, y0, cache_modifier=STORE_CACHE)
            tl.store(query_out + offset + 64 + d, y1, cache_modifier=STORE_CACHE)
    else:
        first_row = (row_group - NUM_QUERY_GROUPS) * 4
        for i in tl.static_range(0, 4):
            row = first_row + i
            offset = (row * SEQ_LEN + pos) * 128
            x0 = tl.load(key + offset + d, cache_modifier=INPUT_CACHE, eviction_policy=INPUT_EVICTION)
            x1 = tl.load(key + offset + 64 + d, cache_modifier=INPUT_CACHE, eviction_policy=INPUT_EVICTION)
            y0 = x0 * c0 + (-x1) * s0
            y1 = x1 * c1 + x0 * s1
            tl.store(key_out + offset + d, y0, cache_modifier=STORE_CACHE)
            tl.store(key_out + offset + 64 + d, y1, cache_modifier=STORE_CACHE)
def run(
    query: torch.Tensor,
    key: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
):
    query_out = torch.empty_like(query)
    key_out = torch.empty_like(key)
    batch_size = query.shape[0]
    num_query_heads = query.shape[1]
    num_key_heads = key.shape[1]
    seq_len = query.shape[2]
    num_query_groups = batch_size * num_query_heads // 4
    num_key_groups = batch_size * num_key_heads // 4
    head_groups = num_query_groups + num_key_groups
    store_cache = ".cs" if seq_len >= 8192 else ".wt"
    _rope_flat_kernel[(seq_len, head_groups)](
        query,
        key,
        cos,
        sin,
        query_out,
        key_out,
        NUM_QUERY_GROUPS=num_query_groups,
        SEQ_LEN=seq_len,
        INPUT_CACHE=".cg",
        STORE_CACHE=store_cache,
        INPUT_EVICTION="",
        ROPE_EVICTION="evict_last",
        num_warps=1,
    )
    return query_out, key_out

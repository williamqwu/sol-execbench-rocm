import torch
import triton
import triton.language as tl


@triton.jit
def _copy_caches(
    key_in, value_in, key_out, value_out,
    n_elements: tl.constexpr,
    BLOCK: tl.constexpr,
):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < n_elements
    tl.store(key_out + offsets, tl.load(key_in + offsets, mask=mask), mask=mask)
    tl.store(value_out + offsets, tl.load(value_in + offsets, mask=mask), mask=mask)


@triton.jit
def _copy_caches_streaming(
    key_in, value_in, key_out, value_out,
    n_elements: tl.constexpr,
    BLOCK: tl.constexpr,
):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < n_elements
    key = tl.load(key_in + offsets, mask=mask, cache_modifier=".cg")
    value = tl.load(value_in + offsets, mask=mask, cache_modifier=".cg")
    tl.store(key_out + offsets, key, mask=mask, cache_modifier=".cs")
    tl.store(value_out + offsets, value, mask=mask, cache_modifier=".cs")


@triton.jit
def _mark_updated(cache_position, marker, new_seq_len: tl.constexpr, max_seq_len: tl.constexpr):
    # This kernel deliberately has one CTA: after clearing the compact marker,
    # a CTA barrier makes arbitrary (and duplicate) cache positions safe to
    # scatter without requiring a second initialization launch.
    offsets = tl.arange(0, max_seq_len)
    tl.store(marker + offsets, 0.0)
    tl.debug_barrier()
    positions = tl.load(cache_position + offsets, mask=offsets < new_seq_len, other=0)
    tl.store(marker + positions, 1.0, mask=offsets < new_seq_len)


@triton.jit
def _copy_caches_selective(
    key_in, value_in, key_out, value_out, marker,
    max_seq_len: tl.constexpr,
    ROWS: tl.constexpr,
):
    # Several complete 128-wide cache rows per single-wave program.  Marker
    # values are loaded once per row rather than once per cache element.
    row = tl.arange(0, ROWS)[:, None]
    dim = tl.arange(0, 128)[None, :]
    linear_row = tl.program_id(0) * ROWS + row
    offsets = linear_row * 128 + dim
    updated = tl.load(marker + (linear_row % max_seq_len)) != 0.0
    key = tl.load(key_in + offsets, mask=~updated, other=0.0, cache_modifier=".cg")
    value = tl.load(value_in + offsets, mask=~updated, other=0.0, cache_modifier=".cg")
    tl.store(key_out + offsets, key, cache_modifier=".cs")
    tl.store(value_out + offsets, value, cache_modifier=".cs")


@triton.jit
def _rope_backward(
    grad_key_cache, grad_value_cache, key_states, cos, sin, cache_position,
    grad_key_states, grad_value_states, grad_cos, grad_sin,
    grad_key_cache_input, grad_value_cache_input,
    new_seq_len: tl.constexpr,
    max_seq_len: tl.constexpr,
    HALF: tl.constexpr,
    STREAMING: tl.constexpr,
    ZERO_CACHE: tl.constexpr,
):
    # One program handles all eight KV heads and both 64-element halves for
    # one (batch, sequence) row.  Keeping heads in the reduction dimension
    # also makes the two embedding gradients single-writer outputs.
    row = tl.program_id(0)
    batch = row // new_seq_len
    seq = row - batch * new_seq_len
    pos = tl.load(cache_position + seq)

    head = tl.arange(0, 8)[:, None]
    dim = tl.arange(0, HALF)[None, :]

    cache_base = batch * 8 * max_seq_len * 128 + head * max_seq_len * 128 + pos * 128
    state_base = batch * 8 * new_seq_len * 128 + head * new_seq_len * 128 + seq * 128
    emb_base = batch * new_seq_len * 128 + seq * 128

    if STREAMING:
        gk1 = tl.load(grad_key_cache + cache_base + dim, cache_modifier=".cg")
        gk2 = tl.load(grad_key_cache + cache_base + HALF + dim, cache_modifier=".cg")
        gv1 = tl.load(grad_value_cache + cache_base + dim, cache_modifier=".cg")
        gv2 = tl.load(grad_value_cache + cache_base + HALF + dim, cache_modifier=".cg")
        k1 = tl.load(key_states + state_base + dim, cache_modifier=".cg")
        k2 = tl.load(key_states + state_base + HALF + dim, cache_modifier=".cg")
        c1 = tl.load(cos + emb_base + dim, cache_modifier=".cg")
        c2 = tl.load(cos + emb_base + HALF + dim, cache_modifier=".cg")
        s1 = tl.load(sin + emb_base + dim, cache_modifier=".cg")
        s2 = tl.load(sin + emb_base + HALF + dim, cache_modifier=".cg")
    else:
        gk1 = tl.load(grad_key_cache + cache_base + dim)
        gk2 = tl.load(grad_key_cache + cache_base + HALF + dim)
        gv1 = tl.load(grad_value_cache + cache_base + dim)
        gv2 = tl.load(grad_value_cache + cache_base + HALF + dim)
        k1 = tl.load(key_states + state_base + dim)
        k2 = tl.load(key_states + state_base + HALF + dim)
        c1 = tl.load(cos + emb_base + dim)
        c2 = tl.load(cos + emb_base + HALF + dim)
        s1 = tl.load(sin + emb_base + dim)
        s2 = tl.load(sin + emb_base + HALF + dim)

    # PyTorch produces BF16 intermediates for both multiplies, then rounds
    # once more after the add/subtract.  Explicit casts preserve that order.
    cos_term1 = (gk1 * c1).to(tl.bfloat16)
    cos_term2 = (gk2 * c2).to(tl.bfloat16)
    sin_term1 = (gk1 * s1).to(tl.bfloat16)
    sin_term2 = (gk2 * s2).to(tl.bfloat16)
    out_k1 = (cos_term1.to(tl.float32) + sin_term2.to(tl.float32)).to(tl.bfloat16)
    out_k2 = (cos_term2.to(tl.float32) - sin_term1.to(tl.float32)).to(tl.bfloat16)

    if STREAMING:
        tl.store(grad_key_states + state_base + dim, out_k1, cache_modifier=".cs")
        tl.store(grad_key_states + state_base + HALF + dim, out_k2, cache_modifier=".cs")
        tl.store(grad_value_states + state_base + dim, gv1, cache_modifier=".cs")
        tl.store(grad_value_states + state_base + HALF + dim, gv2, cache_modifier=".cs")
    else:
        tl.store(grad_key_states + state_base + dim, out_k1)
        tl.store(grad_key_states + state_base + HALF + dim, out_k2)
        tl.store(grad_value_states + state_base + dim, gv1)
        tl.store(grad_value_states + state_base + HALF + dim, gv2)

    # Multiplication materializes a BF16 tensor in the reference before the
    # float-accumulating head reduction.
    cos_prod1 = (gk1 * k1).to(tl.bfloat16).to(tl.float32)
    cos_prod2 = (gk2 * k2).to(tl.bfloat16).to(tl.float32)
    sin_prod1 = (gk1 * (-k2)).to(tl.bfloat16).to(tl.float32)
    sin_prod2 = (gk2 * k1).to(tl.bfloat16).to(tl.float32)
    sum_cos1 = tl.sum(cos_prod1, axis=0).to(tl.bfloat16)
    sum_cos2 = tl.sum(cos_prod2, axis=0).to(tl.bfloat16)
    sum_sin1 = tl.sum(sin_prod1, axis=0).to(tl.bfloat16)
    sum_sin2 = tl.sum(sin_prod2, axis=0).to(tl.bfloat16)
    if STREAMING:
        tl.store(grad_cos + emb_base + dim, sum_cos1, cache_modifier=".cs")
        tl.store(grad_cos + emb_base + HALF + dim, sum_cos2, cache_modifier=".cs")
        tl.store(grad_sin + emb_base + dim, sum_sin1, cache_modifier=".cs")
        tl.store(grad_sin + emb_base + HALF + dim, sum_sin2, cache_modifier=".cs")
    else:
        tl.store(grad_cos + emb_base + dim, sum_cos1)
        tl.store(grad_cos + emb_base + HALF + dim, sum_cos2)
        tl.store(grad_sin + emb_base + dim, sum_sin1)
        tl.store(grad_sin + emb_base + HALF + dim, sum_sin2)

    # The copy kernel ran first, so indexed cache rows only need overwriting.
    if ZERO_CACHE:
        if STREAMING:
            tl.store(grad_key_cache_input + cache_base + dim, 0.0, cache_modifier=".cs")
            tl.store(grad_key_cache_input + cache_base + HALF + dim, 0.0, cache_modifier=".cs")
            tl.store(grad_value_cache_input + cache_base + dim, 0.0, cache_modifier=".cs")
            tl.store(grad_value_cache_input + cache_base + HALF + dim, 0.0, cache_modifier=".cs")
        else:
            tl.store(grad_key_cache_input + cache_base + dim, 0.0)
            tl.store(grad_key_cache_input + cache_base + HALF + dim, 0.0)
            tl.store(grad_value_cache_input + cache_base + dim, 0.0)
            tl.store(grad_value_cache_input + cache_base + HALF + dim, 0.0)


@triton.jit
def _selective_copy_and_rope(
    grad_key_cache, grad_value_cache, key_states, cos, sin, cache_position,
    grad_key_states, grad_value_states, grad_cos, grad_sin,
    grad_key_cache_input, grad_value_cache_input, marker,
    new_seq_len: tl.constexpr,
    max_seq_len: tl.constexpr,
    total_rows: tl.constexpr,
    COPY_ROWS: tl.constexpr,
):
    copy_row = tl.arange(0, COPY_ROWS)[:, None]
    copy_dim = tl.arange(0, 128)[None, :]
    linear_row = tl.program_id(0) * COPY_ROWS + copy_row
    offsets = linear_row * 128 + copy_dim
    updated = tl.load(marker + (linear_row % max_seq_len)) != 0.0
    key = tl.load(grad_key_cache + offsets, mask=~updated, other=0.0, cache_modifier=".cg")
    value = tl.load(grad_value_cache + offsets, mask=~updated, other=0.0, cache_modifier=".cg")
    tl.store(grad_key_cache_input + offsets, key, cache_modifier=".cs")
    tl.store(grad_value_cache_input + offsets, value, cache_modifier=".cs")

    if tl.program_id(0) < total_rows:
        _rope_backward(
            grad_key_cache, grad_value_cache, key_states, cos, sin, cache_position,
            grad_key_states, grad_value_states, grad_cos, grad_sin,
            grad_key_cache_input, grad_value_cache_input,
            new_seq_len, max_seq_len, 64, True, False,
        )


@torch.no_grad()
def run(
    grad_key_cache: torch.Tensor,
    grad_value_cache: torch.Tensor,
    key_states: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    cache_position: torch.Tensor,
):
    batch_size = grad_key_cache.shape[0]
    max_seq_len = grad_key_cache.shape[2]
    new_seq_len = key_states.shape[2]

    out_key_states = torch.empty_like(key_states)
    out_value_states = torch.empty_like(key_states)
    out_cos = torch.empty_like(cos)
    out_sin = torch.empty_like(sin)
    out_key_cache = torch.empty_like(grad_key_cache)
    out_value_cache = torch.empty_like(grad_value_cache)

    cache_elements = grad_key_cache.numel()
    state_elements = key_states.numel()
    rows = batch_size * new_seq_len
    selective_copy = state_elements >= 8 * 1024 * 1024
    # For sufficiently large updates, an exact position marker pays for itself
    # by avoiding cache rows that would immediately be overwritten with zero.
    # The copy and RoPE work then share one launch; the independent marker is
    # needed because grad_cos is written concurrently by that fused launch.
    if selective_copy:
        marker = torch.empty(max_seq_len, device=grad_key_cache.device, dtype=torch.bfloat16)
        _mark_updated[(1,)](
            cache_position, marker,
            new_seq_len=new_seq_len, max_seq_len=max_seq_len, num_warps=8,
        )
        copy_rows = 8 if state_elements >= 32 * 1024 * 1024 else 4
        _selective_copy_and_rope[(cache_elements // (copy_rows * 128),)](
            grad_key_cache, grad_value_cache, key_states, cos, sin, cache_position,
            out_key_states, out_value_states, out_cos, out_sin,
            out_key_cache, out_value_cache, marker,
            new_seq_len=new_seq_len, max_seq_len=max_seq_len,
            total_rows=rows, COPY_ROWS=copy_rows, num_warps=1,
        )
    elif cache_elements <= 8 * 1024 * 1024:
        copy_block = 512
        _copy_caches[(triton.cdiv(cache_elements, copy_block),)](
            grad_key_cache, grad_value_cache, out_key_cache, out_value_cache,
            cache_elements, BLOCK=copy_block, num_warps=1,
        )
    else:
        copy_block = 256
        _copy_caches_streaming[(triton.cdiv(cache_elements, copy_block),)](
            grad_key_cache, grad_value_cache, out_key_cache, out_value_cache,
            cache_elements, BLOCK=copy_block, num_warps=1,
        )
    if not selective_copy:
        if rows < 1024:
            rope_warps = 8
        elif rows < 4096:
            rope_warps = 2
        else:
            rope_warps = 1
        _rope_backward[(rows,)](
            grad_key_cache, grad_value_cache, key_states, cos, sin, cache_position,
            out_key_states, out_value_states, out_cos, out_sin,
            out_key_cache, out_value_cache,
            new_seq_len=new_seq_len, max_seq_len=max_seq_len, HALF=64,
            STREAMING=True, ZERO_CACHE=True, num_warps=rope_warps,
        )
    return (
        out_key_states,
        out_value_states,
        out_cos,
        out_sin,
        out_key_cache,
        out_value_cache,
    )

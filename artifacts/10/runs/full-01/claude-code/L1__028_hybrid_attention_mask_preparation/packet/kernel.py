import torch
import triton
import triton.language as tl


@triton.jit
def _hybrid_mask_flat(
    full_ptr,
    swa_ptr,
    total,          # int64: B*H*T*S
    T,              # int32
    S,              # int32
    TP,             # int32: T + past  (unused directly; kept for clarity)
    P,              # int32
    BLOCK: tl.constexpr,
    EXACT: tl.constexpr,
):
    """Flat, 16B-aligned pass over the output.

    Element k of the flat [B,H,T,S] buffer has s = k % S, row = k // S,
    t = row % T.  full[k] = (s > t + P);  swa[k] = 0.

    BLOCK <= S is guaranteed by the launcher, so a program spans at most two
    consecutive rows -> the per-lane row/column decode is a compare, not a
    division.
    """
    pid = tl.program_id(0).to(tl.int64)
    base = pid * BLOCK

    row0 = base // S                       # scalar int64 divide, once
    rem0 = (base - row0 * S).to(tl.int32)  # column of the first lane, < S
    t0 = (row0 % T).to(tl.int32)           # scalar int64 mod, once

    lane = tl.arange(0, BLOCK)
    pos = rem0 + lane                      # in [0, S + BLOCK) -> fits int32
    adv = pos >= S                         # crossed into the next row?
    s = tl.where(adv, pos - S, pos)

    t = t0 + adv.to(tl.int32)
    t = tl.where(t >= T, t - T, t)         # row wrap within the batch/head

    v = (s > (t + P)).to(tl.int8)

    off = base + lane
    if EXACT:
        tl.store(full_ptr + off, v)
        tl.store(swa_ptr + off, tl.zeros([BLOCK], dtype=tl.int8))
    else:
        m = off < total
        tl.store(full_ptr + off, v, mask=m)
        tl.store(swa_ptr + off, tl.zeros([BLOCK], dtype=tl.int8), mask=m)


def _prev_pow2(n: int) -> int:
    p = 1
    while p * 2 <= n:
        p *= 2
    return p


@torch.no_grad()
def run(
    batch_size_scalar: int,
    seq_length_scalar: int,
    past_key_values_length_scalar: int,
):
    num_attention_heads = 64
    swa_num_attention_heads = 64

    B = int(batch_size_scalar)
    T = int(seq_length_scalar)
    P = int(past_key_values_length_scalar)
    S = T + P

    shape_f = (B, num_attention_heads, T, S)
    shape_s = (B, swa_num_attention_heads, T, S)

    n_full = B * num_attention_heads * T * S
    n_swa = B * swa_num_attention_heads * T * S

    if n_full == 0 or n_swa == 0:
        return (
            torch.empty(shape_f, dtype=torch.bool, device="cuda"),
            torch.empty(shape_s, dtype=torch.bool, device="cuda"),
        )

    # One allocation, two views: halves the allocator work on the small shapes
    # where launch/alloc overhead is the entire runtime.
    buf = torch.empty(n_full + n_swa, dtype=torch.bool, device="cuda")
    full = buf[:n_full].view(shape_f)
    swa = buf[n_full:].view(shape_s)

    u8 = buf.view(torch.uint8)
    full_u8 = u8[:n_full]
    swa_u8 = u8[n_full:]

    # BLOCK <= S keeps the two-row decode valid; power of two keeps stores
    # 16B-aligned and dense.
    BLOCK = min(4096, _prev_pow2(S))
    exact = (n_full % BLOCK) == 0

    if BLOCK >= 2048:
        num_warps = 8
    elif BLOCK >= 512:
        num_warps = 4
    else:
        num_warps = 2

    grid = (triton.cdiv(n_full, BLOCK),)
    _hybrid_mask_flat[grid](
        full_u8,
        swa_u8,
        n_full,
        T,
        S,
        T + P,
        P,
        BLOCK=BLOCK,
        EXACT=exact,
        num_warps=num_warps,
    )
    return full, swa

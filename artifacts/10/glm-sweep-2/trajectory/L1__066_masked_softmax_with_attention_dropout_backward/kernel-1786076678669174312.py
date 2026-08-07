import torch
import triton
import triton.language as tl

@triton.jit
def _k_single(go_ptr, pa_ptr, mask_ptr, dm_ptr, out_ptr, scale,
              n_cols, n_heads,
              APPLY_DROPOUT: tl.constexpr, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    r = pid % n_cols
    bh = pid // n_cols
    b = bh // n_heads
    row = pid * n_cols
    mrow = (b * n_cols + r) * n_cols
    cols = tl.arange(0, BLOCK)
    valid = cols < n_cols
    g = tl.load(go_ptr + row + cols, mask=valid, other=0.0)
    p = tl.load(pa_ptr + row + cols, mask=valid, other=0.0)
    if APPLY_DROPOUT:
        d = tl.load(dm_ptr + row + cols, mask=valid, other=0).to(tl.float32)
        g = g * d * scale
    s = tl.sum(p * g)
    v = p * (g - s)
    m = tl.load(mask_ptr + mrow + cols, mask=valid, other=0).to(tl.int1)
    v = tl.where(m, v, 0.0)
    tl.store(out_ptr + row + cols, v, mask=valid)

@triton.jit
def _k_chunked(go_ptr, pa_ptr, mask_ptr, dm_ptr, out_ptr, scale,
               n_cols, n_heads,
               APPLY_DROPOUT: tl.constexpr, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    r = pid % n_cols
    bh = pid // n_cols
    b = bh // n_heads
    row = pid * n_cols
    mrow = (b * n_cols + r) * n_cols
    n_chunks = tl.cdiv(n_cols, BLOCK)
    s_acc = tl.zeros([BLOCK], dtype=tl.float32)
    for c in range(n_chunks):
        cols = c * BLOCK + tl.arange(0, BLOCK)
        valid = cols < n_cols
        g = tl.load(go_ptr + row + cols, mask=valid, other=0.0)
        p = tl.load(pa_ptr + row + cols, mask=valid, other=0.0)
        if APPLY_DROPOUT:
            d = tl.load(dm_ptr + row + cols, mask=valid, other=0).to(tl.float32)
            g = g * d * scale
        s_acc += p * g
    sum_term = tl.sum(s_acc)
    for c in range(n_chunks):
        cols = c * BLOCK + tl.arange(0, BLOCK)
        valid = cols < n_cols
        g = tl.load(go_ptr + row + cols, mask=valid, other=0.0)
        p = tl.load(pa_ptr + row + cols, mask=valid, other=0.0)
        if APPLY_DROPOUT:
            d = tl.load(dm_ptr + row + cols, mask=valid, other=0).to(tl.float32)
            g = g * d * scale
        v = p * (g - sum_term)
        m = tl.load(mask_ptr + mrow + cols, mask=valid, other=0).to(tl.int1)
        v = tl.where(m, v, 0.0)
        tl.store(out_ptr + row + cols, v, mask=valid)


def _np2(x):
    p = 1
    while p < x:
        p <<= 1
    return p

def _floor_pow2(x):
    p = 1
    while (p << 1) <= x:
        p <<= 1
    return p


@torch.no_grad()
def run(grad_output, p_attn, mask, dropout_mask, p_dropout):
    B, H, T, _ = grad_output.shape
    out = torch.empty_like(grad_output)
    ad = p_dropout > 0.0
    scale = (1.0 / (1.0 - p_dropout)) if ad else 1.0
    grid = (B * H * T,)

    is_pow2 = (T & (T - 1)) == 0
    use_chunked = (not is_pow2) and T < 512

    if use_chunked:
        BLOCK = _floor_pow2(T)
        if BLOCK < 128:
            BLOCK = 128
        nw = 2
        _k_chunked[grid](grad_output, p_attn, mask, dropout_mask, out, scale,
                         T, H, APPLY_DROPOUT=ad, BLOCK=BLOCK, num_warps=nw)
    else:
        BLOCK = _np2(T)
        if BLOCK <= 1024:
            nw = 2
        elif BLOCK <= 2048:
            nw = 8
        else:
            nw = 16
        _k_single[grid](grad_output, p_attn, mask, dropout_mask, out, scale,
                        T, H, APPLY_DROPOUT=ad, BLOCK=BLOCK, num_warps=nw)
    return out

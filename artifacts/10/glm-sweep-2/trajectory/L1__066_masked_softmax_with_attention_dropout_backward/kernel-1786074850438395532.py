import torch
import triton
import triton.language as tl

@triton.jit
def _k(go_ptr, pa_ptr, mask_ptr, dm_ptr, out_ptr, scale,
       n_cols, n_heads, APPLY_DROPOUT: tl.constexpr, BLOCK: tl.constexpr):
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

def _np2(x):
    p=1
    while p<x: p<<=1
    return p

@torch.no_grad()
def run(grad_output, p_attn, mask, dropout_mask, p_dropout):
    B,H,T,_ = grad_output.shape
    out=torch.empty_like(grad_output)
    BLOCK=_np2(T)
    ad = p_dropout>0.0
    scale = (1.0/(1.0-p_dropout)) if ad else 1.0
    # pick warps by block size
    if BLOCK <= 256: nw=4
    elif BLOCK <= 1024: nw=8
    else: nw=16
    grid=(B*H*T,)
    _k[grid](grad_output,p_attn,mask,dropout_mask,out,scale,T,H,
             APPLY_DROPOUT=ad, BLOCK=BLOCK, num_warps=nw)
    return out

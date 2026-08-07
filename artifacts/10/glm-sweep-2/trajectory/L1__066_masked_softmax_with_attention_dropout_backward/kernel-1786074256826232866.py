import torch
import triton
import triton.language as tl


@triton.jit
def _softmax_bw_kernel(
    grad_out_ptr, p_attn_ptr, mask_ptr, dropout_ptr, out_ptr,
    scale,
    n_cols, n_heads,
    APPLY_DROPOUT: tl.constexpr,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    row_start = pid * n_cols
    r = pid % n_cols
    bh = pid // n_cols
    b = bh // n_heads
    mask_row_start = (b * n_cols + r) * n_cols

    cols = tl.arange(0, BLOCK)
    valid = cols < n_cols

    g = tl.load(grad_out_ptr + row_start + cols, mask=valid, other=0.0)
    p = tl.load(p_attn_ptr + row_start + cols, mask=valid, other=0.0)
    if APPLY_DROPOUT:
        d = tl.load(dropout_ptr + row_start + cols, mask=valid, other=0).to(tl.float32)
        g = g * d * scale

    sum_term = tl.sum(p * g)
    val = p * (g - sum_term)

    m = tl.load(mask_ptr + mask_row_start + cols, mask=valid, other=0).to(tl.int1)
    val = tl.where(m, val, 0.0)
    tl.store(out_ptr + row_start + cols, val, mask=valid)


def _next_pow2(x):
    p = 1
    while p < x:
        p <<= 1
    return p


@torch.no_grad()
def run(
    grad_output: torch.Tensor,
    p_attn: torch.Tensor,
    mask: torch.Tensor,
    dropout_mask: torch.Tensor,
    p_dropout: float,
) -> torch.Tensor:
    B, H, T, _ = grad_output.shape
    out = torch.empty_like(grad_output)
    BLOCK = _next_pow2(T)
    apply_dropout = p_dropout > 0.0
    scale = (1.0 / (1.0 - p_dropout)) if apply_dropout else 1.0
    grid = (B * H * T,)
    _softmax_bw_kernel[grid](
        grad_output, p_attn, mask, dropout_mask, out,
        scale,
        T, H,
        APPLY_DROPOUT=apply_dropout,
        BLOCK=BLOCK,
        num_warps=8,
    )
    return out

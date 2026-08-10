import torch
import triton
import triton.language as tl


@triton.jit
def _kernel(
    grad_ptr,
    p_ptr,
    mask_ptr,
    dropout_ptr,
    out_ptr,
    n_cols: tl.constexpr,
    n_heads: tl.constexpr,
    scale,
    BLOCK: tl.constexpr,
    HAS_DROPOUT: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    valid = cols < n_cols
    offset = row * n_cols + cols

    grad = tl.load(grad_ptr + offset, mask=valid, other=0.0)
    p = tl.load(p_ptr + offset, mask=valid, other=0.0)
    if HAS_DROPOUT:
        kept = tl.load(dropout_ptr + offset, mask=valid, other=0).to(tl.float32)
        grad = grad * kept * scale

    dot = tl.sum(p * grad, axis=0)
    batch = row // (n_heads * n_cols)
    query = row % n_cols
    mask_offset = (batch * n_cols + query) * n_cols + cols
    unmasked = tl.load(mask_ptr + mask_offset, mask=valid, other=0)
    result = p * (grad - dot)
    tl.store(out_ptr + offset, tl.where(unmasked, result, 0.0), mask=valid)


@torch.no_grad()
def run(
    grad_output: torch.Tensor,
    p_attn: torch.Tensor,
    mask: torch.Tensor,
    dropout_mask: torch.Tensor,
    p_dropout: float,
) -> torch.Tensor:
    n_cols = grad_output.shape[-1]
    n_heads = grad_output.shape[1]
    rows = grad_output.numel() // n_cols
    out = torch.empty_like(grad_output)
    block = triton.next_power_of_2(n_cols)
    num_warps = 2 if block <= 512 else 4
    _kernel[(rows,)](
        grad_output,
        p_attn,
        mask,
        dropout_mask,
        out,
        n_cols,
        n_heads,
        1.0 / (1.0 - p_dropout) if p_dropout > 0.0 else 1.0,
        BLOCK=block,
        HAS_DROPOUT=p_dropout > 0.0,
        num_warps=num_warps,
        num_stages=2,
    )
    return out

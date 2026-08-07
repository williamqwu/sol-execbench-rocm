import torch
import triton
import triton.language as tl


@triton.jit
def softcap_softmax_kernel(
    x_ptr, out_ptr,
    n_cols,
    x_row_stride,
    out_row_stride,
    BLOCK_K: tl.constexpr,
):
    row = tl.program_id(0)
    x_row_ptr = x_ptr + row * x_row_stride
    out_row_ptr = out_ptr + row * out_row_stride

    offs = tl.arange(0, BLOCK_K)
    mask = offs < n_cols

    x = tl.load(x_row_ptr + offs, mask=mask, other=-float('inf')).to(tl.float32)
    # softcap: tanh(x/30) * 30, via stable exp form
    t = x * (1.0 / 30.0)
    a = tl.exp(-2.0 * tl.abs(t))
    tanh_t = (1.0 - a) / (1.0 + a)
    tanh_t = tl.where(t < 0, -tanh_t, tanh_t)
    sc = tanh_t * 30.0

    # softmax (row-wise)
    m = tl.max(sc, axis=0)
    e = tl.exp(sc - m)
    s = tl.sum(e, axis=0)
    y = e / s

    tl.store(out_row_ptr + offs, y.to(tl.bfloat16), mask=mask)


@torch.no_grad()
def run(attn_weights: torch.Tensor) -> torch.Tensor:
    n_cols = attn_weights.shape[-1]
    x = attn_weights
    orig_shape = x.shape
    if not x.is_contiguous():
        x = x.contiguous()
    x2d = x.reshape(-1, n_cols)
    n_rows = x2d.shape[0]
    out = torch.empty_like(x2d)

    BLOCK_K = triton.next_power_of_2(n_cols)
    if BLOCK_K <= 256:
        num_warps = 2
    elif BLOCK_K <= 1024:
        num_warps = 4
    else:
        num_warps = 8

    grid = (n_rows,)
    softcap_softmax_kernel[grid](
        x2d, out, n_cols,
        x2d.stride(0),
        out.stride(0),
        BLOCK_K=BLOCK_K,
        num_warps=num_warps,
    )
    return out.reshape(orig_shape)

import torch
import triton
import triton.language as tl
import triton.language.extra.libdevice as libdevice


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
    sc = libdevice.tanh(x * (1.0 / 30.0)) * 30.0

    m = tl.max(sc, axis=0)
    e = tl.exp(sc - m)
    s = tl.sum(e, axis=0)
    y = e / s

    tl.store(out_row_ptr + offs, y.to(tl.bfloat16), mask=mask)


@triton.jit
def softcap_softmax_multirow_kernel(
    x_ptr, out_ptr,
    n_rows, n_cols,
    x_row_stride,
    out_row_stride,
    BLOCK_K: tl.constexpr,
    ROWS_PER_PROG: tl.constexpr,
):
    pid = tl.program_id(0)
    row_start = pid * ROWS_PER_PROG
    offs_k = tl.arange(0, BLOCK_K)
    mask_k = offs_k < n_cols
    for r in range(ROWS_PER_PROG):
        row = row_start + r
        x = tl.load(x_ptr + row * x_row_stride + offs_k, mask=mask_k, other=-float('inf')).to(tl.float32)
        sc = libdevice.tanh(x * (1.0 / 30.0)) * 30.0
        m = tl.max(sc, axis=0)
        e = tl.exp(sc - m)
        s = tl.sum(e, axis=0)
        y = e / s
        tl.store(out_ptr + row * out_row_stride + offs_k, y.to(tl.bfloat16), mask=mask_k)


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

    # Many short rows: pack multiple rows per program for better occupancy
    if BLOCK_K <= 128 and n_rows >= 49152:
        rpp = 2
        grid = (triton.cdiv(n_rows, rpp),)
        softcap_softmax_multirow_kernel[grid](
            x2d, out, n_rows, n_cols,
            x2d.stride(0), out.stride(0),
            BLOCK_K=BLOCK_K,
            ROWS_PER_PROG=rpp,
            num_warps=1,
            num_stages=2,
        )
    else:
        grid = (n_rows,)
        softcap_softmax_kernel[grid](
            x2d, out, n_cols,
            x2d.stride(0),
            out.stride(0),
            BLOCK_K=BLOCK_K,
            num_warps=1,
        )
    return out.reshape(orig_shape)

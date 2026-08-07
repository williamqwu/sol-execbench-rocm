import torch
import triton
import triton.language as tl
import triton.language.extra.libdevice as libdevice


@triton.jit
def softcap_softmax_kernel(
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
        # softcap: tanh(x/30) * 30
        sc = libdevice.tanh(x * (1.0 / 30.0)) * 30.0
        # softmax (row-wise), fp32 for stability
        m = tl.max(sc, axis=0)
        e = tl.exp(sc - m)
        s = tl.sum(e, axis=0)
        y = e / s
        tl.store(out_ptr + row * out_row_stride + offs_k, y.to(tl.bfloat16), mask=mask_k)


def _get_config(n_cols, n_rows):
    BLOCK_K = triton.next_power_of_2(n_cols)
    if BLOCK_K <= 128:
        # small rows: pack multiple rows per program for occupancy
        if n_rows >= 49152:
            return BLOCK_K, 1, 8, 2   # 64x128
        elif n_rows >= 24576:
            return BLOCK_K, 1, 4, 2
        elif n_rows >= 12288:
            return BLOCK_K, 2, 1, 2
        else:
            return BLOCK_K, 4, 1, 1
    elif BLOCK_K <= 256:
        if n_rows >= 24576:
            return BLOCK_K, 1, 2, 1
        elif n_rows >= 8192:
            return BLOCK_K, 1, 2, 1
        else:
            return BLOCK_K, 4, 4, 2
    elif BLOCK_K <= 512:
        if n_rows >= 24576:
            return BLOCK_K, 1, 1, 2
        elif n_rows >= 8192:
            return BLOCK_K, 1, 1, 2
        else:
            return BLOCK_K, 4, 4, 1
    elif BLOCK_K <= 1024:
        return BLOCK_K, 1, 1, 2
    else:  # 2048
        return BLOCK_K, 1, 1, 2


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

    BLOCK_K, num_warps, rpp, num_stages = _get_config(n_cols, n_rows)
    grid = (triton.cdiv(n_rows, rpp),)

    softcap_softmax_kernel[grid](
        x2d, out, n_rows, n_cols,
        x2d.stride(0),
        out.stride(0),
        BLOCK_K=BLOCK_K,
        ROWS_PER_PROG=rpp,
        num_warps=num_warps,
        num_stages=num_stages,
    )
    return out.reshape(orig_shape)

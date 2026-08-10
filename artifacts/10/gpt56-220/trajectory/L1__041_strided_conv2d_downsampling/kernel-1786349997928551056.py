import torch
import triton
import triton.language as tl


@triton.jit
def _conv_kernel(x, weight, bias, output, n_elements: tl.constexpr,
                 height: tl.constexpr, width: tl.constexpr,
                 out_h: tl.constexpr, out_w: tl.constexpr,
                 BLOCK_M: tl.constexpr, BLOCK_K: tl.constexpr):
    pid = tl.program_id(0)
    m = pid * BLOCK_M + tl.arange(0, BLOCK_M)
    oc = tl.arange(0, 64)
    valid_m = m < n_elements
    spatial = m % (out_h * out_w)
    n = m // (out_h * out_w)
    oh = spatial // out_w
    ow = spatial % out_w
    acc = tl.zeros((BLOCK_M, 64), tl.float32)

    for k0 in range(0, 288, BLOCK_K):
        k = k0 + tl.arange(0, BLOCK_K)
        tap = k // 32
        channel = k % 32
        kh = tap // 3
        kw = tap % 3
        ih = oh[:, None] * 2 + kh[None, :] - 1
        iw = ow[:, None] * 2 + kw[None, :] - 1
        x_off = ((n[:, None] * 32 + channel[None, :]) * height + ih) * width + iw
        x_mask = valid_m[:, None] & (ih >= 0) & (ih < height) & (iw >= 0) & (iw < width)
        a = tl.load(x + x_off, mask=x_mask, other=0.0)
        w_off = oc[None, :] * 288 + channel[:, None] * 9 + tap[:, None]
        b = tl.load(weight + w_off)
        acc += tl.dot(a, b, input_precision="ieee")

    acc += tl.load(bias + oc)[None, :]
    out_off = (n[:, None] * 64 + oc[None, :]) * (out_h * out_w) + spatial[:, None]
    tl.store(output + out_off, acc, mask=valid_m[:, None])


@torch.no_grad()
def run(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
    n, _, height, width = x.shape
    out_h = (height + 1) // 2
    out_w = (width + 1) // 2
    output = torch.empty((n, 64, out_h, out_w), device=x.device, dtype=x.dtype)
    n_elements = n * out_h * out_w
    block_m = 32
    _conv_kernel[(triton.cdiv(n_elements, block_m),)](
        x, weight, bias, output, n_elements, height, width, out_h, out_w,
        BLOCK_M=block_m, BLOCK_K=16, num_warps=8,
    )
    return output

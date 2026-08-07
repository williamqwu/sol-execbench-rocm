import torch
import triton
import triton.language as tl


_MIOPEN_CONV = torch.ops.aten.miopen_convolution.default
_PADDING = [1, 1]
_STRIDE = [2, 2]
_DILATION = [1, 1]


@triton.jit
def _bias_add_1d(y_ptr, b_ptr, NUMEL: tl.constexpr, SPATIAL: tl.constexpr,
                 BLOCK: tl.constexpr):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offs < NUMEL
    channel = (offs // SPATIAL) % 64
    value = tl.load(y_ptr + offs, mask=mask)
    value += tl.load(b_ptr + channel, mask=mask)
    tl.store(y_ptr + offs, value, mask=mask)


@triton.jit
def _bias_add_2d(y_ptr, b_ptr, SPATIAL: tl.constexpr, BLOCK: tl.constexpr):
    pixel = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    nc = tl.program_id(1)
    mask = pixel < SPATIAL
    offs = nc * SPATIAL + pixel
    value = tl.load(y_ptr + offs, mask=mask)
    value += tl.load(b_ptr + (nc % 64))
    tl.store(y_ptr + offs, value, mask=mask)


@torch.no_grad()
def run(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
    # Keep MIOpen's exact per-shape convolution reduction order, but replace
    # its generic broadcast-bias kernel with a wider specialized epilogue.
    y = _MIOPEN_CONV(
        x, weight, None, _PADDING, _STRIDE, _DILATION, 1, False, False
    )
    spatial = y.shape[2] * y.shape[3]
    numel = y.numel()

    if spatial < 8192:
        _bias_add_1d[((numel + 1023) // 1024,)](
            y, bias, NUMEL=numel, SPATIAL=spatial, BLOCK=1024, num_warps=4
        )
    elif numel >= 4_000_000 and x.shape[0] != 64:
        _bias_add_2d[((spatial + 2047) // 2048, x.shape[0] * 64)](
            y, bias, SPATIAL=spatial, BLOCK=2048, num_warps=8
        )
    else:
        _bias_add_2d[((spatial + 1023) // 1024, x.shape[0] * 64)](
            y, bias, SPATIAL=spatial, BLOCK=1024, num_warps=4
        )
    return y

import torch
import triton
import triton.language as tl


SQRT_2_OVER_PI = 0.7978845608028654  # math.sqrt(2.0 / math.pi)
COEFF = 0.044715


@triton.jit
def _gelu_tanh_kernel(
    X, OUT,
    n_elements,
    SQ: tl.constexpr,
    CF: tl.constexpr,
    BLOCK: tl.constexpr,
    UNROLL: tl.constexpr,
):
    pid = tl.program_id(0)
    base = pid * (BLOCK * UNROLL)
    for u in tl.static_range(UNROLL):
        offs = base + u * BLOCK + tl.arange(0, BLOCK)
        x = tl.load(X + offs, mask=offs < n_elements, other=0.0)
        x3 = x * x * x
        inner = SQ * (x + CF * x3)
        t = tl.extra.libdevice.tanh(inner)
        out = (0.5 * x) * (1.0 + t)
        tl.store(OUT + offs, out, mask=offs < n_elements)


@torch.no_grad()
def run(x: torch.Tensor) -> torch.Tensor:
    xc = x if x.is_contiguous() else x.contiguous()
    out = torch.empty_like(xc)
    n = xc.numel()
    if n == 0:
        return out.view(x.shape)

    BLOCK = 1024
    UNROLL = 8
    per_prog = BLOCK * UNROLL
    grid = (triton.cdiv(n, per_prog),)
    _gelu_tanh_kernel[grid](
        xc, out, n,
        SQ=SQRT_2_OVER_PI, CF=COEFF,
        BLOCK=BLOCK, UNROLL=UNROLL,
        num_warps=4, num_stages=1,
    )
    return out.view(x.shape)

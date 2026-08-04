import torch
import triton
import triton.language as tl


@triton.jit
def _cos_sin_emb_kernel(
    freqs_ptr,
    cos_ptr,
    sin_ptr,
    n_rows,
    scale,
    D: tl.constexpr,       # half head_dim (input row width)
    BLOCK_R: tl.constexpr,  # rows per program
):
    pid = tl.program_id(0)
    r = pid * BLOCK_R + tl.arange(0, BLOCK_R)
    c = tl.arange(0, D)

    mask = r[:, None] < n_rows
    off = r[:, None] * D + c[None, :]

    x = tl.load(freqs_ptr + off, mask=mask, other=0.0)

    co = tl.cos(x) * scale
    si = tl.sin(x) * scale

    co = co.to(tl.bfloat16)
    si = si.to(tl.bfloat16)

    # output row width is 2*D; the two halves are identical (torch.cat((f, f)))
    o = r[:, None] * (2 * D) + c[None, :]
    tl.store(cos_ptr + o, co, mask=mask)
    tl.store(cos_ptr + o + D, co, mask=mask)
    tl.store(sin_ptr + o, si, mask=mask)
    tl.store(sin_ptr + o + D, si, mask=mask)


def _pick(n_rows: int, D: int):
    """Choose rows-per-block / num_warps so that we fill the GPU but keep
    enough work per lane on the large shapes."""
    # target roughly 2048 workgroups (256 CUs * 8) before growing the tile
    target = 2048
    block_r = 1
    while block_r < 16 and n_rows > target * block_r * 2:
        block_r *= 2

    elems = block_r * D
    # AMD wave = 64 lanes; aim for ~4 elements per lane
    warps = max(1, min(8, elems // 256))
    return block_r, warps


@torch.no_grad()
def run(freqs: torch.Tensor, attention_scaling: float):
    if not freqs.is_contiguous():
        freqs = freqs.contiguous()

    shape = freqs.shape
    D = shape[-1]
    n_rows = freqs.numel() // D

    out_shape = shape[:-1] + (2 * D,)
    cos = torch.empty(out_shape, dtype=torch.bfloat16, device=freqs.device)
    sin = torch.empty(out_shape, dtype=torch.bfloat16, device=freqs.device)

    if n_rows == 0:
        return cos, sin

    block_r, warps = _pick(n_rows, D)
    grid = (triton.cdiv(n_rows, block_r),)

    _cos_sin_emb_kernel[grid](
        freqs,
        cos,
        sin,
        n_rows,
        float(attention_scaling),
        D=D,
        BLOCK_R=block_r,
        num_warps=warps,
        num_stages=1,
    )
    return cos, sin

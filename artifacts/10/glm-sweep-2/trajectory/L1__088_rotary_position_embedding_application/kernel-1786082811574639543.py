import torch
import triton
import triton.language as tl


@triton.jit
def _rope_kernel(
    x_ptr, cos_ptr, sin_ptr, out_ptr,
    n_rows, seq_len,
    head_dim: tl.constexpr, half_dim: tl.constexpr,
    BLOCK_ROW: tl.constexpr,
):
    pid = tl.program_id(0)
    row_off = pid * BLOCK_ROW + tl.arange(0, BLOCK_ROW)
    mask = row_off < n_rows

    h_offs = tl.arange(0, half_dim)
    x_base = row_off * head_dim
    seq_idx = row_off % seq_len
    cos_base = seq_idx * head_dim

    x1 = tl.load(x_ptr + x_base[:, None] + h_offs[None, :], mask=mask[:, None], other=0.0)
    x2 = tl.load(x_ptr + x_base[:, None] + (h_offs + half_dim)[None, :], mask=mask[:, None], other=0.0)

    cos1 = tl.load(cos_ptr + cos_base[:, None] + h_offs[None, :], mask=mask[:, None], other=0.0)
    cos2 = tl.load(cos_ptr + cos_base[:, None] + (h_offs + half_dim)[None, :], mask=mask[:, None], other=0.0)
    sin1 = tl.load(sin_ptr + cos_base[:, None] + h_offs[None, :], mask=mask[:, None], other=0.0)
    sin2 = tl.load(sin_ptr + cos_base[:, None] + (h_offs + half_dim)[None, :], mask=mask[:, None], other=0.0)

    out1 = x1 * cos1 - x2 * sin1
    out2 = x2 * cos2 + x1 * sin2

    out_base = row_off * head_dim
    tl.store(out_ptr + out_base[:, None] + h_offs[None, :], out1, mask=mask[:, None])
    tl.store(out_ptr + out_base[:, None] + (h_offs + half_dim)[None, :], out2, mask=mask[:, None])


def _apply_rope(x, cos, sin, head_dim, half_dim, BLOCK_ROW):
    b, h, s, d = x.shape
    n_rows = b * h * s
    out = torch.empty_like(x)
    grid = (triton.cdiv(n_rows, BLOCK_ROW),)
    _rope_kernel[grid](
        x, cos, sin, out, n_rows, s,
        head_dim=head_dim, half_dim=half_dim,
        BLOCK_ROW=BLOCK_ROW,
    )
    return out


@torch.no_grad()
def run(
    query: torch.Tensor,
    key: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
):
    head_dim = query.shape[-1]
    half_dim = head_dim // 2
    n_q = query.numel() // head_dim
    BLOCK_ROW = 4 if n_q < 8192 else 8
    query_rotated = _apply_rope(query, cos, sin, head_dim, half_dim, BLOCK_ROW)
    key_rotated = _apply_rope(key, cos, sin, head_dim, half_dim, BLOCK_ROW)
    return query_rotated, key_rotated

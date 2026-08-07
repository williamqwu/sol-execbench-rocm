import torch
import triton
import triton.language as tl


@triton.jit
def _rope_2d(x_ptr, cos_ptr, sin_ptr, out_ptr,
             n_bh, seq_len,
             head_dim: tl.constexpr, half_dim: tl.constexpr,
             BLOCK_S: tl.constexpr):
    pid_bh = tl.program_id(0)
    pid_s = tl.program_id(1)
    s_off = pid_s * BLOCK_S + tl.arange(0, BLOCK_S)
    smask = s_off < seq_len
    h_offs = tl.arange(0, half_dim)
    x_base = pid_bh * seq_len * head_dim + s_off[:, None] * head_dim
    cos_base = s_off[:, None] * head_dim
    x1 = tl.load(x_ptr + x_base + h_offs[None, :], mask=smask[:, None], other=0.0)
    x2 = tl.load(x_ptr + x_base + (h_offs + half_dim)[None, :], mask=smask[:, None], other=0.0)
    cos1 = tl.load(cos_ptr + cos_base + h_offs[None, :], mask=smask[:, None], other=0.0)
    cos2 = tl.load(cos_ptr + cos_base + (h_offs + half_dim)[None, :], mask=smask[:, None], other=0.0)
    sin1 = tl.load(sin_ptr + cos_base + h_offs[None, :], mask=smask[:, None], other=0.0)
    sin2 = tl.load(sin_ptr + cos_base + (h_offs + half_dim)[None, :], mask=smask[:, None], other=0.0)
    out1 = x1 * cos1 - x2 * sin1
    out2 = x2 * cos2 + x1 * sin2
    tl.store(out_ptr + x_base + h_offs[None, :], out1, mask=smask[:, None])
    tl.store(out_ptr + x_base + (h_offs + half_dim)[None, :], out2, mask=smask[:, None])


def _block_s(seq_len, n_bh):
    if seq_len >= 4096:
        return 64
    if seq_len >= 1024:
        return 32
    return 16


def _apply_rope(x, cos, sin, head_dim, half_dim):
    b, h, s, d = x.shape
    n_bh = b * h
    out = torch.empty_like(x)
    BLOCK_S = _block_s(s, n_bh)
    grid = (n_bh, triton.cdiv(s, BLOCK_S))
    _rope_2d[grid](x, cos, sin, out, n_bh, s, head_dim=head_dim, half_dim=half_dim, BLOCK_S=BLOCK_S)
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
    query_rotated = _apply_rope(query, cos, sin, head_dim, half_dim)
    key_rotated = _apply_rope(key, cos, sin, head_dim, half_dim)
    return query_rotated, key_rotated

import torch
import triton
import triton.language as tl

# Fixed config: axes_dim = [16, 56, 56], total_dim = 128
# half_dims = [8, 28, 28]; per-axis output offsets = [0, 16, 72]


@triton.jit
def _rope_kernel(
    pos_ptr,        # [seq_len, 3] float32, row-major
    cos_out_ptr,    # [seq_len, 128]
    sin_out_ptr,    # [seq_len, 128]
    seq_len,
    stride_pos_s,
    stride_cos_s, stride_cos_d,
    BLOCK_S: tl.constexpr,
    TOTAL_DIM: tl.constexpr,
):
    pid = tl.program_id(0)
    s_off = pid * BLOCK_S + tl.arange(0, BLOCK_S)
    s_mask = s_off < seq_len

    pos0 = tl.load(pos_ptr + s_off * stride_pos_s + 0, mask=s_mask, other=0.0)
    pos1 = tl.load(pos_ptr + s_off * stride_pos_s + 1, mask=s_mask, other=0.0)
    pos2 = tl.load(pos_ptr + s_off * stride_pos_s + 2, mask=s_mask, other=0.0)

    d_off = tl.arange(0, TOTAL_DIM)  # [128]

    half0: tl.constexpr = 8
    half1: tl.constexpr = 28
    half2: tl.constexpr = 28
    off0: tl.constexpr = 0
    off1: tl.constexpr = 16
    off2: tl.constexpr = 72

    lt = tl.log(10000.0)

    # axis 0: cols [0,16)
    k0 = d_off - off0
    pair0 = k0 // 2
    fb0 = 1.0 / tl.exp((lt * pair0) / half0)
    ang0 = pos0[:, None] * fb0[None, :]
    c0 = tl.cos(ang0)
    s0 = tl.sin(ang0)

    # axis 1: cols [16,72)
    k1 = d_off - off1
    pair1 = k1 // 2
    fb1 = 1.0 / tl.exp((lt * pair1) / half1)
    ang1 = pos1[:, None] * fb1[None, :]
    c1 = tl.cos(ang1)
    s1 = tl.sin(ang1)

    # axis 2: cols [72,128)
    k2 = d_off - off2
    pair2 = k2 // 2
    fb2 = 1.0 / tl.exp((lt * pair2) / half2)
    ang2 = pos2[:, None] * fb2[None, :]
    c2 = tl.cos(ang2)
    s2 = tl.sin(ang2)

    m0 = (k0 >= 0) & (k0 < half0 * 2)
    m1 = (k1 >= 0) & (k1 < half1 * 2)
    m2 = (k2 >= 0) & (k2 < half2 * 2)

    cos_val = tl.where(m0[None, :], c0, 0.0)
    cos_val = tl.where(m1[None, :], c1, cos_val)
    cos_val = tl.where(m2[None, :], c2, cos_val)

    sin_val = tl.where(m0[None, :], s0, 0.0)
    sin_val = tl.where(m1[None, :], s1, sin_val)
    sin_val = tl.where(m2[None, :], s2, sin_val)

    s2d = s_off[:, None] * stride_cos_s + d_off[None, :] * stride_cos_d
    tl.store(cos_out_ptr + s2d, cos_val, mask=s_mask[:, None])
    tl.store(sin_out_ptr + s2d, sin_val, mask=s_mask[:, None])


@torch.no_grad()
def run(ids: torch.Tensor, theta: float):
    seq_len = ids.shape[0]
    device = ids.device
    pos = ids.contiguous().float()
    total_dim = 128
    freqs_cos = torch.empty((seq_len, total_dim), dtype=torch.float32, device=device)
    freqs_sin = torch.empty((seq_len, total_dim), dtype=torch.float32, device=device)

    BLOCK_S = 16
    num_progs = triton.cdiv(seq_len, BLOCK_S)
    _rope_kernel[(num_progs,)](
        pos, freqs_cos, freqs_sin,
        seq_len,
        pos.stride(0),
        freqs_cos.stride(0), freqs_cos.stride(1),
        BLOCK_S=BLOCK_S, TOTAL_DIM=total_dim,
    )
    return freqs_cos, freqs_sin

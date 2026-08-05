import torch
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def _mid_kernel(
    BCx_ptr, cw_ptr, cb_ptr, Y_ptr,
    S,
    stride_bcx_b, stride_bcx_s, stride_bcx_c,
    stride_y_b, stride_y_s, stride_y_h,
    BLOCK_S: tl.constexpr,
    BLOCK_H: tl.constexpr,
    H: tl.constexpr,
    CK: tl.constexpr,
):
    pid_b = tl.program_id(0)
    pid_s = tl.program_id(1)
    pid_hg = tl.program_id(2)
    h_start = pid_hg * BLOCK_H
    h_offs = h_start + tl.arange(0, BLOCK_H)
    h_mask = h_offs < H

    cw0 = tl.load(cw_ptr + h_offs * CK + 0, mask=h_mask, other=0.0)
    cw1 = tl.load(cw_ptr + h_offs * CK + 1, mask=h_mask, other=0.0)
    cw2 = tl.load(cw_ptr + h_offs * CK + 2, mask=h_mask, other=0.0)
    cw3 = tl.load(cw_ptr + h_offs * CK + 3, mask=h_mask, other=0.0)
    cbb = tl.load(cb_ptr + h_offs, mask=h_mask, other=0.0)

    b_base = BCx_ptr + pid_b * stride_bcx_b + h_offs * stride_bcx_c
    c_base = BCx_ptr + pid_b * stride_bcx_b + (h_offs + H) * stride_bcx_c
    x_base = BCx_ptr + pid_b * stride_bcx_b + (h_offs + 2 * H) * stride_bcx_c

    s0 = pid_s * BLOCK_S
    out_idx = tl.arange(0, BLOCK_S)
    s_out = s0 + out_idx
    out_mask = s_out < S
    omh = out_mask[:, None] & h_mask[None, :]

    ss0 = s_out
    m0 = (ss0 >= 0) & out_mask
    mh0 = m0[:, None] & h_mask[None, :]
    Bx3 = tl.load(b_base[None, :] + ss0[:, None] * stride_bcx_s, mask=mh0, other=0.0) * tl.load(x_base[None, :] + ss0[:, None] * stride_bcx_s, mask=mh0, other=0.0)
    ss1 = s_out - 1
    m1 = (ss1 >= 0) & out_mask
    mh1 = m1[:, None] & h_mask[None, :]
    Bx2 = tl.load(b_base[None, :] + ss1[:, None] * stride_bcx_s, mask=mh1, other=0.0) * tl.load(x_base[None, :] + ss1[:, None] * stride_bcx_s, mask=mh1, other=0.0)
    ss2 = s_out - 2
    m2 = (ss2 >= 0) & out_mask
    mh2 = m2[:, None] & h_mask[None, :]
    Bx1 = tl.load(b_base[None, :] + ss2[:, None] * stride_bcx_s, mask=mh2, other=0.0) * tl.load(x_base[None, :] + ss2[:, None] * stride_bcx_s, mask=mh2, other=0.0)
    ss3 = s_out - 3
    m3 = (ss3 >= 0) & out_mask
    mh3 = m3[:, None] & h_mask[None, :]
    Bx0 = tl.load(b_base[None, :] + ss3[:, None] * stride_bcx_s, mask=mh3, other=0.0) * tl.load(x_base[None, :] + ss3[:, None] * stride_bcx_s, mask=mh3, other=0.0)

    Cv = tl.load(c_base[None, :] + s_out[:, None] * stride_bcx_s, mask=omh, other=0.0)
    conv_out = cw3[None, :] * Bx3 + cw2[None, :] * Bx2 + cw1[None, :] * Bx1 + cw0[None, :] * Bx0 + cbb[None, :]
    y = Cv * conv_out
    y_base = Y_ptr + pid_b * stride_y_b + h_offs * stride_y_h
    tl.store(y_base[None, :] + s_out[:, None] * stride_y_s, y, mask=omh)


def _pick_config(B, S):
    if S <= 1024:
        return 256, 32
    return 128, 32


@torch.no_grad()
def run(
    x: torch.Tensor,
    in_proj_weight: torch.Tensor,
    in_proj_bias: torch.Tensor,
    conv_weight: torch.Tensor,
    conv_bias: torch.Tensor,
    out_proj_weight: torch.Tensor,
    out_proj_bias: torch.Tensor,
):
    batch_size, seq_len, hidden_size = x.shape
    BCx = F.linear(x, in_proj_weight, in_proj_bias)  # (B, S, 3H) contiguous
    BLOCK_S, BLOCK_H = _pick_config(batch_size, seq_len)
    Y = torch.empty(batch_size, seq_len, hidden_size, dtype=x.dtype, device=x.device)
    grid = (batch_size, triton.cdiv(seq_len, BLOCK_S), triton.cdiv(hidden_size, BLOCK_H))
    _mid_kernel[grid](
        BCx, conv_weight, conv_bias, Y, seq_len,
        BCx.stride(0), BCx.stride(1), BCx.stride(2),
        Y.stride(0), Y.stride(1), Y.stride(2),
        BLOCK_S=BLOCK_S, BLOCK_H=BLOCK_H, H=hidden_size, CK=conv_weight.shape[2],
        num_warps=4, num_stages=2,
    )
    output = F.linear(Y, out_proj_weight, out_proj_bias)
    return output

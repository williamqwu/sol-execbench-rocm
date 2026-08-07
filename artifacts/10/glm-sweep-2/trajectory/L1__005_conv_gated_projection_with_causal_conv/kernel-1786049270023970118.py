import torch
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def _fused_gate_conv_gate_kernel(
    B_ptr, C_ptr, XP_ptr, W_ptr, BIAS_ptr, Y_ptr,
    sb, ss,
    S, H,
    K: tl.constexpr,
    BLOCK_S: tl.constexpr,
    BLOCK_H: tl.constexpr,
):
    pid_b = tl.program_id(0)
    pid_s = tl.program_id(1)
    pid_h = tl.program_id(2)

    h_off = pid_h * BLOCK_H + tl.arange(0, BLOCK_H)
    hmask = h_off < H

    s_start = pid_s * BLOCK_S
    s_idx = s_start + tl.arange(0, BLOCK_S)
    smask = s_idx < S

    bias = tl.load(BIAS_ptr + h_off, mask=hmask, other=0.0).to(tl.float32)

    base = pid_b * sb + s_idx[:, None] * ss + h_off[None, :]

    acc = tl.zeros([BLOCK_S, BLOCK_H], dtype=tl.float32)
    for k in tl.static_range(K):
        src_idx = s_idx - k
        sm = smask & (src_idx >= 0)
        sc = tl.where(sm, src_idx, 0)
        off = pid_b * sb + sc[:, None] * ss + h_off[None, :]
        m2 = sm[:, None] & hmask[None, :]
        b_val = tl.load(B_ptr + off, mask=m2, other=0.0).to(tl.float32)
        xp_val = tl.load(XP_ptr + off, mask=m2, other=0.0).to(tl.float32)
        bx = b_val * xp_val
        wk = tl.load(W_ptr + h_off * K + (K - 1 - k), mask=hmask, other=0.0).to(tl.float32)
        acc += wk[None, :] * bx
    acc = acc + bias[None, :]
    c_val = tl.load(C_ptr + base, mask=smask[:, None] & hmask[None, :], other=0.0).to(tl.float32)
    y = acc * c_val
    tl.store(Y_ptr + base, y.to(Y_ptr.dtype.element_ty), mask=smask[:, None] & hmask[None, :])


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
    conv_kernel_size = conv_weight.shape[2]

    BCx = F.linear(x, in_proj_weight, in_proj_bias)
    B, C, x_proj = BCx.chunk(3, dim=-1)
    B = B.contiguous()
    C = C.contiguous()
    x_proj = x_proj.contiguous()

    Y = torch.empty_like(B)
    w = conv_weight.squeeze(1).contiguous()
    cb = conv_bias.contiguous()
    if seq_len >= 1024:
        BLOCK_S, BLOCK_H = 256, 32
    elif seq_len >= 512:
        BLOCK_S, BLOCK_H = 128, 64
    else:
        BLOCK_S, BLOCK_H = 32, 256
    grid = (batch_size, triton.cdiv(seq_len, BLOCK_S), triton.cdiv(hidden_size, BLOCK_H))
    _fused_gate_conv_gate_kernel[grid](
        B, C, x_proj, w, cb, Y,
        B.stride(0), B.stride(1),
        seq_len, hidden_size, K=conv_kernel_size, BLOCK_S=BLOCK_S, BLOCK_H=BLOCK_H,
    )

    output = F.linear(Y, out_proj_weight, out_proj_bias)
    return output

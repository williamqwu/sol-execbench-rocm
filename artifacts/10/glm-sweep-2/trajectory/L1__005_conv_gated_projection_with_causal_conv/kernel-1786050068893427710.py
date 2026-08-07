import torch
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def _fused_gate_conv_gate_kernel(
    BCX_ptr, W_ptr, BIAS_ptr, Y_ptr,
    sb, ss, sh,
    S, H,
    offB, offC, offXP,
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

    baseB = pid_b * sb + offB
    baseC = pid_b * sb + offC
    baseXP = pid_b * sb + offXP
    out_base = pid_b * (S * H) + s_idx[:, None] * H + h_off[None, :]

    acc = tl.zeros([BLOCK_S, BLOCK_H], dtype=tl.float32)
    for k in tl.static_range(K):
        src_idx = s_idx - k
        sm = smask & (src_idx >= 0)
        sc = tl.where(sm, src_idx, 0)
        off = sc[:, None] * ss + h_off[None, :] * sh
        m2 = sm[:, None] & hmask[None, :]
        b_val = tl.load(BCX_ptr + baseB + off, mask=m2, other=0.0).to(tl.float32)
        xp_val = tl.load(BCX_ptr + baseXP + off, mask=m2, other=0.0).to(tl.float32)
        bx = b_val * xp_val
        wk = tl.load(W_ptr + h_off * K + (K - 1 - k), mask=hmask, other=0.0).to(tl.float32)
        acc += wk[None, :] * bx
    acc = acc + bias[None, :]
    c_val = tl.load(BCX_ptr + baseC + s_idx[:, None] * ss + h_off[None, :] * sh,
                    mask=smask[:, None] & hmask[None, :], other=0.0).to(tl.float32)
    y = acc * c_val
    tl.store(Y_ptr + out_base, y.to(Y_ptr.dtype.element_ty),
             mask=smask[:, None] & hmask[None, :])


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

    # Triple linear projection -> [B, S, 3H] contiguous
    BCx = F.linear(x, in_proj_weight, in_proj_bias)
    H = hidden_size

    # Output of the fused gate-conv-gate stage, contiguous [B, S, H]
    Y = torch.empty(batch_size, seq_len, H, dtype=x.dtype, device=x.device)
    w = conv_weight.squeeze(1).contiguous()
    cb = conv_bias.contiguous()

    if seq_len >= 1024:
        BLOCK_S, BLOCK_H = 256, 64
    elif seq_len >= 512:
        BLOCK_S, BLOCK_H = 128, 64
    else:
        BLOCK_S, BLOCK_H = 32, 256
    grid = (batch_size, triton.cdiv(seq_len, BLOCK_S), triton.cdiv(H, BLOCK_H))
    _fused_gate_conv_gate_kernel[grid](
        BCx, w, cb, Y,
        BCx.stride(0), BCx.stride(1), BCx.stride(2),
        seq_len, H,
        0, H, 2 * H,
        K=conv_kernel_size, BLOCK_S=BLOCK_S, BLOCK_H=BLOCK_H,
    )

    output = F.linear(Y, out_proj_weight, out_proj_bias)
    return output

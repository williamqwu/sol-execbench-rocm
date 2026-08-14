import torch
import triton
import triton.language as tl

_F = torch.nn.functional


# ---------------------------------------------------------------------------
# Kernel A: read the B and x_proj thirds of the in_proj output, multiply them
# (bf16-rounded, as the reference does), and write the result *transposed* and
# left-zero-padded straight into the (Bsz, H, S+3) buffer conv1d wants.
# This fuses the chunk, the elementwise gate, the transpose and the F.pad.
# ---------------------------------------------------------------------------
@triton.jit
def _gate_pad_kernel(
    BCX,            # (M, 3H) bf16
    OUT,            # (Bsz, H, S+3) bf16, zero-filled at s < 3
    S, H, M,
    stride_bm,
    BS: tl.constexpr,
    BH: tl.constexpr,
    PAD: tl.constexpr,
):
    pid_s = tl.program_id(0)
    pid_h = tl.program_id(1)
    pid_b = tl.program_id(2)

    offs_s = pid_s * BS + tl.arange(0, BS)
    offs_h = pid_h * BH + tl.arange(0, BH)
    smask = offs_s < S
    hmask = offs_h < H

    rows = pid_b * S + offs_s
    src = BCX + rows[:, None] * stride_bm + offs_h[None, :]
    m2 = smask[:, None] & hmask[None, :]

    b = tl.load(src, mask=m2, other=0.0)
    xp = tl.load(src + 2 * H, mask=m2, other=0.0)
    bx = (b.to(tl.float32) * xp.to(tl.float32)).to(tl.bfloat16)

    # transposed store: (BS, BH) -> [h, s]
    dst = (OUT + pid_b * H * (S + PAD)
           + offs_h[:, None] * (S + PAD)
           + (offs_s + PAD)[None, :])
    tl.store(dst, tl.trans(bx), mask=hmask[:, None] & smask[None, :])


# ---------------------------------------------------------------------------
# Kernel B: gate the conv output with the C third and transpose back to (M, H).
# ---------------------------------------------------------------------------
@triton.jit
def _gate_out_kernel(
    BCX,            # (M, 3H) bf16
    CONV,           # (Bsz, H, S) bf16
    Y,              # (M, H) bf16
    S, H, M,
    stride_bm,
    BS: tl.constexpr,
    BH: tl.constexpr,
):
    pid_s = tl.program_id(0)
    pid_h = tl.program_id(1)
    pid_b = tl.program_id(2)

    offs_s = pid_s * BS + tl.arange(0, BS)
    offs_h = pid_h * BH + tl.arange(0, BH)
    smask = offs_s < S
    hmask = offs_h < H

    csrc = (CONV + pid_b * H * S
            + offs_h[:, None] * S
            + offs_s[None, :])
    conv = tl.load(csrc, mask=hmask[:, None] & smask[None, :], other=0.0)
    convT = tl.trans(conv)                       # (BS, BH)

    rows = pid_b * S + offs_s
    m2 = smask[:, None] & hmask[None, :]
    c = tl.load(BCX + rows[:, None] * stride_bm + (offs_h + H)[None, :],
                mask=m2, other=0.0)

    y = (c.to(tl.float32) * convT.to(tl.float32)).to(tl.bfloat16)
    tl.store(Y + rows[:, None] * H + offs_h[None, :], y, mask=m2)


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
    Bsz, S, H = x.shape
    M = Bsz * S
    K = conv_weight.shape[2]
    PAD = K - 1

    bcx = _F.linear(x.reshape(M, H), in_proj_weight, in_proj_bias)

    bxp = torch.zeros((Bsz, H, S + PAD), dtype=torch.bfloat16, device=x.device)
    BS, BH = 64, 64
    grid = (triton.cdiv(S, BS), triton.cdiv(H, BH), Bsz)
    _gate_pad_kernel[grid](
        bcx, bxp, S, H, M, bcx.stride(0),
        BS=BS, BH=BH, PAD=PAD, num_warps=4, num_stages=2,
    )

    conv = _F.conv1d(bxp, conv_weight, conv_bias, groups=H)

    y = torch.empty((M, H), dtype=torch.bfloat16, device=x.device)
    _gate_out_kernel[grid](
        bcx, conv, y, S, H, M, bcx.stride(0),
        BS=BS, BH=BH, num_warps=4, num_stages=2,
    )

    out = _F.linear(y, out_proj_weight, out_proj_bias)
    return out.view(Bsz, S, H)

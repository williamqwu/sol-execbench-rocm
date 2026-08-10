import torch
import torch.nn.functional as F
import triton
import triton.language as tl

@torch.compile(fullgraph=True, dynamic=True)
def _gate_layout(c, conv):
    return (c * conv).transpose(-1, -2).contiguous()

@triton.jit
def _causal_gate(BC, W, Bias, Y, N: tl.constexpr, S: tl.constexpr, H: tl.constexpr,
                 BT: tl.constexpr, BH: tl.constexpr):
    pid_t = tl.program_id(0)
    pid_h = tl.program_id(1)
    tt = pid_t * BT + tl.arange(0, BT)
    hh = pid_h * BH + tl.arange(0, BH)
    mask = (tt[:, None] < N) & (hh[None, :] < H)
    batch_start = (tt // S) * S
    base = tt[:, None] * (3 * H) + hh[None, :]
    c = tl.load(BC + base + H, mask=mask, other=0.0)
    bias = tl.load(Bias + hh, mask=hh < H, other=0.0).to(tl.float32)
    acc = tl.zeros((BT, BH), tl.float32) + bias[None, :]
    for k in range(4):
        src_t = tt - k
        smask = mask & (src_t[:, None] >= batch_start[:, None])
        src = src_t[:, None] * (3 * H) + hh[None, :]
        b = tl.load(BC + src, mask=smask, other=0.0)
        xp = tl.load(BC + src + 2 * H, mask=smask, other=0.0)
        bx = (b * xp).to(tl.bfloat16)
        w = tl.load(W + hh * 4 + (3 - k), mask=hh < H, other=0.0)
        acc += bx.to(tl.float32) * w[None, :].to(tl.float32)
    out = (c * acc.to(tl.bfloat16)).to(tl.bfloat16)
    tl.store(Y + tt[:, None] * H + hh[None, :], out, mask=mask)

def _fused_middle(BCx, conv_weight, conv_bias):
    B, S, three_h = BCx.shape
    H = three_h // 3
    y = torch.empty((B, S, H), device=BCx.device, dtype=BCx.dtype)
    _causal_gate[(triton.cdiv(B * S, 8), triton.cdiv(H, 256))](
        BCx, conv_weight, conv_bias, y, N=B*S, S=S, H=H, BT=8, BH=256)
    return y

@torch.no_grad()
def run(x, in_proj_weight, in_proj_bias, conv_weight, conv_bias,
        out_proj_weight, out_proj_bias):
    B, S, H = x.shape
    BCx = F.linear(x, in_proj_weight, in_proj_bias)
    y = _fused_middle(BCx, conv_weight, conv_bias)
    return F.linear(y, out_proj_weight, out_proj_bias)

import torch
import torch.nn.functional as F
import triton
import triton.language as tl

@triton.jit
def _causal_gate(BC, W, Bias, Y, S: tl.constexpr, H: tl.constexpr,
                 BT: tl.constexpr, BH: tl.constexpr):
    pid_t = tl.program_id(0)
    pid_h = tl.program_id(1)
    pid_b = tl.program_id(2)
    tt = pid_t * BT + tl.arange(0, BT)
    hh = pid_h * BH + tl.arange(0, BH)
    mask = (tt[:, None] < S) & (hh[None, :] < H)
    global_t = pid_b * S + tt
    base = global_t[:, None] * (3 * H) + hh[None, :]
    c = tl.load(BC + base + H, mask=mask, other=0.0)
    bias = tl.load(Bias + hh, mask=hh < H, other=0.0).to(tl.float32)
    acc = tl.zeros((BT, BH), tl.float32) + bias[None, :]
    for k in tl.static_range(4):
        src_t = tt - k
        smask = mask & (src_t[:, None] >= 0)
        src = (pid_b * S + src_t[:, None]) * (3 * H) + hh[None, :]
        b = tl.load(BC + src, mask=smask, other=0.0)
        xp = tl.load(BC + src + 2 * H, mask=smask, other=0.0)
        bx = (b * xp).to(tl.bfloat16)
        w = tl.load(W + hh * 4 + (3 - k), mask=hh < H, other=0.0)
        acc += bx.to(tl.float32) * w[None, :].to(tl.float32)
    out = (c * acc.to(tl.bfloat16)).to(tl.bfloat16)
    tl.store(Y + global_t[:, None] * H + hh[None, :], out, mask=mask)

def _fused_middle(BCx, conv_weight, conv_bias):
    B, S, three_h = BCx.shape
    H = three_h // 3
    y = torch.empty((B, S, H), device=BCx.device, dtype=BCx.dtype)
    _causal_gate[(triton.cdiv(S, 8), triton.cdiv(H, 512), B)](
        BCx, conv_weight, conv_bias, y, S=S, H=H, BT=8, BH=512,
        num_stages=1, waves_per_eu=2)
    return y

@torch.no_grad()
def run(x, in_proj_weight, in_proj_bias, conv_weight, conv_bias,
        out_proj_weight, out_proj_bias):
    B, S, H = x.shape
    BCx = F.linear(x, in_proj_weight, in_proj_bias)
    y = _fused_middle(BCx, conv_weight, conv_bias)
    return F.linear(y, out_proj_weight, out_proj_bias)

import torch
import torch.nn.functional as F

@torch.compile(fullgraph=True, dynamic=True)
def _gate_layout(c, conv):
    return (c * conv).transpose(-1, -2).contiguous()

@torch.compile(fullgraph=True, dynamic=True)
def _input_gate(BCx):
    b, c, xp = BCx.transpose(-1, -2).chunk(3, dim=1)
    return b * xp, c

@torch.no_grad()
def run(x, in_proj_weight, in_proj_bias, conv_weight, conv_bias,
        out_proj_weight, out_proj_bias):
    B, S, H = x.shape
    BCx = F.linear(x, in_proj_weight, in_proj_bias)
    bx, c = _input_gate(BCx)
    # Native symmetric padding produces S+3 values; its first S values are
    # exactly the left-padded causal convolution.
    conv = F.conv1d(bx, conv_weight, conv_bias, padding=3, groups=H)[..., :S]
    if B * S <= 1024:
        y = (c * conv).transpose(-1, -2).contiguous()
    else:
        y = _gate_layout(c, conv)
    return F.linear(y, out_proj_weight, out_proj_bias)

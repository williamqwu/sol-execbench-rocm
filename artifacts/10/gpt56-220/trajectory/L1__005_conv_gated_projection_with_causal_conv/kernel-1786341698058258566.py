import torch
import torch.nn.functional as F

@torch.no_grad()
def run(x, in_proj_weight, in_proj_bias, conv_weight, conv_bias,
        out_proj_weight, out_proj_bias):
    B, S, H = x.shape
    BCx = F.linear(x, in_proj_weight, in_proj_bias).transpose(-1, -2)
    b, c, xp = BCx.chunk(3, dim=1)
    bx = b * xp
    # Native symmetric padding produces S+3 values; its first S values are
    # exactly the left-padded causal convolution.
    conv = F.conv1d(bx, conv_weight, conv_bias, padding=3, groups=H)[..., :S]
    y = (c * conv).transpose(-1, -2).contiguous()
    return F.linear(y, out_proj_weight, out_proj_bias)

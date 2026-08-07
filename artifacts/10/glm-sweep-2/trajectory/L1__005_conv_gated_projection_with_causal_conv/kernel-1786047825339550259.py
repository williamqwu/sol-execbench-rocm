import torch
import torch.nn.functional as F


def _run(x, in_proj_weight, in_proj_bias, conv_weight, conv_bias, out_proj_weight, out_proj_bias):
    conv_kernel_size = conv_weight.shape[2]
    BCx = F.linear(x, in_proj_weight, in_proj_bias)
    B, C, x_proj = BCx.chunk(3, dim=-1)
    Bx = B * x_proj
    Bx_padded = F.pad(Bx, (0, 0, conv_kernel_size - 1, 0))
    windows = Bx_padded.unfold(1, conv_kernel_size, 1)
    w = conv_weight.squeeze(1)
    conv_out = (windows * w).sum(-1) + conv_bias
    y = C * conv_out
    output = F.linear(y, out_proj_weight, out_proj_bias)
    return output


_compiled = torch.compile(_run, mode="default", dynamic=False)


@torch.no_grad()
def run(x, in_proj_weight, in_proj_bias, conv_weight, conv_bias, out_proj_weight, out_proj_bias):
    return _compiled(x, in_proj_weight, in_proj_bias, conv_weight, conv_bias, out_proj_weight, out_proj_bias)

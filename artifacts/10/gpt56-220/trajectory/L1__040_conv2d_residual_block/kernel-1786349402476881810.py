import torch
import torch.nn.functional as F

torch._dynamo.config.recompile_limit = 32

def _impl(x, conv_in_weight, conv_in_bias, conv_out_weight, conv_out_bias):
    out = F.conv2d(x, conv_in_weight, conv_in_bias, padding=1)
    out = F.conv2d(out, conv_out_weight, conv_out_bias, padding=1)
    return out + x

_compiled = torch.compile(_impl, fullgraph=True, dynamic=False, mode="reduce-overhead")

@torch.no_grad()
def run(
    x: torch.Tensor,
    conv_in_weight: torch.Tensor,
    conv_in_bias: torch.Tensor,
    conv_out_weight: torch.Tensor,
    conv_out_bias: torch.Tensor,
):
    """
    Convolutional residual block:
    1. conv_in: (B, 3, H, W) -> (B, 32, H, W)
    2. conv_out: (B, 32, H, W) -> (B, 3, H, W)
    3. residual add: output + input
    """
    return _compiled(x, conv_in_weight, conv_in_bias, conv_out_weight, conv_out_bias)

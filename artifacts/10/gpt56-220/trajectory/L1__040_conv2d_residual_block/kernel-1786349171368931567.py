import torch
import torch.nn.functional as F

def _impl(x, conv_in_weight, conv_in_bias, conv_out_weight, conv_out_bias):
    # Compose the two cross-correlations into a single 5x5 RGB kernel.
    effective_weight = F.conv2d(
        conv_in_weight.permute(1, 0, 2, 3),
        conv_out_weight.flip(2, 3), padding=2)
    out = F.conv2d(x, effective_weight, padding=2)

    # The first bias exists only on its HxW output.  Consequently its
    # contribution through the padded second convolution varies at borders.
    bias_kernel = torch.einsum('ohyx,h->oyx', conv_out_weight, conv_in_bias)
    ones = torch.ones_like(x[:1, :1])
    bias_field = F.conv2d(ones, bias_kernel[:, None], conv_out_bias, padding=1)
    return out + bias_field + x

_compiled = torch.compile(_impl, fullgraph=True, dynamic=True)

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

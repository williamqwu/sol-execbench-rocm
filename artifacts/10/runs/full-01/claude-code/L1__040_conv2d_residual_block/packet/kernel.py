import torch
import torch.nn.functional as F


@torch.no_grad()
def run(x, conv_in_weight, conv_in_bias, conv_out_weight, conv_out_bias):
    out = F.conv2d(x, conv_in_weight, conv_in_bias, padding=1)
    out = F.conv2d(out, conv_out_weight, conv_out_bias, padding=1)
    return out + x

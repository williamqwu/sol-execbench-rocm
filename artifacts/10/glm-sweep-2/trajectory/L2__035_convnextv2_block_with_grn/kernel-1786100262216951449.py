import torch
import torch.nn.functional as F


def _run(
    x: torch.Tensor,
    dwconv_weight: torch.Tensor,
    dwconv_bias: torch.Tensor,
    layernorm_weight: torch.Tensor,
    layernorm_bias: torch.Tensor,
    pwconv1_weight: torch.Tensor,
    pwconv1_bias: torch.Tensor,
    grn_weight: torch.Tensor,
    grn_bias: torch.Tensor,
    pwconv2_weight: torch.Tensor,
    pwconv2_bias: torch.Tensor,
    eps: float,
    layer_norm_eps: float,
):
    residual = x
    B, C, H, W = x.shape

    out = F.conv2d(x, dwconv_weight, dwconv_bias, padding=3, groups=C)
    out = out.permute(0, 2, 3, 1)
    out = F.layer_norm(out, (C,), layernorm_weight, layernorm_bias, eps=layer_norm_eps)
    out = torch.matmul(out, pwconv1_weight.T) + pwconv1_bias
    out = F.gelu(out)

    global_features = torch.linalg.vector_norm(out, ord=2, dim=(1, 2), keepdim=True)
    norm_features = global_features / (global_features.mean(dim=-1, keepdim=True) + eps)
    out = grn_weight * (out * norm_features) + grn_bias + out

    out = torch.matmul(out, pwconv2_weight.T) + pwconv2_bias
    out = out.permute(0, 3, 1, 2)
    out = residual + out
    return out


_compiled = torch.compile(_run, dynamic=False)


@torch.no_grad()
def run(
    x: torch.Tensor,
    dwconv_weight: torch.Tensor,
    dwconv_bias: torch.Tensor,
    layernorm_weight: torch.Tensor,
    layernorm_bias: torch.Tensor,
    pwconv1_weight: torch.Tensor,
    pwconv1_bias: torch.Tensor,
    grn_weight: torch.Tensor,
    grn_bias: torch.Tensor,
    pwconv2_weight: torch.Tensor,
    pwconv2_bias: torch.Tensor,
    eps: float,
    layer_norm_eps: float,
):
    return _compiled(
        x, dwconv_weight, dwconv_bias, layernorm_weight, layernorm_bias,
        pwconv1_weight, pwconv1_bias, grn_weight, grn_bias,
        pwconv2_weight, pwconv2_bias, eps, layer_norm_eps,
    )

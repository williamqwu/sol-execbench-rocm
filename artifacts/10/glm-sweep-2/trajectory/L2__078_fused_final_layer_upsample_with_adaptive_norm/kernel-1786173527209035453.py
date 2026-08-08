import torch
import torch.nn.functional as F

torch.backends.cudnn.benchmark = True


def _conv3x3_gemm(x, weight, bias, padding=1):
    """3x3 conv via im2col + batched GEMM. Faster than MIOpen for huge-channel, tiny-spatial convs on MI350X."""
    B, C, H, W = x.shape
    out_c = weight.shape[0]
    xp = F.pad(x, (padding, padding, padding, padding))
    cols = F.unfold(xp, (3, 3))                       # (B, C*9, H*W)
    w_flat = weight.reshape(out_c, -1)                 # (out_c, C*9)
    out = torch.matmul(w_flat, cols)                   # (B, out_c, H*W)
    out = out + bias.view(1, -1, 1)
    return out.reshape(B, out_c, H, W)


@torch.no_grad()
def run(
    x: torch.Tensor,
    timestep_emb: torch.Tensor,
    time_emb_mlp_linear_weight: torch.Tensor,
    time_emb_mlp_linear_bias: torch.Tensor,
    resblock_in_norm_weight: torch.Tensor,
    resblock_in_norm_bias: torch.Tensor,
    resblock_in_conv_weight: torch.Tensor,
    resblock_in_conv_bias: torch.Tensor,
    resblock_emb_linear_weight: torch.Tensor,
    resblock_emb_linear_bias: torch.Tensor,
    resblock_out_norm_weight: torch.Tensor,
    resblock_out_norm_bias: torch.Tensor,
    resblock_out_conv_weight: torch.Tensor,
    resblock_out_conv_bias: torch.Tensor,
    resblock_skip_conv_weight: torch.Tensor,
    resblock_skip_conv_bias: torch.Tensor,
    final_norm_weight: torch.Tensor,
    final_norm_bias: torch.Tensor,
    final_conv_weight: torch.Tensor,
    final_conv_bias: torch.Tensor,
    eps: float,
):
    batch_size = x.shape[0]
    seq_len = x.shape[1]
    hidden_size = x.shape[2]
    hidden_channels = resblock_in_conv_weight.shape[0]
    out_channels = final_conv_weight.shape[0]

    token_h = int(seq_len ** 0.5)
    token_w = seq_len // token_h
    while token_h * token_w != seq_len:
        token_h -= 1
        token_w = seq_len // token_h

    x_spatial = x.reshape(batch_size, token_h, token_w, hidden_size)
    x_spatial = x_spatial.permute(0, 3, 1, 2).contiguous()

    silu_emb = F.silu(timestep_emb)
    emb_out = F.linear(silu_emb, resblock_emb_linear_weight, resblock_emb_linear_bias)
    emb_out = emb_out.unsqueeze(-1).unsqueeze(-1)
    scale, shift = torch.chunk(emb_out, 2, dim=1)

    h = F.group_norm(x_spatial, 32, resblock_in_norm_weight, resblock_in_norm_bias, eps)
    h = F.silu(h)
    h = _conv3x3_gemm(h, resblock_in_conv_weight, resblock_in_conv_bias, padding=1)

    h = F.group_norm(h, 32, resblock_out_norm_weight, resblock_out_norm_bias, eps)
    h = h * (1.0 + scale) + shift

    h = F.silu(h)
    h = _conv3x3_gemm(h, resblock_out_conv_weight, resblock_out_conv_bias, padding=1)

    skip = F.conv2d(x_spatial, resblock_skip_conv_weight, resblock_skip_conv_bias)
    h = skip + h

    h = F.group_norm(h, 32, final_norm_weight, final_norm_bias, eps)
    h = F.silu(h)
    output = F.conv2d(h, final_conv_weight, final_conv_bias, padding=1)

    return output

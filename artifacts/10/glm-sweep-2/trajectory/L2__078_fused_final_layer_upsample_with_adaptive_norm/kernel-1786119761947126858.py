import math

import torch
import torch.nn.functional as F


def get_inputs(axes_and_scalars: dict, device: torch.device) -> dict[str, torch.Tensor]:
    batch_size = axes_and_scalars["batch_size"]
    token_h = axes_and_scalars["token_h"]
    token_w = axes_and_scalars["token_w"]
    seq_len = token_h * token_w

    hidden_size = 4096
    emb_channels = 4096
    hidden_channels = 3072
    out_channels = 4
    double_hidden_channels = hidden_channels * 2
    dtype = torch.bfloat16

    g = torch.Generator(device=device)
    g.manual_seed(42)

    def xavier(out_f, in_f):
        return (torch.randn(out_f, in_f, device=device, generator=g) / math.sqrt(in_f)).to(dtype)

    def kaiming_conv(out_c, in_c, kh, kw):
        fan_in = in_c * kh * kw
        return (torch.randn(out_c, in_c, kh, kw, device=device, generator=g) * math.sqrt(2.0 / fan_in)).to(dtype)

    return {
        "x": torch.randn(batch_size, seq_len, hidden_size, device=device, generator=g).to(dtype),
        "timestep_emb": (torch.randn(batch_size, emb_channels, device=device, generator=g) * 0.1).to(dtype),
        "time_emb_mlp_linear_weight": xavier(double_hidden_channels, emb_channels),
        "time_emb_mlp_linear_bias": torch.randn(double_hidden_channels, device=device, generator=g).to(dtype),
        "resblock_in_norm_weight": torch.ones(hidden_size, device=device, dtype=dtype),
        "resblock_in_norm_bias": torch.zeros(hidden_size, device=device, dtype=dtype),
        "resblock_in_conv_weight": kaiming_conv(hidden_channels, hidden_size, 3, 3),
        "resblock_in_conv_bias": torch.randn(hidden_channels, device=device, generator=g).to(dtype),
        "resblock_emb_linear_weight": xavier(double_hidden_channels, emb_channels),
        "resblock_emb_linear_bias": torch.randn(double_hidden_channels, device=device, generator=g).to(dtype),
        "resblock_out_norm_weight": torch.ones(hidden_channels, device=device, dtype=dtype),
        "resblock_out_norm_bias": torch.zeros(hidden_channels, device=device, dtype=dtype),
        "resblock_out_conv_weight": kaiming_conv(hidden_channels, hidden_channels, 3, 3),
        "resblock_out_conv_bias": torch.randn(hidden_channels, device=device, generator=g).to(dtype),
        "resblock_skip_conv_weight": kaiming_conv(hidden_channels, hidden_size, 1, 1),
        "resblock_skip_conv_bias": torch.randn(hidden_channels, device=device, generator=g).to(dtype),
        "final_norm_weight": torch.ones(hidden_channels, device=device, dtype=dtype),
        "final_norm_bias": torch.zeros(hidden_channels, device=device, dtype=dtype),
        "final_conv_weight": kaiming_conv(out_channels, hidden_channels, 3, 3),
        "final_conv_bias": torch.randn(out_channels, device=device, generator=g).to(dtype),
        "eps": 1e-6,
    }


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
    x_spatial = x_spatial.permute(0, 3, 1, 2).contiguous()  # (B, C, H, W)

    # Compute silu(timestep_emb) once; reused by both linear layers.
    silu_emb = F.silu(timestep_emb)
    emb = F.linear(silu_emb, time_emb_mlp_linear_weight, time_emb_mlp_linear_bias)
    emb_out = F.linear(silu_emb, resblock_emb_linear_weight, resblock_emb_linear_bias)
    emb_out = emb_out.unsqueeze(-1).unsqueeze(-1)

    h = F.group_norm(x_spatial, 32, resblock_in_norm_weight, resblock_in_norm_bias, eps)
    h = F.silu(h)
    h = F.conv2d(h, resblock_in_conv_weight, resblock_in_conv_bias, padding=1)

    h = F.group_norm(h, 32, resblock_out_norm_weight, resblock_out_norm_bias, eps)
    scale, shift = torch.chunk(emb_out, 2, dim=1)
    h = h * (1.0 + scale) + shift

    h = F.silu(h)
    h = F.conv2d(h, resblock_out_conv_weight, resblock_out_conv_bias, padding=1)

    skip = F.conv2d(x_spatial, resblock_skip_conv_weight, resblock_skip_conv_bias)
    h = skip + h

    h = F.group_norm(h, 32, final_norm_weight, final_norm_bias, eps)
    h = F.silu(h)
    output = F.conv2d(h, final_conv_weight, final_conv_bias, padding=1)

    return output

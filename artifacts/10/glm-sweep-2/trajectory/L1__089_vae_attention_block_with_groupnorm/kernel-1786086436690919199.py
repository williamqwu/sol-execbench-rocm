import torch
import torch.nn.functional as F


def _norm_scale_shift(x_grouped, mean, var, gnw, gnb, eps):
    batch = mean.shape[0]
    channels = gnw.shape[0]
    height = x_grouped.shape[-2]
    width = x_grouped.shape[-1]
    x_norm = (x_grouped - mean) / torch.sqrt(var + eps)
    x_norm = x_norm.view(batch, channels, height, width)
    x_norm = x_norm * gnw.view(1, channels, 1, 1) + gnb.view(1, channels, 1, 1)
    return x_norm


_compiled_norm = torch.compile(_norm_scale_shift, dynamic=True)


@torch.no_grad()
def run(
    x: torch.Tensor,
    group_norm_weight: torch.Tensor,
    group_norm_bias: torch.Tensor,
    query_weight: torch.Tensor,
    query_bias: torch.Tensor,
    key_weight: torch.Tensor,
    key_bias: torch.Tensor,
    value_weight: torch.Tensor,
    value_bias: torch.Tensor,
    proj_out_weight: torch.Tensor,
    proj_out_bias: torch.Tensor,
    eps: float,
):
    batch, channels, height, width = x.shape
    num_groups = 32

    residual = x

    channels_per_group = channels // num_groups
    x_grouped = x.view(batch, num_groups, channels_per_group, height, width)
    mean = x_grouped.mean(dim=(2, 3, 4), keepdim=True)
    var = x_grouped.var(dim=(2, 3, 4), keepdim=True, unbiased=False)
    x_norm = _compiled_norm(x_grouped, mean, var, group_norm_weight, group_norm_bias, eps)

    seq_len = height * width
    x_seq = x_norm.view(batch, channels, seq_len)
    x_seq = x_seq.permute(0, 2, 1).contiguous()

    qkv_weight = torch.cat([query_weight, key_weight, value_weight], dim=0)
    qkv_bias = torch.cat([query_bias, key_bias, value_bias], dim=0)
    qkv = F.linear(x_seq, qkv_weight, qkv_bias)
    q, k, v = qkv.split(channels, dim=-1)

    scale = channels ** -0.5
    attn_scores = torch.baddbmm(
        torch.empty(batch, seq_len, seq_len, device=x.device, dtype=x.dtype),
        q, k.transpose(1, 2), alpha=scale, beta=0.0,
    )
    attn_weights = F.softmax(attn_scores, dim=-1)
    attn_output = torch.bmm(attn_weights, v)

    attn_output = F.linear(attn_output, proj_out_weight, proj_out_bias)

    attn_output = attn_output.permute(0, 2, 1).contiguous()
    attn_output = attn_output.view(batch, channels, height, width)

    output = residual + attn_output
    return output

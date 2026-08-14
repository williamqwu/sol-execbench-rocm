import torch
import torch.nn.functional as F
@torch.no_grad()
def run(
    x,
    group_norm_weight,
    group_norm_bias,
    query_weight,
    query_bias,
    key_weight,
    key_bias,
    value_weight,
    value_bias,
    proj_out_weight,
    proj_out_bias,
    eps,
):
    batch, channels, height, width = x.shape
    grouped = x.view(batch, 32, channels // 32, height, width)
    mean = grouped.mean(dim=(2, 3, 4), keepdim=True)
    var = grouped.var(dim=(2, 3, 4), keepdim=True, unbiased=False)
    norm = grouped - mean
    norm.div_(torch.sqrt(var + eps))
    norm = norm.view(batch, channels, height, width)
    norm.mul_(group_norm_weight.view(1, channels, 1, 1))
    norm.add_(group_norm_bias.view(1, channels, 1, 1))
    seq = norm.view(batch, channels, height * width).permute(0, 2, 1).contiguous()
    q = F.linear(seq, query_weight, query_bias)
    k = F.linear(seq, key_weight, key_bias)
    v = F.linear(seq, value_weight, value_bias)
    scores = torch.bmm(q, k.transpose(1, 2))
    scores.mul_(channels ** -0.5)
    torch.softmax(scores, dim=-1, out=scores)
    out = torch.bmm(scores, v)
    out = F.linear(out, proj_out_weight, proj_out_bias)
    out = out.permute(0, 2, 1).contiguous().view_as(x)
    out.add_(x)
    return out

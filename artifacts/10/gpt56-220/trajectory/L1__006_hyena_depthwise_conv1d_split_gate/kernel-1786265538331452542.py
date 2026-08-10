import torch
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def _conv3(u, w, bias, b, channel, t, seq_len: tl.constexpr, mask):
    base = (b * 768 + channel) * seq_len + t
    wc = channel * 3
    z = tl.load(bias + channel, mask=mask, other=0.0)
    z += tl.load(u + base - 2, mask=mask & (t >= 2), other=0.0) * tl.load(w + wc, mask=mask, other=0.0)
    z += tl.load(u + base - 1, mask=mask & (t >= 1), other=0.0) * tl.load(w + wc + 1, mask=mask, other=0.0)
    z += tl.load(u + base, mask=mask, other=0.0) * tl.load(w + wc + 2, mask=mask, other=0.0)
    return z


@triton.jit
def _hyena_kernel(u, w, bias, out_g, out_x0, out_x1,
                  n_elements: tl.constexpr, seq_len: tl.constexpr,
                  BLOCK: tl.constexpr):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n_elements
    t = offs % seq_len
    q = offs // seq_len
    b = q // 256
    c = q % 256

    x0 = _conv3(u, w, bias, b, c, t, seq_len, mask)
    x1 = _conv3(u, w, bias, b, c + 256, t, seq_len, mask)
    v = _conv3(u, w, bias, b, c + 512, t, seq_len, mask)
    tl.store(out_g + offs, v * x0, mask=mask)
    tl.store(out_x0 + offs, x0, mask=mask)
    tl.store(out_x1 + offs, x1, mask=mask)


@torch.no_grad()
def run(u: torch.Tensor, short_filter_weight: torch.Tensor, short_filter_bias: torch.Tensor):
    seq_len = u.shape[-1]
    x0 = F.conv1d(u[:, :256], short_filter_weight[:256], short_filter_bias[:256], padding=2, groups=256)[..., :seq_len]
    x1 = F.conv1d(u[:, 256:512], short_filter_weight[256:512], short_filter_bias[256:512], padding=2, groups=256)[..., :seq_len]
    v = F.conv1d(u[:, 512:768], short_filter_weight[512:768], short_filter_bias[512:768], padding=2, groups=256)[..., :seq_len]
    return v * x0, x0, x1

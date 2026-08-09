import torch
import triton
import triton.language as tl


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

    def conv(channel):
        base = (b * 768 + channel) * seq_len + t
        wc = channel * 3
        z = tl.load(bias + channel, mask=mask, other=0.0)
        m2 = mask & (t >= 2)
        m1 = mask & (t >= 1)
        z += tl.load(u + base - 2, mask=m2, other=0.0) * tl.load(w + wc, mask=mask, other=0.0)
        z += tl.load(u + base - 1, mask=m1, other=0.0) * tl.load(w + wc + 1, mask=mask, other=0.0)
        z += tl.load(u + base, mask=mask, other=0.0) * tl.load(w + wc + 2, mask=mask, other=0.0)
        return z

    x0 = conv(c)
    x1 = conv(c + 256)
    v = conv(c + 512)
    tl.store(out_g + offs, v * x0, mask=mask)
    tl.store(out_x0 + offs, x0, mask=mask)
    tl.store(out_x1 + offs, x1, mask=mask)


@torch.no_grad()
def run(u: torch.Tensor, short_filter_weight: torch.Tensor, short_filter_bias: torch.Tensor):
    batch, _, seq_len = u.shape
    shape = (batch, 256, seq_len)
    out_g = torch.empty(shape, device=u.device, dtype=u.dtype)
    out_x0 = torch.empty(shape, device=u.device, dtype=u.dtype)
    out_x1 = torch.empty(shape, device=u.device, dtype=u.dtype)
    n = batch * 256 * seq_len
    _hyena_kernel[(triton.cdiv(n, 256),)](
        u, short_filter_weight, short_filter_bias, out_g, out_x0, out_x1,
        n_elements=n, seq_len=seq_len, BLOCK=256, num_warps=4)
    return out_g, out_x0, out_x1

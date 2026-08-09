import torch
import triton
import triton.language as tl


@triton.jit
def _conv3(u, w, bias, b, channel, t, seq_len: tl.constexpr, mask):
    base = (b * 768 + channel) * seq_len + t
    wc = channel * 3
    z = tl.load(bias + channel)
    z += tl.load(u + base - 2, mask=mask & (t >= 2), other=0.0) * tl.load(w + wc)
    z += tl.load(u + base - 1, mask=mask & (t >= 1), other=0.0) * tl.load(w + wc + 1)
    z += tl.load(u + base, mask=mask, other=0.0) * tl.load(w + wc + 2)
    return z


@triton.jit
def _hyena_kernel(u, w, bias, out_g, out_x0, out_x1,
                  seq_len: tl.constexpr,
                  BLOCK: tl.constexpr):
    t = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    q = tl.program_id(1)
    mask = t < seq_len
    b = q // 256
    c = q % 256

    x0 = _conv3(u, w, bias, b, c, t, seq_len, mask)
    x1 = _conv3(u, w, bias, b, c + 256, t, seq_len, mask)
    v = _conv3(u, w, bias, b, c + 512, t, seq_len, mask)
    offs = q * seq_len + t
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
    _hyena_kernel[(triton.cdiv(seq_len, 128), batch * 256)](
        u, short_filter_weight, short_filter_bias, out_g, out_x0, out_x1,
        seq_len=seq_len, BLOCK=128, num_warps=2)
    return out_g, out_x0, out_x1

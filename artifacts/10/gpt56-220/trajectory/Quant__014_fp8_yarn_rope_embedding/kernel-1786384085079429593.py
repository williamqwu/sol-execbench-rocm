import math
import torch
import triton
import triton.language as tl


@triton.jit
def _rope_kernel(pos, cos_out, sin_out, n: tl.constexpr, factor: tl.constexpr,
                 BM: tl.constexpr):
    rows = tl.program_id(0) * BM + tl.arange(0, BM)
    cols = tl.arange(0, 32)
    valid = rows[:, None] < n
    p = tl.load(pos + rows[:, None], mask=valid, other=0).to(tl.float32)
    c = cols[None, :].to(tl.float32)
    extra = 1.0 / tl.exp2((c / 32.0) * 13.287712379549449)
    ramp = tl.maximum(0.0, tl.minimum(1.0, (c - 10.0) / 13.0))
    inv = extra * (1.0 - ramp) + (extra / 40.0) * ramp
    f = p * inv
    mx = tl.max(tl.abs(f), axis=1)[:, None]
    scale = tl.maximum(mx / 448.0, 1.0e-12)
    q = tl.maximum(-448.0, tl.minimum(448.0, f / scale)).to(tl.float8e4nv)
    x = q.to(tl.float32) * scale
    offsets = rows[:, None] * 64 + cols[None, :]
    tl.store(cos_out + offsets, tl.cos(x) * factor, mask=valid)
    tl.store(cos_out + offsets + 32, tl.cos(x) * factor, mask=valid)
    tl.store(sin_out + offsets, tl.sin(x) * factor, mask=valid)
    tl.store(sin_out + offsets + 32, tl.sin(x) * factor, mask=valid)


@torch.compile(fullgraph=True, dynamic=True)
def _compiled(position_ids: torch.Tensor, factor: float):
    device = position_ids.device
    # YaRN correction range is [10, 23] for this fixed configuration.
    i = torch.arange(0, 64, 2, dtype=torch.float32, device=device)
    extra = 1.0 / (10000 ** (i / 64))
    inter = 1.0 / (40 * 10000 ** (i / 64))
    ramp = ((torch.arange(32, dtype=torch.float32, device=device) - 10) / 13).clamp(0, 1)
    mask = 1.0 - ramp
    inv = inter * (1.0 - mask) + extra * mask
    freqs = position_ids.float().unsqueeze(1) * inv.unsqueeze(0)
    scale = freqs.abs().amax(dim=1, keepdim=True).div(448.0).clamp(min=1e-12)
    q = (freqs / scale).clamp(-448.0, 448.0).to(torch.float8_e4m3fn)
    deq = q.float() * scale
    emb = torch.cat((deq, deq), dim=1)
    return (emb.cos() * factor).to(torch.bfloat16), (emb.sin() * factor).to(torch.bfloat16)


@torch.no_grad()
def run(position_ids: torch.Tensor, mscale: float, mscale_all_dim: float):
    a = 0.1 * math.log(40.0)
    factor = (a * float(mscale) + 1.0) / (a * float(mscale_all_dim) + 1.0)
    n = position_ids.numel()
    co = torch.empty((n, 64), device=position_ids.device, dtype=torch.bfloat16)
    si = torch.empty_like(co)
    _rope_kernel[(triton.cdiv(n, 8),)](position_ids, co, si, n=n, factor=factor, BM=8,
                                      num_warps=4)
    return co, si

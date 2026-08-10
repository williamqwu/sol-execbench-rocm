import math
import torch
import triton
import triton.language as tl


@triton.jit
def _rope_kernel(pos, output, n: tl.constexpr, factor: tl.constexpr,
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
    cos_x = tl.cos(x) * factor
    sin_x = tl.sin(x) * factor
    tl.store(output + offsets, cos_x, mask=valid)
    tl.store(output + offsets + 32, cos_x, mask=valid)
    sin_base = n * 64
    tl.store(output + sin_base + offsets, sin_x, mask=valid)
    tl.store(output + sin_base + offsets + 32, sin_x, mask=valid)


@torch.no_grad()
def run(position_ids: torch.Tensor, mscale: float, mscale_all_dim: float):
    a = 0.1 * math.log(40.0)
    factor = (a * float(mscale) + 1.0) / (a * float(mscale_all_dim) + 1.0)
    n = position_ids.numel()
    outputs = torch.empty((2, n, 64), device=position_ids.device,
                          dtype=torch.bfloat16)
    block_rows = 16 if n <= 32768 else 32
    _rope_kernel[(triton.cdiv(n, block_rows),)](position_ids, outputs, n=n,
                                      factor=factor, BM=block_rows,
                                      num_warps=4, num_stages=1)
    return outputs[0], outputs[1]

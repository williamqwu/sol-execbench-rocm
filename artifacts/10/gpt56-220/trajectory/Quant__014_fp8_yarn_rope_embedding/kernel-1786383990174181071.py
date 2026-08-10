import math
import torch


@torch.no_grad()
def run(position_ids: torch.Tensor, mscale: float, mscale_all_dim: float):
    device = position_ids.device
    # YaRN correction range is [8, 21] for this fixed configuration.
    i = torch.arange(0, 64, 2, dtype=torch.float32, device=device)
    extra = 1.0 / (10000 ** (i / 64))
    inter = 1.0 / (40 * 10000 ** (i / 64))
    ramp = ((torch.arange(32, dtype=torch.float32, device=device) - 8) / 13).clamp(0, 1)
    mask = 1.0 - ramp
    inv = inter * (1.0 - mask) + extra * mask
    freqs = position_ids.float().unsqueeze(1) * inv.unsqueeze(0)
    scale = freqs.abs().amax(dim=1, keepdim=True).div(448.0).clamp(min=1e-12)
    q = (freqs / scale).clamp(-448.0, 448.0).to(torch.float8_e4m3fn)
    deq = q.float() * scale
    emb = torch.cat((deq, deq), dim=1)
    a = 0.1 * math.log(40.0)
    factor = (a * float(mscale) + 1.0) / (a * float(mscale_all_dim) + 1.0)
    return (emb.cos() * factor).to(torch.bfloat16), (emb.sin() * factor).to(torch.bfloat16)

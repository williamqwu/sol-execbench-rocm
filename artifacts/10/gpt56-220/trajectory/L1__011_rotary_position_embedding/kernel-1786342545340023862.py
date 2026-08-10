import torch
import triton
import triton.language as tl


@triton.jit
def _rope_kernel(position_ids, inv_freq, output, n_unique: tl.constexpr,
                 scaling: tl.constexpr, BLOCK: tl.constexpr):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < n_unique
    d = offsets % 64
    token = offsets // 64
    pos = tl.load(position_ids + token, mask=mask, other=0).to(tl.float32)
    freq = tl.load(inv_freq + d, mask=mask, other=0.0)
    angle = pos * freq
    cos = tl.cos(angle) * scaling
    sin = tl.sin(angle) * scaling
    # Output is [token, 128, 2]. The second half repeats the first.
    first = token * 256 + d * 2
    second = first + 128
    tl.store(output + first, cos, mask=mask)
    tl.store(output + first + 1, sin, mask=mask)
    tl.store(output + second, cos, mask=mask)
    tl.store(output + second + 1, sin, mask=mask)


@torch.no_grad()
def run(position_ids: torch.Tensor, inv_freq: torch.Tensor,
        attention_scaling: float) -> torch.Tensor:
    batch, seq_len = position_ids.shape
    output = torch.empty((batch, seq_len, 128, 2), device=position_ids.device,
                         dtype=torch.bfloat16)
    n_unique = batch * seq_len * 64
    _rope_kernel[(triton.cdiv(n_unique, 256),)](
        position_ids, inv_freq, output, n_unique,
        scaling=attention_scaling, BLOCK=256, num_warps=2)
    return output

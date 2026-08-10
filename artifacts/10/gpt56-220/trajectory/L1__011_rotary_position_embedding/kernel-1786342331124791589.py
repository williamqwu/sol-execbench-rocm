import torch
import triton
import triton.language as tl


@triton.jit
def _rope_kernel(position_ids, inv_freq, output, n_elements: tl.constexpr,
                 scaling: tl.constexpr, BLOCK: tl.constexpr):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < n_elements
    d = offsets % 128
    token = offsets // 128
    pos = tl.load(position_ids + token, mask=mask, other=0).to(tl.float32)
    freq = tl.load(inv_freq + (d % 64), mask=mask, other=0.0)
    angle = pos * freq
    base = offsets * 2
    tl.store(output + base, tl.cos(angle) * scaling, mask=mask)
    tl.store(output + base + 1, tl.sin(angle) * scaling, mask=mask)


@torch.no_grad()
def run(position_ids: torch.Tensor, inv_freq: torch.Tensor,
        attention_scaling: float) -> torch.Tensor:
    batch, seq_len = position_ids.shape
    output = torch.empty((batch, seq_len, 128, 2), device=position_ids.device,
                         dtype=torch.bfloat16)
    n_elements = batch * seq_len * 128
    _rope_kernel[(triton.cdiv(n_elements, 256),)](
        position_ids, inv_freq, output, n_elements,
        scaling=attention_scaling, BLOCK=256)
    return output

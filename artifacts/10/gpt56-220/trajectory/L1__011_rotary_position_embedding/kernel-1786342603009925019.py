import torch
import triton
import triton.language as tl


@triton.jit
def _rope_kernel(position_ids, inv_freq, output,
                 scaling: tl.constexpr):
    token = tl.program_id(0)
    d = tl.arange(0, 64)
    pos = tl.load(position_ids + token).to(tl.float32)
    freq = tl.load(inv_freq + d)
    angle = pos * freq
    cos = tl.cos(angle) * scaling
    sin = tl.sin(angle) * scaling
    # Output is [token, 128, 2]. The second half repeats the first.
    first = token * 256 + d * 2
    second = first + 128
    tl.store(output + first, cos)
    tl.store(output + first + 1, sin)
    tl.store(output + second, cos)
    tl.store(output + second + 1, sin)


@torch.no_grad()
def run(position_ids: torch.Tensor, inv_freq: torch.Tensor,
        attention_scaling: float) -> torch.Tensor:
    batch, seq_len = position_ids.shape
    output = torch.empty((batch, seq_len, 128, 2), device=position_ids.device,
                         dtype=torch.bfloat16)
    n_tokens = batch * seq_len
    _rope_kernel[(n_tokens,)](
        position_ids, inv_freq, output,
        scaling=attention_scaling, num_warps=1)
    return output

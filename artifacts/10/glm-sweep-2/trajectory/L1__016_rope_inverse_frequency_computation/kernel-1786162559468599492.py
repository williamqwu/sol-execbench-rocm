import torch
import triton
import triton.language as tl

@triton.jit
def _inv_freq_kernel(out_ptr, rope_theta, HEAD_DIM: tl.constexpr):
    pid = tl.program_id(0)
    offs = tl.arange(0, HEAD_DIM // 2)
    exps = (2.0 * offs) / float(HEAD_DIM)
    powers = tl.exp(exps * tl.log(rope_theta))
    inv = 1.0 / powers
    tl.store(out_ptr + offs, inv)

@torch.no_grad()
def run(rope_theta: float) -> torch.Tensor:
    head_dim = 128
    out = torch.empty(head_dim // 2, dtype=torch.float32, device='cuda')
    _inv_freq_kernel[(1,)](out, float(rope_theta), HEAD_DIM=head_dim)
    return out

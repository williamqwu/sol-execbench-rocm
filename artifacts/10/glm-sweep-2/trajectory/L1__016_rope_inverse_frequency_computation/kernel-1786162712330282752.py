import torch
import triton
import triton.language as tl
import math

@triton.jit
def _inv_freq_kernel(out_ptr, log_theta, HEAD_DIM: tl.constexpr):
    offs = tl.arange(0, HEAD_DIM // 2)
    lt = log_theta.to(tl.float64)
    exps = (2.0 * offs.to(tl.float64)) / float(HEAD_DIM)
    inv = (1.0 / tl.exp(exps * lt)).to(tl.float32)
    tl.store(out_ptr + offs, inv)

@torch.no_grad()
def run(rope_theta: float) -> torch.Tensor:
    head_dim = 128
    out = torch.empty(head_dim // 2, dtype=torch.float32, device='cuda')
    _inv_freq_kernel[(1,)](out, math.log(float(rope_theta)), HEAD_DIM=head_dim, num_warps=1)
    return out

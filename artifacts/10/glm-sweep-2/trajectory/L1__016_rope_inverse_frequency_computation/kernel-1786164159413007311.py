import torch
import triton
import triton.language as tl

@triton.jit
def _inv_freq_kernel(out_ptr, rope_theta, HEAD_DIM: tl.constexpr):
    offs = tl.arange(0, HEAD_DIM // 2)
    theta = rope_theta.to(tl.float64)
    exps = (2.0 * offs.to(tl.float64)) / float(HEAD_DIM)
    powers = tl.exp(exps * tl.log(theta))
    inv = (1.0 / powers).to(tl.float32)
    tl.store(out_ptr + offs, inv)

@torch.no_grad()
def run(rope_theta: float) -> torch.Tensor:
    out = torch.empty(64, dtype=torch.float32, device='cuda')
    _inv_freq_kernel[(1,)](out, float(rope_theta), HEAD_DIM=128)
    return out

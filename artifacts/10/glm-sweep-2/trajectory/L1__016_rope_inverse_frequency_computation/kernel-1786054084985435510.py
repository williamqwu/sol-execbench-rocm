import torch
import triton
import triton.language as tl

HEAD_DIM = 128
HALF = HEAD_DIM // 2  # 64

@triton.jit
def _inv_freq_kernel(out_ptr, rope_theta, HEAD_DIM: tl.constexpr, HALF: tl.constexpr):
    pid = tl.program_id(0)
    idx = pid.to(tl.float32)
    exp = -(2.0 * idx) / HEAD_DIM
    val = tl.exp(exp * tl.log(rope_theta))
    tl.store(out_ptr + pid, val)

@torch.no_grad()
def run(rope_theta: float) -> torch.Tensor:
    out = torch.empty(HALF, dtype=torch.float32, device='cuda')
    _inv_freq_kernel[(HALF,)](out, float(rope_theta), HEAD_DIM=HEAD_DIM, HALF=HALF)
    return out

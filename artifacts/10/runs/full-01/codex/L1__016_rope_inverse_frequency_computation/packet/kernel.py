from functools import partial

import torch
import triton
import triton.language as tl


@triton.jit
def _rope_inv_freq_kernel(output, rope_theta):
    i = tl.arange(0, 64)
    exponent = i.to(tl.float32) * (1.0 / 64.0)
    value = tl.exp2(-exponent * tl.log2(rope_theta))
    tl.store(output + i, value)


_launch_rope_inv_freq = partial(_rope_inv_freq_kernel[(1,)], num_warps=1)


def run(rope_theta: float) -> torch.Tensor:
    output = torch.empty(64, device="cuda:0", dtype=torch.float32)
    _launch_rope_inv_freq(output, rope_theta)
    return output

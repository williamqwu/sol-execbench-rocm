import torch
import triton
import triton.language as tl

@triton.jit
def _inv_freq_kernel(out_ptr, theta_ptr, HEAD_DIM: tl.constexpr):
    offs = tl.arange(0, HEAD_DIM // 2)
    theta = tl.load(theta_ptr).to(tl.float64)
    exps = (2.0 * offs.to(tl.float64)) / float(HEAD_DIM)
    powers = tl.exp(exps * tl.log(theta))
    inv = (1.0 / powers).to(tl.float32)
    tl.store(out_ptr + offs, inv)

# Static buffers for CUDA graph
_static_out = torch.empty(64, dtype=torch.float32, device='cuda')
_static_theta = torch.empty(1, dtype=torch.float32, device='cuda')

# Warm up the kernel
_inv_freq_kernel[(1,)](_static_out, _static_theta, HEAD_DIM=128)
torch.cuda.synchronize()

# Capture CUDA graph
_graph = torch.cuda.CUDAGraph()
with torch.cuda.graph(_graph):
    _inv_freq_kernel[(1,)](_static_out, _static_theta, HEAD_DIM=128)

@torch.no_grad()
def run(rope_theta: float) -> torch.Tensor:
    _static_theta.fill_(float(rope_theta))
    _graph.replay()
    return _static_out.clone()

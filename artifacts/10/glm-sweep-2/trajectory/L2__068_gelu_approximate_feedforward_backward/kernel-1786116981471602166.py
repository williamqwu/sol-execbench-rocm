import os

# Build the HIP elementwise kernel for gfx950 only (fast compile, cached after first run).
os.environ.setdefault("PYTORCH_ROCM_ARCH", "gfx950")

import torch
from torch.utils.cpp_extension import load_inline

_HIP_SRC = r"""
#include <torch/extension.h>
#include <hip/hip_runtime.h>
#include <c10/cuda/CUDAStream.h>

// Fused GELU-approximate backward elementwise, bitwise-identical to the
// PyTorch eager reference. -ffp-contract=off disables FMA contraction so each
// multiply/add rounds separately (matching eager's mul+add), while keeping
// values in registers (unlike volatile, which forces memory round-trips).
//
//   gelu_grad_part1 = 0.5 * (1.0 + tanh_inner)
//   sech_squared     = 1.0 - (tanh_inner * tanh_inner)
//   d_inner_dx       = 0.7978845608028654 * (1.0 + ((0.134145 * linear1_out) * linear1_out))
//   gelu_grad_part2  = ((0.5 * linear1_out) * sech_squared) * d_inner_dx
//   grad_linear1_out = grad_gelu_out * (gelu_grad_part1 + gelu_grad_part2)
__global__ void gelu_bwd_kernel(
    const float* __restrict__ ggo,
    const float* __restrict__ ti,
    const float* __restrict__ l1o,
    float* __restrict__ out,
    int64_t n) {
  int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < n) {
    float g = ggo[idx];
    float t = ti[idx];
    float x = l1o[idx];
    float p1 = 0.5f * (1.0f + t);
    float t2 = t * t;
    float sech = 1.0f - t2;
    float kx = 0.134145f * x;
    float kxx = kx * x;
    float inner = 1.0f + kxx;
    float didx = 0.7978845608028654f * inner;
    float p2a = 0.5f * x;
    float p2b = p2a * sech;
    float p2 = p2b * didx;
    float gg = p1 + p2;
    out[idx] = g * gg;
  }
}

torch::Tensor gelu_bwd(torch::Tensor ggo, torch::Tensor ti, torch::Tensor l1o) {
  int64_t n = ggo.numel();
  auto out = torch::empty_like(ggo);
  int64_t block = 256;
  int64_t grid = (n + block - 1) / block;
  gelu_bwd_kernel<<<grid, block, 0, c10::cuda::getCurrentCUDAStream()>>>(
      ggo.data_ptr<float>(), ti.data_ptr<float>(), l1o.data_ptr<float>(),
      out.data_ptr<float>(), n);
  return out;
}
"""

_mod = load_inline(
    name="gelu_bwd_fc_mod",
    cpp_sources="torch::Tensor gelu_bwd(torch::Tensor ggo, torch::Tensor ti, torch::Tensor l1o);",
    cuda_sources=_HIP_SRC,
    functions=["gelu_bwd"],
    verbose=False,
    extra_cuda_cflags=["-ffp-contract=off"],
)
_gelu_bwd = _mod.gelu_bwd


@torch.no_grad()
def run(
    grad_output: torch.Tensor,
    hidden_states: torch.Tensor,
    weight1: torch.Tensor,
    weight2: torch.Tensor,
    linear1_out: torch.Tensor,
    tanh_inner: torch.Tensor,
    gelu_out: torch.Tensor,
):
    grad_gelu_out = grad_output.matmul(weight2)
    grad_output_2d = grad_output.reshape(-1, grad_output.shape[-1])
    gelu_out_2d = gelu_out.reshape(-1, gelu_out.shape[-1])
    grad_weight2 = grad_output_2d.t().matmul(gelu_out_2d)

    grad_linear1_out = _gelu_bwd(grad_gelu_out, tanh_inner, linear1_out)

    grad_hidden_states = grad_linear1_out.matmul(weight1)
    hidden_states_2d = hidden_states.reshape(-1, hidden_states.shape[-1])
    grad_linear1_out_2d = grad_linear1_out.reshape(-1, grad_linear1_out.shape[-1])
    grad_weight1 = grad_linear1_out_2d.t().matmul(hidden_states_2d)

    return grad_hidden_states, grad_weight1, grad_weight2

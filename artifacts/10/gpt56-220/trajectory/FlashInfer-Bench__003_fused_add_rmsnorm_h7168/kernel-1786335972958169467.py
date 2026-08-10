import torch
from torch.utils.cpp_extension import load_inline


_CPP = r''' 
#include <torch/extension.h>
torch::Tensor add_rmsnorm_hip(torch::Tensor hidden, torch::Tensor residual,
                              torch::Tensor weight);
'''

_HIP = r'''
#include <torch/extension.h>
#include <hip/hip_runtime.h>
#include <hip/hip_bfloat16.h>
#include <c10/cuda/CUDAStream.h>

__global__ void add_rmsnorm_kernel(const __hip_bfloat16* hidden,
                                   const __hip_bfloat16* residual,
                                   const __hip_bfloat16* weight,
                                   __hip_bfloat16* output) {
  constexpr int N = 7168;
  constexpr int ITEMS = 28;
  const int tid = threadIdx.x;
  const int base = blockIdx.x * N;
  float vals[ITEMS];
  float sum = 0.0f;
  #pragma unroll
  for (int i = 0; i < ITEMS; ++i) {
    int col = tid + i * 256;
    float x = __bfloat162float(hidden[base + col]) +
              __bfloat162float(residual[base + col]);
    vals[i] = x;
    sum = fmaf(x, x, sum);
  }
  #pragma unroll
  for (int d = 32; d > 0; d >>= 1) sum += __shfl_down(sum, d, 64);
  __shared__ float wave_sums[4];
  if ((tid & 63) == 0) wave_sums[tid >> 6] = sum;
  __syncthreads();
  if (tid < 64) {
    sum = tid < 4 ? wave_sums[tid] : 0.0f;
    #pragma unroll
    for (int d = 32; d > 0; d >>= 1) sum += __shfl_down(sum, d, 64);
    if (tid == 0) wave_sums[0] = rsqrtf(sum * (1.0f / N) + 1.0e-6f);
  }
  __syncthreads();
  float inv = wave_sums[0];
  #pragma unroll
  for (int i = 0; i < ITEMS; ++i) {
    int col = tid + i * 256;
    float y = vals[i] * inv * __bfloat162float(weight[col]);
    output[base + col] = __float2bfloat16_rn(y);
  }
}

torch::Tensor add_rmsnorm_hip(torch::Tensor hidden, torch::Tensor residual,
                              torch::Tensor weight) {
  auto output = torch::empty_like(hidden);
  add_rmsnorm_kernel<<<hidden.size(0), 256, 0,
      c10::cuda::getCurrentCUDAStream().stream()>>>(
      reinterpret_cast<const __hip_bfloat16*>(hidden.data_ptr()),
      reinterpret_cast<const __hip_bfloat16*>(residual.data_ptr()),
      reinterpret_cast<const __hip_bfloat16*>(weight.data_ptr()),
      reinterpret_cast<__hip_bfloat16*>(output.data_ptr()));
  return output;
}
'''

_ext = load_inline(
    name="add_rmsnorm_h7168_ext",
    cpp_sources=_CPP,
    cuda_sources=_HIP,
    functions=["add_rmsnorm_hip"],
    extra_cuda_cflags=["-O3", "-ffast-math"],
    verbose=False,
)


@torch.no_grad()
def run(hidden_states, residual, weight):
    assert hidden_states.shape[1] == 7168
    return _ext.add_rmsnorm_hip(hidden_states, residual, weight)

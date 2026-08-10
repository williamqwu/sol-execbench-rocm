import torch
from torch.utils.cpp_extension import load_inline


_CPP = r"""
#include <torch/extension.h>

torch::Tensor rope_inv_freq(double theta);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("run", &rope_inv_freq);
}
"""

_HIP = r"""
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <hip/hip_runtime.h>

__global__ void rope_kernel(float* out, float theta) {
  const int i = threadIdx.x;
  out[i] = powf(theta, -float(i) * (1.0f / 64.0f));
}

torch::Tensor rope_inv_freq(double theta) {
  auto out = torch::empty({64}, torch::TensorOptions()
      .dtype(torch::kFloat32).device(torch::kCUDA));
  hipLaunchKernelGGL(rope_kernel, dim3(1), dim3(64), 0,
                     at::cuda::getCurrentCUDAStream(),
                     out.data_ptr<float>(), float(theta));
  return out;
}
"""

_ext = load_inline(
    name="rope_inv_freq_016",
    cpp_sources=_CPP,
    cuda_sources=_HIP,
    functions=None,
    extra_cuda_cflags=["-O3"],
    verbose=False,
)


@torch.no_grad()
def run(rope_theta: float) -> torch.Tensor:
    return _ext.run(float(rope_theta))

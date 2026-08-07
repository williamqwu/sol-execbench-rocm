import os
import tempfile

import torch
import torch.nn.functional as F
from torch.utils.cpp_extension import load

_CUDA_SRC = r"""
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>

template <typename scalar_t>
__global__ void silu_kernel(const scalar_t* __restrict__ x, scalar_t* __restrict__ out, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    float v = static_cast<float>(x[i]);
    float s = 1.0f / (1.0f + expf(-v));
    out[i] = static_cast<scalar_t>(v * s);
  }
}

at::Tensor silu_fused(at::Tensor x) {
  auto out = at::empty_like(x);
  int n = x.numel();
  int threads = 256;
  int blocks = (n + threads - 1) / threads;
  AT_DISPATCH_FLOATING_TYPES(x.scalar_type(), "silu_fused", ([&] {
    silu_kernel<scalar_t><<<blocks, threads, 0, at::cuda::getCurrentCUDAStream()>>>(
      x.data_ptr<scalar_t>(), out.data_ptr<scalar_t>(), n);
  }));
  return out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("silu_fused", &silu_fused);
}
"""

_BUILD_DIR = os.path.join(tempfile.gettempdir(), "resblock_fused_build")
os.makedirs(_BUILD_DIR, exist_ok=True)
_SRC_PATH = os.path.join(_BUILD_DIR, "fused_ops.cu")
with open(_SRC_PATH, "w") as _f:
    _f.write(_CUDA_SRC)

_EXT = load(name="resblock_fused", sources=[_SRC_PATH], build_directory=_BUILD_DIR, verbose=False)


@torch.no_grad()
def run(
    x, time_emb,
    norm1_weight, norm1_bias,
    conv1_weight, conv1_bias,
    time_emb_proj_weight, time_emb_proj_bias,
    norm2_weight, norm2_bias,
    conv2_weight, conv2_bias,
    norm_eps,
):
    residual = x

    h = F.group_norm(x, num_groups=32, weight=norm1_weight, bias=norm1_bias, eps=norm_eps)
    h = _EXT.silu_fused(h)
    h = F.conv2d(h, conv1_weight, conv1_bias, stride=1, padding=1)

    t = time_emb * torch.sigmoid(time_emb)
    t = F.linear(t, time_emb_proj_weight, time_emb_proj_bias)
    h = h + t[:, :, None, None]

    h = F.group_norm(h, num_groups=32, weight=norm2_weight, bias=norm2_bias, eps=norm_eps)
    h = _EXT.silu_fused(h)
    h = F.conv2d(h, conv2_weight, conv2_bias, stride=1, padding=1)

    output = h + residual
    return output

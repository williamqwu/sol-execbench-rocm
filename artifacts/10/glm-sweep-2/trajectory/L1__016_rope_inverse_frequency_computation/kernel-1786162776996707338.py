import torch
from torch.utils.cpp_extension import load_inline

_source = r"""
#include <torch/extension.h>
#include <hip/hip_runtime.h>

__global__ void inv_freq_kernel(float* out, double rope_theta) {
    int tid = threadIdx.x;
    if (tid < 64) {
        double exps = (2.0 * (double)tid) / 128.0;
        double powers = exp(exps * log(rope_theta));
        out[tid] = (float)(1.0 / powers);
    }
}

torch::Tensor inv_freq(double rope_theta) {
    auto out = torch::empty({64}, torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA));
    inv_freq_kernel<<<1, 64>>>(
        out.data_ptr<float>(), rope_theta
    );
    return out;
}
"""

_mod = load_inline(
    name="inv_freq_mod",
    cpp_sources="torch::Tensor inv_freq(double rope_theta);",
    cuda_sources=_source,
    functions=["inv_freq"],
    verbose=False,
)

@torch.no_grad()
def run(rope_theta: float) -> torch.Tensor:
    return _mod.inv_freq(float(rope_theta))

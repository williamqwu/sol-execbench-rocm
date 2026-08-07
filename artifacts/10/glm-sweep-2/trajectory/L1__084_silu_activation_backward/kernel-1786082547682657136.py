import torch
from torch.utils.cpp_extension import load_inline

_SRC = r"""
#include <hip/hip_runtime.h>
#include <torch/extension.h>

__global__ void silu_bw_kernel(const float* __restrict__ g,
                                const float* __restrict__ x,
                                const float* __restrict__ s,
                                float* __restrict__ out,
                                int64_t n) {
    int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    int64_t base = idx * 4;
    if (base + 3 < n) {
        float4 gv = *reinterpret_cast<const float4*>(g + base);
        float4 xv = *reinterpret_cast<const float4*>(x + base);
        float4 sv = *reinterpret_cast<const float4*>(s + base);
        float4 ov;
        ov.x = gv.x * sv.x * (1.0f + xv.x * (1.0f - sv.x));
        ov.y = gv.y * sv.y * (1.0f + xv.y * (1.0f - sv.y));
        ov.z = gv.z * sv.z * (1.0f + xv.z * (1.0f - sv.z));
        ov.w = gv.w * sv.w * (1.0f + xv.w * (1.0f - sv.w));
        *reinterpret_cast<float4*>(out + base) = ov;
    } else {
        for (int64_t i = base; i < n && i < base + 4; i++) {
            out[i] = g[i] * s[i] * (1.0f + x[i] * (1.0f - s[i]));
        }
    }
}

void silu_bw(torch::Tensor g, torch::Tensor x, torch::Tensor s, torch::Tensor out) {
    int64_t n = g.numel();
    const int threads = 256;
    int64_t elements_per_block = threads * 4;
    int64_t blocks = (n + elements_per_block - 1) / elements_per_block;
    silu_bw_kernel<<<(unsigned int)blocks, threads>>>(
        g.data_ptr<float>(), x.data_ptr<float>(), s.data_ptr<float>(),
        out.data_ptr<float>(), n);
}
"""

_MOD = load_inline(
    name="silu_bw_mod_v1",
    cpp_sources=["void silu_bw(torch::Tensor g, torch::Tensor x, torch::Tensor s, torch::Tensor out);"],
    cuda_sources=[_SRC],
    functions=["silu_bw"],
    verbose=False,
)


@torch.no_grad()
def run(grad_output: torch.Tensor, x: torch.Tensor, sigmoid_x: torch.Tensor) -> torch.Tensor:
    grad_input = torch.empty_like(grad_output)
    _MOD.silu_bw(grad_output, x, sigmoid_x, grad_input)
    return grad_input

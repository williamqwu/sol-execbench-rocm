import torch
from torch.utils.cpp_extension import load_inline

HIP_SOURCE = r"""
#include <hip/hip_runtime.h>
#include <hip/hip_bf16.h>
#include <torch/extension.h>

template <int H, int ELEMS_PER_THREAD>
__global__ void rmsnorm_kernel(
    const __hip_bfloat16* __restrict__ x,
    const __hip_bfloat16* __restrict__ w,
    __hip_bfloat16* __restrict__ y,
    int batch_size)
{
    const int row = blockIdx.x;
    if (row >= batch_size) return;

    const int lane = threadIdx.x & 63;
    const __hip_bfloat16* x_row = x + row * H;
    __hip_bfloat16* y_row = y + row * H;

    float xv[ELEMS_PER_THREAD];
    #pragma unroll
    for (int i = 0; i < ELEMS_PER_THREAD; i++) {
        xv[i] = __bfloat162float(x_row[lane * ELEMS_PER_THREAD + i]);
    }

    float sum_sq = 0.0f;
    #pragma unroll
    for (int i = 0; i < ELEMS_PER_THREAD; i++) {
        sum_sq += xv[i] * xv[i];
    }

    #pragma unroll
    for (int offset = 32; offset > 0; offset >>= 1) {
        sum_sq += __shfl_xor(sum_sq, offset, 64);
    }

    float mean_sq = sum_sq / float(H);
    float inv_rms = rsqrtf(mean_sq + 1e-6f);

    #pragma unroll
    for (int i = 0; i < ELEMS_PER_THREAD; i++) {
        int col = lane * ELEMS_PER_THREAD + i;
        float wv = __bfloat162float(w[col]);
        y_row[col] = __float2bfloat16(xv[i] * inv_rms * wv);
    }
}

torch::Tensor rmsnorm(torch::Tensor hidden_states, torch::Tensor weight) {
    const int B = hidden_states.size(0);
    const int H = hidden_states.size(1);
    TORCH_CHECK(H == 512);
    auto y = torch::empty_like(hidden_states);

    constexpr int ELEMS = 8;
    dim3 grid(B);
    dim3 block(64);
    rmsnorm_kernel<512, ELEMS><<<grid, block, 0, at::cuda::getCurrentCUDAStream()>>>(
        reinterpret_cast<const __hip_bfloat16*>(hidden_states.data_ptr<at::BFloat16>()),
        reinterpret_cast<const __hip_bfloat16*>(weight.data_ptr<at::BFloat16>()),
        reinterpret_cast<__hip_bfloat16*>(y.data_ptr<at::BFloat16>()),
        B);
    return y;
}
"""

CPP_SOURCE = r"""
torch::Tensor rmsnorm(torch::Tensor hidden_states, torch::Tensor weight);
"""

_module = load_inline(
    name="rmsnorm_hip",
    cpp_sources=CPP_SOURCE,
    cuda_sources=HIP_SOURCE,
    functions=["rmsnorm"],
    verbose=False,
)


@torch.no_grad()
def run(hidden_states, weight):
    return _module.rmsnorm(hidden_states, weight)

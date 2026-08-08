import torch
from torch.utils.cpp_extension import load_inline

_hip_src = r"""
#include <hip/hip_runtime.h>
#include <torch/extension.h>
#include <c10/hip/HIPStream.h>

template <int BLOCK_SIZE, int LOADS_PER_THREAD>
__global__ void rmsnorm_vec_kernel(
    const at::BFloat16* __restrict__ X,
    const at::BFloat16* __restrict__ W,
    at::BFloat16* __restrict__ Y,
    int N,
    float eps) {
    
    int row = blockIdx.x;
    int tid = threadIdx.x;
    
    const at::BFloat16* x_row = X + row * N;
    at::BFloat16* y_row = Y + row * N;
    
    constexpr int VEC = 8;
    constexpr int ELEMS_PER_THREAD = VEC * LOADS_PER_THREAD;
    
    float x_local[ELEMS_PER_THREAD];
    float w_local[ELEMS_PER_THREAD];
    float sum_sq = 0.0f;
    
    #pragma unroll
    for (int l = 0; l < LOADS_PER_THREAD; l++) {
        int base = (tid + l * BLOCK_SIZE) * VEC;
        uint4 xv = *reinterpret_cast<const uint4*>(x_row + base);
        uint4 wv = *reinterpret_cast<const uint4*>(W + base);
        
        at::BFloat16* xh = reinterpret_cast<at::BFloat16*>(&xv);
        at::BFloat16* wh = reinterpret_cast<at::BFloat16*>(&wv);
        
        #pragma unroll
        for (int j = 0; j < VEC; j++) {
            float xf = static_cast<float>(xh[j]);
            float wf = static_cast<float>(wh[j]);
            x_local[l * VEC + j] = xf;
            w_local[l * VEC + j] = wf;
            sum_sq += xf * xf;
        }
    }
    
    for (int offset = 32; offset > 0; offset >>= 1) {
        sum_sq += __shfl_xor(sum_sq, offset, 64);
    }
    
    __shared__ float warp_sums[16];
    int wf_id = tid / 64;
    int lane = tid % 64;
    if (lane == 0) warp_sums[wf_id] = sum_sq;
    __syncthreads();
    
    __shared__ float ssum;
    int num_wf = BLOCK_SIZE / 64;
    if (wf_id == 0) {
        sum_sq = (lane < num_wf) ? warp_sums[lane] : 0.0f;
        for (int offset = 32; offset > 0; offset >>= 1) {
            sum_sq += __shfl_xor(sum_sq, offset, 64);
        }
        if (lane == 0) ssum = sum_sq;
    }
    __syncthreads();
    
    float inv_rms = rsqrtf(ssum / N + eps);
    
    #pragma unroll
    for (int l = 0; l < LOADS_PER_THREAD; l++) {
        int base = (tid + l * BLOCK_SIZE) * VEC;
        uint4 yv;
        at::BFloat16* yh = reinterpret_cast<at::BFloat16*>(&yv);
        #pragma unroll
        for (int j = 0; j < VEC; j++) {
            yh[j] = at::BFloat16(x_local[l * VEC + j] * inv_rms * w_local[l * VEC + j]);
        }
        *reinterpret_cast<uint4*>(y_row + base) = yv;
    }
}

torch::Tensor rmsnorm(torch::Tensor x, torch::Tensor w) {
    int batch = x.size(0);
    int N = x.size(1);
    auto y = torch::empty_like(x);
    
    const int BLOCK_SIZE = 128;
    const int LOADS = 4;
    
    dim3 grid(batch);
    dim3 block(BLOCK_SIZE);
    
    auto stream = c10::hip::getCurrentHIPStream().stream();
    rmsnorm_vec_kernel<BLOCK_SIZE, LOADS><<<grid, block, 0, stream>>>(
        (const at::BFloat16*)x.data_ptr(),
        (const at::BFloat16*)w.data_ptr(),
        (at::BFloat16*)y.data_ptr(),
        N, 1e-5f);
    
    return y;
}
"""

_mod = load_inline(
    name="rmsnorm_hip_kern",
    cpp_sources=["torch::Tensor rmsnorm(torch::Tensor x, torch::Tensor w);"],
    cuda_sources=[_hip_src],
    functions=["rmsnorm"],
    extra_cuda_cflags=["-O3", "-std=c++17"],
    verbose=False,
)

@torch.no_grad()
def run(hidden_states, weight):
    batch_size, hidden_size = hidden_states.shape
    assert hidden_size == 4096
    return _mod.rmsnorm(hidden_states, weight)

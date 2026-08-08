import torch
import triton
import triton.language as tl
from torch.utils.cpp_extension import load_inline

# -----------------------------------------------------------------------
# HIP C++ kernel: low launch overhead, best for small batches
# -----------------------------------------------------------------------
_HIP_CPP = r"""
torch::Tensor fused_add_rmsnorm_hip(torch::Tensor hidden, torch::Tensor residual, torch::Tensor weight);
"""

_HIP_CUDA = r"""
#include <torch/extension.h>
#include <hip/hip_runtime.h>
#include <hip/hip_bfloat16.h>
#include <c10/hip/HIPStream.h>

template<int WARPS_PER_ROW, int ELEMS_PER_THREAD, int ROWS_PER_BLOCK>
__global__ void fused_add_rmsnorm_kernel(
    const hip_bfloat16* __restrict__ hidden,
    const hip_bfloat16* __restrict__ residual,
    const hip_bfloat16* __restrict__ weight,
    hip_bfloat16* __restrict__ output,
    int hidden_size, int num_rows) {

    int block_row = blockIdx.x * ROWS_PER_BLOCK + (threadIdx.y / WARPS_PER_ROW);
    if (block_row >= num_rows) return;

    int my_warp = threadIdx.y % WARPS_PER_ROW;
    int tid = threadIdx.x;
    int BLOCK_DIM = WARPS_PER_ROW * 64;
    int start = my_warp * 64;

    const hip_bfloat16* h = hidden + block_row * hidden_size;
    const hip_bfloat16* r = residual + block_row * hidden_size;
    hip_bfloat16* o = output + block_row * hidden_size;

    float local[ELEMS_PER_THREAD];
    float sum_sq = 0.0f;

    #pragma unroll
    for (int i = 0; i < ELEMS_PER_THREAD; i++) {
        int idx = start + tid + i * BLOCK_DIM;
        float hv = static_cast<float>(h[idx]);
        float rv = static_cast<float>(r[idx]);
        float xv = hv + rv;
        local[i] = xv;
        sum_sq += xv * xv;
    }

    for (int offset = 32; offset > 0; offset /= 2) {
        sum_sq += __shfl_xor(sum_sq, offset);
    }

    __shared__ float warp_sum[ROWS_PER_BLOCK][WARPS_PER_ROW];
    int my_row = threadIdx.y / WARPS_PER_ROW;
    if (tid == 0) warp_sum[my_row][my_warp] = sum_sq;
    __syncthreads();

    if (my_warp == 0 && tid < WARPS_PER_ROW) {
        sum_sq = warp_sum[my_row][tid];
        for (int offset = WARPS_PER_ROW / 2; offset > 0; offset /= 2) {
            sum_sq += __shfl_xor(sum_sq, offset);
        }
        if (tid == 0) warp_sum[my_row][0] = sum_sq;
    }
    __syncthreads();

    float inv_rms = rsqrtf(warp_sum[my_row][0] / (float)hidden_size + 1e-5f);

    #pragma unroll
    for (int i = 0; i < ELEMS_PER_THREAD; i++) {
        int idx = start + tid + i * BLOCK_DIM;
        float wv = static_cast<float>(weight[idx]);
        float yv = (local[i] * inv_rms) * wv;
        o[idx] = hip_bfloat16(yv);
    }
}

torch::Tensor fused_add_rmsnorm_hip(torch::Tensor hidden, torch::Tensor residual, torch::Tensor weight) {
    int num_rows = hidden.size(0);
    int hidden_size = hidden.size(1);
    auto output = torch::empty_like(hidden);
    int blocks = num_rows;
    hipStream_t stream = c10::hip::getCurrentHIPStream().stream();
    fused_add_rmsnorm_kernel<8, 8, 1><<<blocks, dim3(64, 8), 0, stream>>>(
        (const hip_bfloat16*)hidden.data_ptr(),
        (const hip_bfloat16*)residual.data_ptr(),
        (const hip_bfloat16*)weight.data_ptr(),
        (hip_bfloat16*)output.data_ptr(),
        hidden_size, num_rows);
    return output;
}
"""

_hip_mod = None

def _get_hip_mod():
    global _hip_mod
    if _hip_mod is None:
        _hip_mod = load_inline(
            name="fused_rmsnorm_hip_kernel",
            cpp_sources=[_HIP_CPP],
            cuda_sources=[_HIP_CUDA],
            functions=["fused_add_rmsnorm_hip"],
            extra_cuda_cflags=["-O3"],
            verbose=False,
        )
    return _hip_mod


# -----------------------------------------------------------------------
# Triton kernel: high bandwidth, best for large batches
# -----------------------------------------------------------------------
@triton.jit
def _fused_add_rmsnorm_triton(
    hidden_ptr, residual_ptr, weight_ptr, output_ptr,
    stride_h, stride_r, stride_o,
    H: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)

    h = tl.load(hidden_ptr + row * stride_h + cols, mask=cols < H, other=0.0).to(tl.float32)
    r = tl.load(residual_ptr + row * stride_r + cols, mask=cols < H, other=0.0).to(tl.float32)
    x = h + r

    sum_sq = tl.sum(x * x, axis=0)
    inv_rms = 1.0 / tl.sqrt(sum_sq / H + 1e-5)

    w = tl.load(weight_ptr + cols, mask=cols < H, other=0.0).to(tl.float32)
    y = (x * inv_rms) * w
    tl.store(output_ptr + row * stride_o + cols, y.to(tl.bfloat16), mask=cols < H)


_TRITON_BLOCK = 4096


@torch.no_grad()
def run(hidden_states, residual, weight):
    _, hidden_size = hidden_states.shape
    assert hidden_size == 4096

    rows = hidden_states.shape[0]

    if rows < 2000:
        return _get_hip_mod().fused_add_rmsnorm_hip(hidden_states, residual, weight)

    out = torch.empty_like(hidden_states)
    num_warps = 4 if rows >= 1024 else 16
    grid = (rows,)
    _fused_add_rmsnorm_triton[grid](
        hidden_states, residual, weight, out,
        hidden_states.stride(0), residual.stride(0), out.stride(0),
        H=hidden_size, BLOCK=_TRITON_BLOCK,
        num_warps=num_warps, num_stages=1,
    )
    return out

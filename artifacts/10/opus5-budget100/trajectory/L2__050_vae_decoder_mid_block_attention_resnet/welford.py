import torch, os
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline

decl = r'''
#include <torch/extension.h>
std::vector<torch::Tensor> moments(torch::Tensor X, int64_t G, double eps);
'''

src = r'''
#include <torch/extension.h>
#include <hip/hip_runtime.h>

#define WARP 64

struct W { float mean; float m2; float nf; };

__device__ __forceinline__ W wreduce(W a, float x){
  float d = x - a.mean;
  float nf = a.nf + 1.0f;
  float nm = a.mean + d / nf;
  float nd = x - nm;
  return W{nm, a.m2 + d*nd, nf};
}
__device__ __forceinline__ W wcombine(W a, W b){
  if (a.nf == 0.f) return b;
  if (b.nf == 0.f) return a;
  float d = b.mean - a.mean;
  float nn = a.nf + b.nf;
  float nb = b.nf / nn;
  return W{a.mean + d*nb, a.m2 + b.m2 + d*d*a.nf*nb, nn};
}
__device__ __forceinline__ W wshfl(W v, int off){
  W r;
  r.mean = __shfl_down(v.mean, off, WARP);
  r.m2   = __shfl_down(v.m2,   off, WARP);
  r.nf   = __shfl_down(v.nf,   off, WARP);
  return r;
}
__device__ __forceinline__ W warp_reduce(W v){
  #pragma unroll
  for (int off = WARP/2; off > 0; off >>= 1) v = wcombine(v, wshfl(v, off));
  return v;
}

template<int NT>
__global__ void rowwise_moments(long N, float eps, const float* __restrict__ X,
                                float* __restrict__ mean, float* __restrict__ rstd){
  __shared__ W sh[NT / WARP > 0 ? NT / WARP : 1];
  long i = blockIdx.x;
  W val{0.f, 0.f, 0.f};
  for (long j = threadIdx.x; j < N; j += NT) {
    val = wreduce(val, X[i*N + j]);
  }
  if (NT <= WARP) {
    val = warp_reduce(val);
  } else {
    int lane = threadIdx.x % WARP;
    int wid  = threadIdx.x / WARP;
    val = warp_reduce(val);
    if (lane == 0) sh[wid] = val;
    __syncthreads();
    if (threadIdx.x < (NT / WARP)) val = sh[lane];
    else val = W{0.f, 0.f, 0.f};
    if (wid == 0) val = warp_reduce(val);
  }
  if (threadIdx.x == 0) {
    mean[i] = val.mean;
    rstd[i] = rsqrtf(val.m2 / val.nf + (float)eps);
  }
}

std::vector<torch::Tensor> moments(torch::Tensor X, int64_t G, double eps){
  auto B = X.size(0); auto C = X.size(1);
  long HxW = X.numel() / (B * C);
  long D = C / G;
  long N = D * HxW;
  auto mean = torch::empty({B, G}, X.options());
  auto rstd = torch::empty({B, G}, X.options());
  long blocks = B * G;
  const float* xp = X.data_ptr<float>();
  float* mp = mean.data_ptr<float>(); float* rp = rstd.data_ptr<float>();
  if (N < 512) {
    hipLaunchKernelGGL((rowwise_moments<WARP>), dim3(blocks), dim3(WARP), 0, 0, N, (float)eps, xp, mp, rp);
  } else {
    hipLaunchKernelGGL((rowwise_moments<512>), dim3(blocks), dim3(512), 0, 0, N, (float)eps, xp, mp, rp);
  }
  return {mean, rstd};
}
'''

m = load_inline(name='welfordprobe', cpp_sources=decl, cuda_sources=src,
                functions=['moments'], verbose=False, extra_cuda_cflags=['-O3'])

torch.manual_seed(0)
for (B,C,H,W) in [(1,512,32,32),(32,512,32,32),(2,512,64,64),(1,512,16,16),(4,512,16,16),(1,512,61,61)]:
    x = torch.randn(B,C,H,W,device='cuda')
    g = torch.randn(C,device='cuda'); b=torch.randn(C,device='cuda')
    eps=1e-6
    out, mean, rstd = torch.native_group_norm(x, g, b, B, C, H*W, 32, eps)
    mm, rr = m.moments(x, 32, eps)
    print(f"{B}x{C}x{H}x{W}: mean mismatch {(mm!=mean).sum().item()}/{mean.numel()}  rstd mismatch {(rr!=rstd).sum().item()}  maxdiff {(mm-mean).abs().max().item():.3e}/{(rr-rstd).abs().max().item():.3e}")

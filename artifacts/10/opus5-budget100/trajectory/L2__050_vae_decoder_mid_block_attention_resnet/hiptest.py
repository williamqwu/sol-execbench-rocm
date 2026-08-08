import torch, os
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline

os.environ.setdefault('TORCH_EXTENSIONS_DIR','/var/tmp/solbench/torchext')

src = r'''
#include <torch/extension.h>
#include <hip/hip_runtime.h>

__global__ void k_exp(const float* __restrict__ x, float* __restrict__ y, int n){
  int i = blockIdx.x*blockDim.x + threadIdx.x;
  if(i<n) y[i] = expf(x[i]);
}
__global__ void k_silu(const float* __restrict__ x, float* __restrict__ y, int n){
  int i = blockIdx.x*blockDim.x + threadIdx.x;
  if(i<n){ float v=x[i]; y[i] = v/(1.0f+expf(-v)); }
}
__global__ void k_silu2(const float* __restrict__ x, float* __restrict__ y, int n){
  int i = blockIdx.x*blockDim.x + threadIdx.x;
  if(i<n){ float v=x[i]; y[i] = v*(1.0f/(1.0f+expf(-v))); }
}
__global__ void k_rsqrt(const float* __restrict__ x, float* __restrict__ y, int n){
  int i = blockIdx.x*blockDim.x + threadIdx.x;
  if(i<n) y[i] = rsqrtf(x[i]);
}

torch::Tensor f_exp(torch::Tensor x){ auto y=torch::empty_like(x); int n=x.numel();
  hipLaunchKernelGGL(k_exp, dim3((n+255)/256), dim3(256), 0, 0, x.data_ptr<float>(), y.data_ptr<float>(), n); return y; }
torch::Tensor f_silu(torch::Tensor x){ auto y=torch::empty_like(x); int n=x.numel();
  hipLaunchKernelGGL(k_silu, dim3((n+255)/256), dim3(256), 0, 0, x.data_ptr<float>(), y.data_ptr<float>(), n); return y; }
torch::Tensor f_silu2(torch::Tensor x){ auto y=torch::empty_like(x); int n=x.numel();
  hipLaunchKernelGGL(k_silu2, dim3((n+255)/256), dim3(256), 0, 0, x.data_ptr<float>(), y.data_ptr<float>(), n); return y; }
torch::Tensor f_rsqrt(torch::Tensor x){ auto y=torch::empty_like(x); int n=x.numel();
  hipLaunchKernelGGL(k_rsqrt, dim3((n+255)/256), dim3(256), 0, 0, x.data_ptr<float>(), y.data_ptr<float>(), n); return y; }
'''

decl = r'''
#include <torch/extension.h>
torch::Tensor f_exp(torch::Tensor x);
torch::Tensor f_silu(torch::Tensor x);
torch::Tensor f_silu2(torch::Tensor x);
torch::Tensor f_rsqrt(torch::Tensor x);
'''

m = load_inline(name='hipprobe', cpp_sources=decl, cuda_sources=src,
                functions=['f_exp','f_silu','f_silu2','f_rsqrt'], verbose=False,
                extra_cuda_cflags=['-O3'])

x = torch.randn(1<<22, device='cuda')*3
print("exp   mismatch:", (m.f_exp(x)!=torch.exp(x)).sum().item(), "/", x.numel())
print("silu  mismatch:", (m.f_silu(x)!=F.silu(x)).sum().item())
print("silu2 mismatch:", (m.f_silu2(x)!=F.silu(x)).sum().item())
xp = x.abs()+0.5
print("rsqrt mismatch:", (m.f_rsqrt(xp)!=torch.rsqrt(xp)).sum().item())

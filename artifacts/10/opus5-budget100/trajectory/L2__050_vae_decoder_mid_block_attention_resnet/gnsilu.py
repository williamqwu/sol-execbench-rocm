import torch, os
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline

decl = r'''
#include <torch/extension.h>
torch::Tensor gn_silu(torch::Tensor X, torch::Tensor gamma, torch::Tensor beta, int64_t G, double eps, int64_t variant);
torch::Tensor gn_only(torch::Tensor X, torch::Tensor gamma, torch::Tensor beta, int64_t G, double eps, int64_t variant);
'''

src = r'''
#include <torch/extension.h>
#include <hip/hip_runtime.h>
#define WARP 64
struct W { float mean; float m2; float nf; };
__device__ __forceinline__ W wreduce(W a, float x){
  float d = x - a.mean; float nf = a.nf + 1.0f; float nm = a.mean + d/nf; float nd = x - nm;
  return W{nm, a.m2 + d*nd, nf};
}
__device__ __forceinline__ W wcombine(W a, W b){
  if (a.nf==0.f) return b; if (b.nf==0.f) return a;
  float d=b.mean-a.mean; float nn=a.nf+b.nf; float nb=b.nf/nn;
  return W{a.mean+d*nb, a.m2+b.m2+d*d*a.nf*nb, nn};
}
__device__ __forceinline__ W wshfl(W v,int off){
  return W{__shfl_down(v.mean,off,WARP), __shfl_down(v.m2,off,WARP), __shfl_down(v.nf,off,WARP)};
}
__device__ __forceinline__ W warp_reduce(W v){
  #pragma unroll
  for(int o=WARP/2;o>0;o>>=1) v=wcombine(v,wshfl(v,o));
  return v;
}
template<int NT>
__global__ void rowmom(long N,float eps,const float* __restrict__ X,float* __restrict__ mean,float* __restrict__ rstd){
  __shared__ W sh[(NT/WARP)>0?(NT/WARP):1];
  long i=blockIdx.x; W val{0.f,0.f,0.f};
  for(long j=threadIdx.x;j<N;j+=NT) val=wreduce(val,X[i*N+j]);
  if(NT<=WARP){ val=warp_reduce(val); }
  else{
    int lane=threadIdx.x%WARP, wid=threadIdx.x/WARP;
    val=warp_reduce(val);
    if(lane==0) sh[wid]=val;
    __syncthreads();
    val = (threadIdx.x < (NT/WARP)) ? sh[lane] : W{0.f,0.f,0.f};
    if(wid==0) val=warp_reduce(val);
  }
  if(threadIdx.x==0){ mean[i]=val.mean; rstd[i]=rsqrtf(val.m2/val.nf+eps); }
}

// fused params: a = rstd*gamma ; b = beta - mean*rstd*gamma  (variant selects order)
__global__ void fusedparams(long N,long C,long G,const float* __restrict__ mean,const float* __restrict__ rstd,
                            const float* __restrict__ gamma,const float* __restrict__ beta,
                            float* __restrict__ a,float* __restrict__ b,int variant){
  long i=blockIdx.x*blockDim.x+threadIdx.x;
  if(i>=N*C) return;
  long ng = i/C*G + (i%C)/(C/G);
  long c = i%C;
  float r=rstd[ng], mu=mean[ng], g=gamma[c], be=beta[c];
  float av, bv;
  if(variant==0){ av=r*g; bv=-mu*av+be; }
  else if(variant==1){ av=r*g; bv=be-mu*av; }
  else { av=g*r; bv=fmaf(-mu, av, be); }
  a[i]=av; b[i]=bv;
}

__global__ void applyk(long total,long HxW,const float* __restrict__ X,const float* __restrict__ a,
                       const float* __restrict__ b,float* __restrict__ Y,int silu,int variant){
  long i=blockIdx.x*blockDim.x+threadIdx.x;
  if(i>=total) return;
  long nc=i/HxW;
  float v;
  if(variant==0) v = a[nc]*X[i]+b[nc];
  else v = fmaf(a[nc], X[i], b[nc]);
  if(silu) v = v/(1.0f+expf(-v));
  Y[i]=v;
}

static torch::Tensor impl(torch::Tensor X,torch::Tensor gamma,torch::Tensor beta,int64_t G,double eps,int64_t variant,int silu){
  auto B=X.size(0), C=X.size(1);
  long HxW = X.numel()/(B*C);
  long N = (C/G)*HxW;
  auto opt=X.options();
  auto mean=torch::empty({B,G},opt), rstd=torch::empty({B,G},opt);
  const float* xp=X.data_ptr<float>();
  if(N<512) hipLaunchKernelGGL((rowmom<WARP>),dim3(B*G),dim3(WARP),0,0,N,(float)eps,xp,mean.data_ptr<float>(),rstd.data_ptr<float>());
  else      hipLaunchKernelGGL((rowmom<512>), dim3(B*G),dim3(512), 0,0,N,(float)eps,xp,mean.data_ptr<float>(),rstd.data_ptr<float>());
  auto a=torch::empty({B,C},opt), b=torch::empty({B,C},opt);
  long nc=B*C;
  hipLaunchKernelGGL(fusedparams,dim3((nc+255)/256),dim3(256),0,0,B,C,G,mean.data_ptr<float>(),rstd.data_ptr<float>(),
                     gamma.data_ptr<float>(),beta.data_ptr<float>(),a.data_ptr<float>(),b.data_ptr<float>(),(int)(variant%3));
  auto Y=torch::empty_like(X);
  long total=X.numel();
  hipLaunchKernelGGL(applyk,dim3((total+255)/256),dim3(256),0,0,total,HxW,xp,a.data_ptr<float>(),b.data_ptr<float>(),
                     Y.data_ptr<float>(),silu,(int)(variant/3));
  return Y;
}
torch::Tensor gn_silu(torch::Tensor X,torch::Tensor g,torch::Tensor b,int64_t G,double eps,int64_t v){ return impl(X,g,b,G,eps,v,1); }
torch::Tensor gn_only(torch::Tensor X,torch::Tensor g,torch::Tensor b,int64_t G,double eps,int64_t v){ return impl(X,g,b,G,eps,v,0); }
'''

m = load_inline(name='gnsiluprobe', cpp_sources=decl, cuda_sources=src,
                functions=['gn_silu','gn_only'], verbose=False, extra_cuda_cflags=['-O3'])

torch.manual_seed(0)
for (B,C,H,W) in [(1,512,32,32),(32,512,32,32),(4,512,16,16)]:
    x = torch.randn(B,C,H,W,device='cuda')
    g = torch.randn(C,device='cuda'); b = torch.randn(C,device='cuda')
    eps=1e-6
    ref_gn = F.group_norm(x,32,g,b,eps)
    ref_gs = F.silu(ref_gn)
    print(f"=== {B}x{C}x{H}x{W}")
    for v in range(6):
        o1 = m.gn_only(x,g,b,32,eps,v)
        o2 = m.gn_silu(x,g,b,32,eps,v)
        print(f"   v{v}: gn mismatch {(o1!=ref_gn).sum().item():8d}  gn+silu mismatch {(o2!=ref_gs).sum().item():8d}")

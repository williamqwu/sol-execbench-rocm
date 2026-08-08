DECL = r'''
#include <torch/extension.h>
torch::Tensor gn_silu(torch::Tensor X, torch::Tensor gamma, torch::Tensor beta, int64_t G, double eps);
torch::Tensor gn_plain(torch::Tensor X, torch::Tensor gamma, torch::Tensor beta, int64_t G, double eps);
torch::Tensor bias_add_gn_silu(torch::Tensor X, torch::Tensor tb, torch::Tensor gamma, torch::Tensor beta, int64_t G, double eps);
torch::Tensor add_bias(torch::Tensor X, torch::Tensor tb);
torch::Tensor add_res(torch::Tensor X, torch::Tensor R);
'''

SRC = r'''
#include <torch/extension.h>
#include <hip/hip_runtime.h>

#define WARP 64

struct W { float mean; float m2; float nf; };

__device__ __forceinline__ W wreduce(W a, float x){
  float d = x - a.mean; float nf = a.nf + 1.0f; float nm = a.mean + d/nf; float nd = x - nm;
  return W{nm, a.m2 + d*nd, nf};
}
__device__ __forceinline__ W wcombine(W a, W b){
  if (a.nf == 0.f) return b;
  if (b.nf == 0.f) return a;
  float d = b.mean - a.mean; float nn = a.nf + b.nf; float nb = b.nf / nn;
  return W{a.mean + d*nb, a.m2 + b.m2 + d*d*a.nf*nb, nn};
}
__device__ __forceinline__ W wshfl(W v, int off){
  return W{__shfl_down(v.mean,off,WARP), __shfl_down(v.m2,off,WARP), __shfl_down(v.nf,off,WARP)};
}
__device__ __forceinline__ W warp_reduce(W v){
  #pragma unroll
  for (int o = WARP/2; o > 0; o >>= 1) v = wcombine(v, wshfl(v, o));
  return v;
}

// Mirrors at::native RowwiseMomentsCUDAKernel: same Welford recurrence, same
// tree-combine order, so mean/rstd come out bit-identical to F.group_norm.
template<int NT>
__global__ void rowmom(long N, float eps, const float* __restrict__ X,
                       const float* __restrict__ TB, long G, long D, long HxW,
                       float* __restrict__ mean, float* __restrict__ rstd){
  __shared__ W sh[(NT/WARP) > 0 ? (NT/WARP) : 1];
  long i = blockIdx.x;
  W val{0.f, 0.f, 0.f};
  if (TB == nullptr) {
    for (long j = threadIdx.x; j < N; j += NT) val = wreduce(val, X[i*N + j]);
  } else {
    // Row i covers group (i%G) of batch (i/G); j runs over D*HxW.
    // TB is [B, C], so the bias for this row starts at batch*C + group*D.
    long cbase = (i / G) * (G * D) + (i % G) * D;
    for (long j = threadIdx.x; j < N; j += NT)
      val = wreduce(val, X[i*N + j] + TB[cbase + j / HxW]);
  }
  if (NT <= WARP) { val = warp_reduce(val); }
  else {
    int lane = threadIdx.x % WARP, wid = threadIdx.x / WARP;
    val = warp_reduce(val);
    if (lane == 0) sh[wid] = val;
    __syncthreads();
    val = (threadIdx.x < (NT/WARP)) ? sh[lane] : W{0.f, 0.f, 0.f};
    if (wid == 0) val = warp_reduce(val);
  }
  if (threadIdx.x == 0){ mean[i] = val.mean; rstd[i] = rsqrtf(val.m2 / val.nf + eps); }
}

__global__ void fusedparams(long B, long C, long G,
                            const float* __restrict__ mean, const float* __restrict__ rstd,
                            const float* __restrict__ gamma, const float* __restrict__ beta,
                            float* __restrict__ a, float* __restrict__ b){
  long i = blockIdx.x*blockDim.x + threadIdx.x;
  if (i >= B*C) return;
  long c = i % C;
  long ng = (i / C) * G + c / (C / G);
  float r = rstd[ng], mu = mean[ng], g = gamma[c], be = beta[c];
  float av = r * g;
  a[i] = av;
  b[i] = -mu * av + be;
}

template<int SILU, int TBIAS>
__global__ void applyk(long total, long HxW, long C,
                       const float* __restrict__ X, const float* __restrict__ TB,
                       const float* __restrict__ a, const float* __restrict__ b,
                       float* __restrict__ Y){
  long i = blockIdx.x*blockDim.x + threadIdx.x;
  if (i >= total) return;
  long nc = i / HxW;          // flat (batch, channel) index
  float x = X[i];
  if (TBIAS) x = x + TB[nc];  // TB is [B, C], indexed the same way
  float v = a[nc]*x + b[nc];
  if (SILU) v = v / (1.0f + expf(-v));
  Y[i] = v;
}

__global__ void addbias_k(long total, long HxW, long C,
                          const float* __restrict__ X, const float* __restrict__ TB,
                          float* __restrict__ Y){
  long i = blockIdx.x*blockDim.x + threadIdx.x;
  if (i >= total) return;
  Y[i] = X[i] + TB[(i / HxW) % C];
}

__global__ void addres_k(long total, const float* __restrict__ X,
                         const float* __restrict__ R, float* __restrict__ Y){
  long i = blockIdx.x*blockDim.x + threadIdx.x;
  if (i >= total) return;
  Y[i] = X[i] + R[i];
}

static void moments(torch::Tensor X, const float* TB, long B, long C, long G, long HxW,
                    double eps, torch::Tensor mean, torch::Tensor rstd){
  long D = C / G;
  long N = D * HxW;
  const float* xp = X.data_ptr<float>();
  float* mp = mean.data_ptr<float>(); float* rp = rstd.data_ptr<float>();
  if (N < 512)
    hipLaunchKernelGGL((rowmom<WARP>), dim3(B*G), dim3(WARP), 0, 0, N, (float)eps, xp, TB, G, D, HxW, mp, rp);
  else
    hipLaunchKernelGGL((rowmom<512>), dim3(B*G), dim3(512), 0, 0, N, (float)eps, xp, TB, G, D, HxW, mp, rp);
}

static torch::Tensor gn_impl(torch::Tensor X, torch::Tensor gamma, torch::Tensor beta,
                             int64_t G, double eps, int silu, const float* TB){
  long B = X.size(0), C = X.size(1);
  long HxW = X.numel() / (B*C);
  auto opt = X.options();
  auto mean = torch::empty({B, G}, opt), rstd = torch::empty({B, G}, opt);
  moments(X, TB, B, C, G, HxW, eps, mean, rstd);
  auto a = torch::empty({B, C}, opt), b = torch::empty({B, C}, opt);
  long nc = B*C;
  hipLaunchKernelGGL(fusedparams, dim3((nc+255)/256), dim3(256), 0, 0, B, C, G,
                     mean.data_ptr<float>(), rstd.data_ptr<float>(),
                     gamma.data_ptr<float>(), beta.data_ptr<float>(),
                     a.data_ptr<float>(), b.data_ptr<float>());
  auto Y = torch::empty_like(X);
  long total = X.numel();
  int blocks = (int)((total + 255)/256);
  const float* xp = X.data_ptr<float>();
  float* yp = Y.data_ptr<float>();
  const float* ap = a.data_ptr<float>(); const float* bp = b.data_ptr<float>();
  if (silu && TB)        hipLaunchKernelGGL((applyk<1,1>), dim3(blocks), dim3(256), 0, 0, total, HxW, C, xp, TB, ap, bp, yp);
  else if (silu)         hipLaunchKernelGGL((applyk<1,0>), dim3(blocks), dim3(256), 0, 0, total, HxW, C, xp, TB, ap, bp, yp);
  else if (TB)           hipLaunchKernelGGL((applyk<0,1>), dim3(blocks), dim3(256), 0, 0, total, HxW, C, xp, TB, ap, bp, yp);
  else                   hipLaunchKernelGGL((applyk<0,0>), dim3(blocks), dim3(256), 0, 0, total, HxW, C, xp, TB, ap, bp, yp);
  return Y;
}

torch::Tensor gn_silu(torch::Tensor X, torch::Tensor g, torch::Tensor b, int64_t G, double eps){
  return gn_impl(X, g, b, G, eps, 1, nullptr);
}
torch::Tensor gn_plain(torch::Tensor X, torch::Tensor g, torch::Tensor b, int64_t G, double eps){
  return gn_impl(X, g, b, G, eps, 0, nullptr);
}
torch::Tensor bias_add_gn_silu(torch::Tensor X, torch::Tensor tb, torch::Tensor g, torch::Tensor b,
                               int64_t G, double eps){
  return gn_impl(X, g, b, G, eps, 1, tb.data_ptr<float>());
}
torch::Tensor add_bias(torch::Tensor X, torch::Tensor tb){
  auto Y = torch::empty_like(X);
  long B = X.size(0), C = X.size(1);
  long HxW = X.numel()/(B*C);
  long total = X.numel();
  hipLaunchKernelGGL(addbias_k, dim3((total+255)/256), dim3(256), 0, 0, total, HxW, C,
                     X.data_ptr<float>(), tb.data_ptr<float>(), Y.data_ptr<float>());
  return Y;
}
torch::Tensor add_res(torch::Tensor X, torch::Tensor R){
  auto Y = torch::empty_like(X);
  long total = X.numel();
  hipLaunchKernelGGL(addres_k, dim3((total+255)/256), dim3(256), 0, 0, total,
                     X.data_ptr<float>(), R.data_ptr<float>(), Y.data_ptr<float>());
  return Y;
}
'''

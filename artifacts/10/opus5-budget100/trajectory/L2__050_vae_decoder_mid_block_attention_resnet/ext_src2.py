DECL = r'''
#include <torch/extension.h>
torch::Tensor gn_silu(torch::Tensor X, torch::Tensor gamma, torch::Tensor beta, int64_t G, double eps);
torch::Tensor gn_plain(torch::Tensor X, torch::Tensor gamma, torch::Tensor beta, int64_t G, double eps);
torch::Tensor bias_add_gn_silu(torch::Tensor X, torch::Tensor tb, torch::Tensor gamma, torch::Tensor beta, int64_t G, double eps);
torch::Tensor add_res(torch::Tensor X, torch::Tensor R);
torch::Tensor add_res_gn_silu(torch::Tensor X, torch::Tensor R, torch::Tensor gamma, torch::Tensor beta, int64_t G, double eps, torch::Tensor out_res);
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

// Mirrors at::native RowwiseMomentsCUDAKernel exactly: same Welford recurrence,
// same strided element order per thread, same warp tree-combine. Bit-identical
// mean/rstd to F.group_norm, which the tight-tolerance workloads require.
template<int NT, int MODE>
__global__ void rowmom(long N, float eps, const float* __restrict__ X,
                       const float* __restrict__ TB, const float* __restrict__ RES,
                       long G, long D, long HxW,
                       float* __restrict__ mean, float* __restrict__ rstd){
  __shared__ W sh[(NT/WARP) > 0 ? (NT/WARP) : 1];
  long i = blockIdx.x;
  W val{0.f, 0.f, 0.f};
  long base = i*N;
  if (MODE == 0) {
    for (long j = threadIdx.x; j < N; j += NT) val = wreduce(val, X[base + j]);
  } else if (MODE == 1) {
    long cbase = (i / G) * (G * D) + (i % G) * D;
    for (long j = threadIdx.x; j < N; j += NT)
      val = wreduce(val, X[base + j] + TB[cbase + j / HxW]);
  } else {
    for (long j = threadIdx.x; j < N; j += NT)
      val = wreduce(val, X[base + j] + RES[base + j]);
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

__global__ void fusedparams(long BC, long C, long G,
                            const float* __restrict__ mean, const float* __restrict__ rstd,
                            const float* __restrict__ gamma, const float* __restrict__ beta,
                            float* __restrict__ a, float* __restrict__ b){
  long i = blockIdx.x*blockDim.x + threadIdx.x;
  if (i >= BC) return;
  long c = i % C;
  long ng = (i / C) * G + c / (C / G);
  float r = rstd[ng], mu = mean[ng], g = gamma[c], be = beta[c];
  float av = r * g;
  a[i] = av;
  b[i] = -mu * av + be;
}

// MODE: 0 plain, 1 add per-channel bias first, 2 add residual first (and store it)
template<int SILU, int MODE>
__global__ void applyk(long total, long HxW,
                       const float* __restrict__ X, const float* __restrict__ TB,
                       const float* __restrict__ RES, float* __restrict__ RESOUT,
                       const float* __restrict__ a, const float* __restrict__ b,
                       float* __restrict__ Y){
  long i = blockIdx.x*blockDim.x + threadIdx.x;
  if (i >= total) return;
  long nc = i / HxW;
  float x = X[i];
  if (MODE == 1) x = x + TB[nc];
  if (MODE == 2) { x = x + RES[i]; RESOUT[i] = x; }
  float v = a[nc]*x + b[nc];
  if (SILU) v = v / (1.0f + expf(-v));
  Y[i] = v;
}

__global__ void addres_k(long total, const float* __restrict__ X,
                         const float* __restrict__ R, float* __restrict__ Y){
  long i = blockIdx.x*blockDim.x + threadIdx.x;
  if (i >= total) return;
  Y[i] = X[i] + R[i];
}

static void launch_moments(const float* xp, const float* TB, const float* RES,
                           long B, long C, long G, long HxW, double eps,
                           float* mp, float* rp, int mode){
  long D = C / G, N = D * HxW;
  dim3 grid(B*G);
  if (N < 512) {
    if (mode == 0)      hipLaunchKernelGGL((rowmom<WARP,0>), grid, dim3(WARP), 0, 0, N, (float)eps, xp, TB, RES, G, D, HxW, mp, rp);
    else if (mode == 1) hipLaunchKernelGGL((rowmom<WARP,1>), grid, dim3(WARP), 0, 0, N, (float)eps, xp, TB, RES, G, D, HxW, mp, rp);
    else                hipLaunchKernelGGL((rowmom<WARP,2>), grid, dim3(WARP), 0, 0, N, (float)eps, xp, TB, RES, G, D, HxW, mp, rp);
  } else {
    if (mode == 0)      hipLaunchKernelGGL((rowmom<512,0>), grid, dim3(512), 0, 0, N, (float)eps, xp, TB, RES, G, D, HxW, mp, rp);
    else if (mode == 1) hipLaunchKernelGGL((rowmom<512,1>), grid, dim3(512), 0, 0, N, (float)eps, xp, TB, RES, G, D, HxW, mp, rp);
    else                hipLaunchKernelGGL((rowmom<512,2>), grid, dim3(512), 0, 0, N, (float)eps, xp, TB, RES, G, D, HxW, mp, rp);
  }
}

static torch::Tensor gn_impl(torch::Tensor X, torch::Tensor gamma, torch::Tensor beta,
                             int64_t G, double eps, int silu, const float* TB,
                             const float* RES, float* RESOUT){
  long B = X.size(0), C = X.size(1);
  long HxW = X.numel() / (B*C);
  auto opt = X.options();
  auto mean = torch::empty({B, G}, opt), rstd = torch::empty({B, G}, opt);
  int mode = RES ? 2 : (TB ? 1 : 0);
  const float* xp = X.data_ptr<float>();
  launch_moments(xp, TB, RES, B, C, G, HxW, eps, mean.data_ptr<float>(), rstd.data_ptr<float>(), mode);

  auto a = torch::empty({B, C}, opt), b = torch::empty({B, C}, opt);
  long BC = B*C;
  hipLaunchKernelGGL(fusedparams, dim3((BC+255)/256), dim3(256), 0, 0, BC, C, G,
                     mean.data_ptr<float>(), rstd.data_ptr<float>(),
                     gamma.data_ptr<float>(), beta.data_ptr<float>(),
                     a.data_ptr<float>(), b.data_ptr<float>());

  auto Y = torch::empty_like(X);
  long total = X.numel();
  dim3 grid((total + 255)/256), blk(256);
  const float* ap = a.data_ptr<float>(); const float* bp = b.data_ptr<float>();
  float* yp = Y.data_ptr<float>();
  #define DISP(S, M) hipLaunchKernelGGL((applyk<S,M>), grid, blk, 0, 0, total, HxW, xp, TB, RES, RESOUT, ap, bp, yp)
  if (silu) { if (mode == 0) DISP(1,0); else if (mode == 1) DISP(1,1); else DISP(1,2); }
  else      { if (mode == 0) DISP(0,0); else if (mode == 1) DISP(0,1); else DISP(0,2); }
  #undef DISP
  return Y;
}

torch::Tensor gn_silu(torch::Tensor X, torch::Tensor g, torch::Tensor b, int64_t G, double eps){
  return gn_impl(X, g, b, G, eps, 1, nullptr, nullptr, nullptr);
}
torch::Tensor gn_plain(torch::Tensor X, torch::Tensor g, torch::Tensor b, int64_t G, double eps){
  return gn_impl(X, g, b, G, eps, 0, nullptr, nullptr, nullptr);
}
torch::Tensor bias_add_gn_silu(torch::Tensor X, torch::Tensor tb, torch::Tensor g, torch::Tensor b,
                               int64_t G, double eps){
  return gn_impl(X, g, b, G, eps, 1, tb.data_ptr<float>(), nullptr, nullptr);
}
// Computes res = X + R, writes it to out_res, and returns silu(groupnorm(res)).
torch::Tensor add_res_gn_silu(torch::Tensor X, torch::Tensor R, torch::Tensor g, torch::Tensor b,
                              int64_t G, double eps, torch::Tensor out_res){
  return gn_impl(X, g, b, G, eps, 1, nullptr, R.data_ptr<float>(), out_res.data_ptr<float>());
}
torch::Tensor add_res(torch::Tensor X, torch::Tensor R){
  auto Y = torch::empty_like(X);
  long total = X.numel();
  hipLaunchKernelGGL(addres_k, dim3((total+255)/256), dim3(256), 0, 0, total,
                     X.data_ptr<float>(), R.data_ptr<float>(), Y.data_ptr<float>());
  return Y;
}
'''

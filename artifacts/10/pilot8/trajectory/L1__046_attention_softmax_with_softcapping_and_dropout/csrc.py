CUDA_SRC = r"""
#include <torch/extension.h>
#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>

typedef unsigned short u16;

typedef unsigned int uint4n __attribute__((ext_vector_type(4)));

union V16 { uint4n u4; u16 s[8]; };

__device__ __forceinline__ float bf2f(u16 u) {
  return __uint_as_float(((unsigned int)u) << 16);
}

__device__ __forceinline__ u16 f2bf(float f) {
  unsigned int u = __float_as_uint(f);
  return (u16)((u + 0x7fffu + ((u >> 16) & 1u)) >> 16);
}

__device__ __forceinline__ float rbf(float f) { return bf2f(f2bf(f)); }

// tanh, accurate to far better than bf16 precision, built from the hardware
// v_exp_f32 instead of the (much slower) libm tanhf.
//   |s| <  2^-5 : tanh(s) rounds to s in bf16 (the error s^3/3 sits well
//                 under the half-ulp boundary), and this branch also avoids
//                 catastrophic cancellation in (1-e)/(1+e).
//   |s| >= 2^-5 : tanh|s| = (1 - e)/(1 + e) with e = exp(-2|s|).
__device__ __forceinline__ float ftanh(float s) {
  float a = __builtin_fabsf(s);
  float e = __expf(-2.0f * a);
  float t = (1.0f - e) / (1.0f + e);
  t = (a < 0.03125f) ? a : t;
  return __builtin_copysignf(t, s);
}

// Bit-exact emulation of the reference chain, which keeps bf16 dtype at
// every step:  (x / 30.0) -> tanh -> (* 30.0), each rounded to bf16.
__device__ __forceinline__ float softcap(u16 xb) {
  float s = rbf(bf2f(xb) * (1.0f / 30.0f));
  float t = rbf(ftanh(s));
  return rbf(t * 30.0f);
}

// exp(softcap(x)).  softcap is bounded to [-30, 30], so exp never overflows
// fp32 -- the softmax max-subtraction pass is therefore unnecessary.
__device__ __forceinline__ float sc_exp(u16 xb) {
  return __expf(softcap(xb));
}

template <int TPR>
__device__ __forceinline__ float grp_sum(float v) {
#pragma unroll
  for (int o = TPR / 2; o >= 1; o >>= 1) v += __shfl_xor(v, o, 64);
  return v;
}

// ---- fast path: N % 8 == 0, 16-byte vector loads --------------------------
template <int TPR, int NC>
__global__ __launch_bounds__(256) void k_vec8(const u16* __restrict__ X,
                                              u16* __restrict__ Y,
                                              long n_rows, int N) {
  constexpr int RPB = 256 / TPR;
  const int lane = threadIdx.x & (TPR - 1);
  const long row = (long)blockIdx.x * RPB + (threadIdx.x / TPR);
  if (row >= n_rows) return;
  const long off = row * (long)N;
  const u16* xr = X + off;
  u16* yr = Y + off;

  float v[NC * 8];
  float sum = 0.f;
#pragma unroll
  for (int j = 0; j < NC; ++j) {
    const int base = (j * TPR + lane) * 8;
    if (base < N) {
      V16 t;
      t.u4 = *(const uint4n*)(xr + base);
#pragma unroll
      for (int e = 0; e < 8; ++e) {
        float g = sc_exp(t.s[e]);
        v[j * 8 + e] = g;
        sum += g;
      }
    }
  }
  sum = grp_sum<TPR>(sum);
  const float inv = 1.0f / sum;

#pragma unroll
  for (int j = 0; j < NC; ++j) {
    const int base = (j * TPR + lane) * 8;
    if (base < N) {
      V16 t;
#pragma unroll
      for (int e = 0; e < 8; ++e) t.s[e] = f2bf(v[j * 8 + e] * inv);
      __builtin_nontemporal_store(t.u4, (uint4n*)(yr + base));
    }
  }
}

// ---- scalar path: arbitrary N, one wavefront per row ----------------------
template <int VPT>
__global__ __launch_bounds__(256) void k_sc(const u16* __restrict__ X,
                                            u16* __restrict__ Y,
                                            long n_rows, int N) {
  const int lane = threadIdx.x & 63;
  const long row = (long)blockIdx.x * 4 + (threadIdx.x >> 6);
  if (row >= n_rows) return;
  const long off = row * (long)N;
  const u16* xr = X + off;
  u16* yr = Y + off;

  float v[VPT];
  float sum = 0.f;
#pragma unroll
  for (int j = 0; j < VPT; ++j) {
    const int idx = j * 64 + lane;
    float g = 0.f;
    if (idx < N) { g = sc_exp(xr[idx]); sum += g; }
    v[j] = g;
  }
  sum = grp_sum<64>(sum);
  const float inv = 1.0f / sum;

#pragma unroll
  for (int j = 0; j < VPT; ++j) {
    const int idx = j * 64 + lane;
    if (idx < N) yr[idx] = f2bf(v[j] * inv);
  }
}

// ---- generic fallback: two passes over memory, any N ---------------------
__global__ __launch_bounds__(256) void k_gen(const u16* __restrict__ X,
                                             u16* __restrict__ Y,
                                             long n_rows, int N) {
  const long row = blockIdx.x;
  if (row >= n_rows) return;
  const long off = row * (long)N;
  const u16* xr = X + off;
  u16* yr = Y + off;
  const int tid = threadIdx.x;

  __shared__ float red[256];
  float sum = 0.f;
  for (int i = tid; i < N; i += 256) sum += sc_exp(xr[i]);
  red[tid] = sum;
  __syncthreads();
  for (int s = 128; s > 0; s >>= 1) {
    if (tid < s) red[tid] += red[tid + s];
    __syncthreads();
  }
  const float inv = 1.0f / red[0];
  for (int i = tid; i < N; i += 256) yr[i] = f2bf(sc_exp(xr[i]) * inv);
}

// ---- debug hook: softcap only, for exhaustive bit-exactness validation ----
__global__ void k_dbg(const u16* __restrict__ X, u16* __restrict__ Y, long n) {
  long i = (long)blockIdx.x * 256 + threadIdx.x;
  if (i < n) Y[i] = f2bf(softcap(X[i]));
}

#define LAUNCH_VEC8(TPR_, NC_)                                                \
  {                                                                           \
    constexpr int RPB = 256 / (TPR_);                                         \
    long blocks = (n_rows + RPB - 1) / RPB;                                   \
    hipLaunchKernelGGL((k_vec8<TPR_, NC_>), dim3(blocks), dim3(256), 0,       \
                       stream, xp, yp, n_rows, N);                            \
    return y;                                                                 \
  }

#define LAUNCH_SC(VPT_)                                                       \
  {                                                                           \
    long blocks = (n_rows + 3) / 4;                                           \
    hipLaunchKernelGGL((k_sc<VPT_>), dim3(blocks), dim3(256), 0, stream, xp,  \
                       yp, n_rows, N);                                        \
    return y;                                                                 \
  }

at::Tensor run(const at::Tensor& x_) {
  at::Tensor x = x_.is_contiguous() ? x_ : x_.contiguous();
  at::Tensor y = at::empty_like(x);
  const int N = (int)x.size(-1);
  const long n_rows = (long)(x.numel() / (N > 0 ? N : 1));
  if (n_rows == 0 || N == 0) return y;

  const u16* xp = (const u16*)x.data_ptr();
  u16* yp = (u16*)y.data_ptr();
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  if (N % 8 == 0) {
    const int need = N / 8;  // 16B vectors per row
    if (need <= 8) LAUNCH_VEC8(8, 1)
    if (need <= 16) LAUNCH_VEC8(16, 1)
    if (need <= 32) LAUNCH_VEC8(32, 1)
    if (need <= 64) LAUNCH_VEC8(64, 1)
    const int nc = (need + 63) / 64;
    if (nc <= 2) LAUNCH_VEC8(64, 2)
    if (nc <= 3) LAUNCH_VEC8(64, 3)
    if (nc <= 4) LAUNCH_VEC8(64, 4)
    if (nc <= 6) LAUNCH_VEC8(64, 6)
    if (nc <= 8) LAUNCH_VEC8(64, 8)
  } else {
    const int vpt = (N + 63) / 64;
    if (vpt <= 1) LAUNCH_SC(1)
    if (vpt <= 2) LAUNCH_SC(2)
    if (vpt <= 4) LAUNCH_SC(4)
    if (vpt <= 6) LAUNCH_SC(6)
    if (vpt <= 8) LAUNCH_SC(8)
    if (vpt <= 12) LAUNCH_SC(12)
    if (vpt <= 16) LAUNCH_SC(16)
    if (vpt <= 24) LAUNCH_SC(24)
    if (vpt <= 32) LAUNCH_SC(32)
  }

  long blocks = n_rows;
  hipLaunchKernelGGL(k_gen, dim3(blocks), dim3(256), 0, stream, xp, yp, n_rows, N);
  return y;
}

at::Tensor softcap_dbg(const at::Tensor& x) {
  at::Tensor y = at::empty_like(x);
  long n = x.numel();
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  hipLaunchKernelGGL(k_dbg, dim3((n + 255) / 256), dim3(256), 0, stream,
                     (const u16*)x.data_ptr(), (u16*)y.data_ptr(), n);
  return y;
}
"""

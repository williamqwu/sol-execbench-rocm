import torch, time, os, sys
from torch.utils.cpp_extension import load_inline

src = r'''
#include <hip/hip_runtime.h>
#include <cstdint>

template<int BLOCK, int VEC, int ITERS>
__global__ __launch_bounds__(BLOCK) void gk(const short* __restrict__ X,
                                            short* __restrict__ Y,
                                            float mult, float inv) {
  const int N = BLOCK * VEC * ITERS;
  const long base = (long)blockIdx.x * N;
  const int tid = threadIdx.x;
  float s = 0.f, s2 = 0.f;
  using V = __attribute__((ext_vector_type(VEC))) short;
  float regs[ITERS][VEC];
  #pragma unroll
  for (int it = 0; it < ITERS; ++it) {
    V v = *(const V*)(X + base + (long)(it * BLOCK + tid) * VEC);
    #pragma unroll
    for (int j = 0; j < VEC; ++j) {
      unsigned int bits = ((unsigned int)(unsigned short)v[j]) << 16;
      float f; __builtin_memcpy(&f, &bits, 4);
      regs[it][j] = f;
      s += f; s2 = fmaf(f, f, s2);
    }
  }
  __shared__ float sm[2][BLOCK/64];
  #pragma unroll
  for (int o = 32; o >= 1; o >>= 1) {
    s  += __shfl_down(s, o, 64);
    s2 += __shfl_down(s2, o, 64);
  }
  if ((tid & 63) == 0) { sm[0][tid >> 6] = s; sm[1][tid >> 6] = s2; }
  __syncthreads();
  float ts = 0.f, ts2 = 0.f;
  #pragma unroll
  for (int w = 0; w < BLOCK/64; ++w) { ts += sm[0][w]; ts2 += sm[1][w]; }
  float mean = ts * inv;
  float var = fmaf(-mean, mean, ts2 * inv);
  var = var > 0.f ? var : 0.f;
  float thr = fmaf(sqrtf(var), mult, mean);
  #pragma unroll
  for (int it = 0; it < ITERS; ++it) {
    V o;
    #pragma unroll
    for (int j = 0; j < VEC; ++j) {
      float f = regs[it][j] - thr;
      if (!(f > 0.f)) f = 0.f;
      unsigned int bits; __builtin_memcpy(&bits, &f, 4);
      bits += ((bits >> 16) & 1u) + 0x7fffu;
      o[j] = (short)(unsigned short)(bits >> 16);
    }
    *(V*)(Y + base + (long)(it * BLOCK + tid) * VEC) = o;
  }
}

// two-pass: reload from cache instead of holding registers
template<int BLOCK, int VEC, int ITERS>
__global__ __launch_bounds__(BLOCK) void gk2(const short* __restrict__ X,
                                             short* __restrict__ Y,
                                             float mult, float inv) {
  const int N = BLOCK * VEC * ITERS;
  const long base = (long)blockIdx.x * N;
  const int tid = threadIdx.x;
  float s = 0.f, s2 = 0.f;
  using V = __attribute__((ext_vector_type(VEC))) short;
  #pragma unroll
  for (int it = 0; it < ITERS; ++it) {
    V v = *(const V*)(X + base + (long)(it * BLOCK + tid) * VEC);
    #pragma unroll
    for (int j = 0; j < VEC; ++j) {
      unsigned int bits = ((unsigned int)(unsigned short)v[j]) << 16;
      float f; __builtin_memcpy(&f, &bits, 4);
      s += f; s2 = fmaf(f, f, s2);
    }
  }
  __shared__ float sm[2][BLOCK/64];
  #pragma unroll
  for (int o = 32; o >= 1; o >>= 1) { s += __shfl_down(s,o,64); s2 += __shfl_down(s2,o,64); }
  if ((tid & 63) == 0) { sm[0][tid>>6] = s; sm[1][tid>>6] = s2; }
  __syncthreads();
  float ts=0.f, ts2=0.f;
  #pragma unroll
  for (int w = 0; w < BLOCK/64; ++w) { ts += sm[0][w]; ts2 += sm[1][w]; }
  float mean = ts * inv;
  float var = fmaf(-mean, mean, ts2 * inv); var = var>0.f?var:0.f;
  float thr = fmaf(sqrtf(var), mult, mean);
  #pragma unroll
  for (int it = 0; it < ITERS; ++it) {
    V v = *(const V*)(X + base + (long)(it*BLOCK+tid)*VEC);
    V o;
    #pragma unroll
    for (int j = 0; j < VEC; ++j) {
      unsigned int bits = ((unsigned int)(unsigned short)v[j]) << 16;
      float f; __builtin_memcpy(&f,&bits,4);
      f -= thr; if (!(f>0.f)) f = 0.f;
      unsigned int b2; __builtin_memcpy(&b2,&f,4);
      b2 += ((b2>>16)&1u)+0x7fffu;
      o[j] = (short)(unsigned short)(b2>>16);
    }
    *(V*)(Y + base + (long)(it*BLOCK+tid)*VEC) = o;
  }
}

#define INST(B,V,I) \
void l_##B##_##V##_##I(int64_t xp,int64_t yp,double mult,int64_t rows,int64_t st){ \
  hipLaunchKernelGGL((gk<B,V,I>), dim3(rows), dim3(B), 0, (hipStream_t)st, \
   (const short*)xp,(short*)yp,(float)mult, 1.0f/(float)(B*V*I)); } \
void m_##B##_##V##_##I(int64_t xp,int64_t yp,double mult,int64_t rows,int64_t st){ \
  hipLaunchKernelGGL((gk2<B,V,I>), dim3(rows), dim3(B), 0, (hipStream_t)st, \
   (const short*)xp,(short*)yp,(float)mult, 1.0f/(float)(B*V*I)); }

INST(64,8,8) INST(128,8,4) INST(256,8,2) INST(512,8,1) INST(256,4,4) INST(128,4,8)
INST(128,8,8) INST(256,8,4) INST(512,8,2) INST(1024,8,1) INST(256,4,8)
INST(128,8,12) INST(256,8,6) INST(512,8,3) INST(384,8,4) INST(768,8,2) INST(192,8,8)
INST(256,8,8) INST(512,8,4) INST(1024,8,2) INST(128,8,16)
'''

names = []
for b,v,i in [(64,8,8),(128,8,4),(256,8,2),(512,8,1),(256,4,4),(128,4,8),
              (128,8,8),(256,8,4),(512,8,2),(1024,8,1),(256,4,8),
              (128,8,12),(256,8,6),(512,8,3),(384,8,4),(768,8,2),(192,8,8),
              (256,8,8),(512,8,4),(1024,8,2),(128,8,16)]:
    names += [f'l_{b}_{v}_{i}', f'm_{b}_{v}_{i}']
cpp = '\n'.join(f'void {n}(int64_t,int64_t,double,int64_t,int64_t);' for n in names)

bd = '/var/tmp/solbench/agent/pilot8/L1__053_gaussian_topk_sparse_activation/hipbuild3'
os.makedirs(bd, exist_ok=True)
mod = load_inline(name='gtk_bench', cpp_sources=cpp, cuda_sources=src, functions=names,
                  extra_cuda_cflags=['-O3','--offload-arch=gfx950'], verbose=False, build_directory=bd)

SHAPES = [(1,512,12288),(4,2048,12288),(32,128,12288),(2,211,8192),(1,8192,4096),
          (1,1024,16384),(16,1163,8192),(4,541,8192),(4,449,4096),(64,1024,8192),
          (2,131,4096),(2,293,12288)]

CFG = {4096:[(64,8,8),(128,8,4),(256,8,2),(512,8,1),(256,4,4),(128,4,8)],
       8192:[(128,8,8),(256,8,4),(512,8,2),(1024,8,1),(256,4,8),(64,8,16) if False else (128,8,8)],
       12288:[(128,8,12),(256,8,6),(512,8,3),(384,8,4),(768,8,2),(192,8,8)],
       16384:[(256,8,8),(512,8,4),(1024,8,2),(128,8,16)]}

def bench_gpu(f,n=200):
    for _ in range(30): f()
    torch.cuda.synchronize()
    e1,e2=torch.cuda.Event(True),torch.cuda.Event(True); e1.record()
    for _ in range(n): f()
    e2.record(); torch.cuda.synchronize()
    return e1.elapsed_time(e2)/n*1000

st = torch.cuda.current_stream().cuda_stream
mult = -1.2815515
for (B,S,N) in SHAPES:
    rows = B*S
    x = torch.randn(rows, N, device='cuda', dtype=torch.bfloat16)
    y = torch.empty_like(x)
    xp, yp = x.data_ptr(), y.data_ptr()
    byt = rows*N*4
    res = []
    for (b,v,i) in dict.fromkeys(CFG[N]):
        for pre in ('l','m'):
            fn = getattr(mod, f'{pre}_{b}_{v}_{i}')
            t = bench_gpu(lambda: fn(xp,yp,mult,rows,st))
            res.append((t, f"{pre}{b}v{v}i{i}"))
    res.sort()
    print(f"{B}x{S}x{N} rows={rows}: " + " | ".join(f"{n}={t:.2f}us({byt/t/1e6:.2f}TB/s)" for t,n in res[:5]))
    sys.stdout.flush()

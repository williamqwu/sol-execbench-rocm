import torch, time, os
from torch.utils.cpp_extension import load_inline

src = r'''
#include <torch/extension.h>
#include <hip/hip_runtime.h>

template<int BLOCK, int VEC>
__global__ __launch_bounds__(BLOCK) void gk(const short* __restrict__ X,
                                            short* __restrict__ Y,
                                            float mult, int N) {
  const long row = blockIdx.x;
  const long base = row * (long)N;
  const int tid = threadIdx.x;
  const int nvec = N / VEC;
  const int iters = nvec / BLOCK;
  float s = 0.f, s2 = 0.f;
  using V = __attribute__((ext_vector_type(VEC))) short;
  float regs[8][VEC];
  #pragma unroll 1
  for (int it = 0; it < iters; ++it) {
    int idx = it * BLOCK + tid;
    V v = *(const V*)(X + base + (long)idx * VEC);
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
  int wave = tid / 64, lane = tid % 64;
  if (lane == 0) { sm[0][wave] = s; sm[1][wave] = s2; }
  __syncthreads();
  float ts = 0.f, ts2 = 0.f;
  #pragma unroll
  for (int w = 0; w < BLOCK/64; ++w) { ts += sm[0][w]; ts2 += sm[1][w]; }
  float inv = 1.0f / (float)N;
  float mean = ts * inv;
  float var = fmaf(-mean, mean, ts2 * inv);
  var = var > 0.f ? var : 0.f;
  float thr = fmaf(sqrtf(var), mult, mean);

  #pragma unroll 1
  for (int it = 0; it < iters; ++it) {
    int idx = it * BLOCK + tid;
    V o;
    #pragma unroll
    for (int j = 0; j < VEC; ++j) {
      float f = regs[it][j] - thr;
      if (!(f > 0.f)) f = 0.f;
      unsigned int bits; __builtin_memcpy(&bits, &f, 4);
      unsigned int r = ((bits >> 16) & 1u) + 0x7fffu;
      bits += r;
      o[j] = (short)(unsigned short)(bits >> 16);
    }
    *(V*)(Y + base + (long)idx * VEC) = o;
  }
}

void launch(int64_t xp, int64_t yp, double mult, int64_t rows, int64_t N, int64_t stream) {
  hipLaunchKernelGGL((gk<256,8>), dim3(rows), dim3(256), 0, (hipStream_t)stream,
     (const short*)xp, (short*)yp, (float)mult, (int)N);
}

'''

cpp = 'void launch(int64_t xp, int64_t yp, double mult, int64_t rows, int64_t N, int64_t stream);'

bd = '/var/tmp/solbench/agent/pilot8/L1__053_gaussian_topk_sparse_activation/hipbuild'
os.makedirs(bd, exist_ok=True)
mod = load_inline(name='gtk_hip', cpp_sources=cpp, cuda_sources=src, functions=['launch'],
                  extra_cuda_cflags=['-O3'], verbose=False, build_directory=bd)

rows, N = 262, 4096
x = torch.randn(rows, N, device='cuda', dtype=torch.bfloat16)
y = torch.empty_like(x)
mult = -1.28
st = torch.cuda.current_stream().cuda_stream
mod.launch(x.data_ptr(), y.data_ptr(), mult, rows, N, st)
torch.cuda.synchronize()
xf = x.float()
ref = torch.clamp(xf - (xf.mean(-1,keepdim=True) + xf.std(-1,keepdim=True,unbiased=False)*mult), min=0).bfloat16()
print("max diff", (y.float()-ref.float()).abs().max().item())

def bench_cpu(f, n=1000):
    for _ in range(50): f()
    torch.cuda.synchronize()
    t0=time.perf_counter()
    for _ in range(n): f()
    torch.cuda.synchronize()
    return (time.perf_counter()-t0)/n*1e6
def bench_gpu(f,n=300):
    for _ in range(50): f()
    torch.cuda.synchronize()
    e1,e2=torch.cuda.Event(True),torch.cuda.Event(True); e1.record()
    for _ in range(n): f()
    e2.record(); torch.cuda.synchronize()
    return e1.elapsed_time(e2)/n*1000
g = lambda: mod.launch(x.data_ptr(), y.data_ptr(), mult, rows, N, st)
print("hip cpu=%.2f gpu=%.2f" % (bench_cpu(g), bench_gpu(g)))

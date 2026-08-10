import torch
from typing import Tuple
from torch.utils.cpp_extension import load_inline

_cpp = r'''
#include <torch/extension.h>
void rope_hip(torch::Tensor ids, torch::Tensor ig, torch::Tensor vg,
              torch::Tensor secs, torch::Tensor mask, torch::Tensor out,
              torch::Tensor delta);
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) { m.def("run", &rope_hip); }
'''

_hip = r'''
#include <torch/extension.h>
#include <hip/hip_runtime.h>

__global__ void rope_kernel(const long* ids, const long* ig, const long* vg,
 const float* secs, const long* mask, long* out, long* delta,
 int B, int S, int NI, int NV) {
  int b=blockIdx.x, tid=threadIdx.x;
  extern __shared__ long tok[];
  __shared__ int n, img0, vid0;
  // Reference initializes every output element to one.
  for(int p=tid;p<S;p+=blockDim.x) {
    out[(0*B+b)*S+p]=1; out[(1*B+b)*S+p]=1; out[(2*B+b)*S+p]=1;
  }
  if(tid==0) {
    n=0;
    for(int p=0;p<S;p++) if(mask[b*S+p]==1) tok[n++]=ids[b*S+p];
    img0=vid0=0;
    // Global metadata indices are ordered by batch, then occurrence.
    for(int q=0;q<b;q++) for(int p=0;p<S-1;p++)
      if(mask[q*S+p]==1 && ids[q*S+p]==151652) {
        int z=p+1; while(z<S && mask[q*S+z]!=1) z++;
        if(z<S && ids[q*S+z]==151655) img0++;
        else if(z<S && ids[q*S+z]==151656) vid0++;
      }
  }
  __syncthreads();
  if(NI==0 && NV==0) {
    for(int p=tid;p<S;p+=blockDim.x) {
      long v=1;
      if(mask[b*S+p]!=0) { v=-1; for(int q=0;q<=p;q++) v += mask[b*S+q]; }
      out[(0*B+b)*S+p]=v; out[(1*B+b)*S+p]=v; out[(2*B+b)*S+p]=v;
    }
    if(tid==0) delta[b]=0;
    return;
  }
  __shared__ int st, base, ii, vi, done;
  if(tid==0) { st=0;base=0;ii=img0;vi=vid0;done=0; }
  __syncthreads();
  while(!done) {
    __shared__ int ed, nt, hh, ww, isvid;
    __shared__ float sec;
    if(tid==0) {
      ed=n; isvid=0;
      for(int p=st;p<n;p++) if(tok[p]==151655 || tok[p]==151656) { ed=p; isvid=(tok[p]==151656); break; }
      if(ed==n) { nt=0; hh=ww=1; sec=0; done=1; }
      else if(isvid && vi<NV) { nt=vg[vi*3];hh=vg[vi*3+1]/2;ww=vg[vi*3+2]/2;sec=secs[vi++]; }
      else if(!isvid && ii<NI) { nt=ig[ii*3];hh=ig[ii*3+1]/2;ww=ig[ii*3+2]/2;sec=0;ii++; }
      else { done=1; nt=0; hh=ww=1; sec=0; }
    }
    __syncthreads();
    int text=ed-st;
    for(int k=tid;k<text;k+=blockDim.x) {
      long v=base+k; int p=st+k;
      out[(0*B+b)*S+p]=v;out[(1*B+b)*S+p]=v;out[(2*B+b)*S+p]=v;
    }
    if(!done) {
      int cnt=nt*hh*ww;
      for(int k=tid;k<cnt;k+=blockDim.x) {
        int tt=k/(hh*ww), rem=k%(hh*ww), p=ed+k;
        out[(0*B+b)*S+p]=base+text+(long)(tt*sec*2.0f);
        out[(1*B+b)*S+p]=base+text+rem/ww;
        out[(2*B+b)*S+p]=base+text+rem%ww;
      }
      __syncthreads();
      if(tid==0) { int mx=nt>0 ? (int)((nt-1)*sec*2.0f) : 0; if(hh-1>mx)mx=hh-1;if(ww-1>mx)mx=ww-1; base += text+mx+1; st=ed+cnt; }
    } else {
      __syncthreads();
      if(tid==0) { base += text; st=n; }
    }
    __syncthreads();
  }
  if(tid==0) delta[b]=base-n;
}

void rope_hip(torch::Tensor ids, torch::Tensor ig, torch::Tensor vg,
 torch::Tensor secs, torch::Tensor mask, torch::Tensor out, torch::Tensor delta) {
 int B=ids.size(0), S=ids.size(1);
 hipLaunchKernelGGL(rope_kernel, dim3(B), dim3(256), S*sizeof(long), 0,
  ids.data_ptr<long>(),ig.data_ptr<long>(),vg.data_ptr<long>(),secs.data_ptr<float>(),
  mask.data_ptr<long>(),out.data_ptr<long>(),delta.data_ptr<long>(),B,S,ig.size(0),vg.size(0));
}
'''

_ext = load_inline(name='mrope_fast_ext', cpp_sources=_cpp, cuda_sources=_hip,
                   functions=None, extra_cuda_cflags=['-O3'], verbose=False)

@torch.no_grad()
def run(input_ids: torch.Tensor, image_grid_thw: torch.Tensor,
        video_grid_thw: torch.Tensor, second_per_grid_ts: torch.Tensor,
        attention_mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    B,S=input_ids.shape
    out=torch.empty((3,B,S),dtype=torch.int64,device=input_ids.device)
    delta=torch.empty((B,1),dtype=torch.int64,device=input_ids.device)
    _ext.run(input_ids,image_grid_thw,video_grid_thw,second_per_grid_ts,attention_mask,out,delta)
    return out,delta

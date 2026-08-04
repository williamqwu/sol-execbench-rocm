import torch, triton, triton.language as tl, time
import reference, kern_impl
dev=torch.device("cuda:0")
def tm(fn,it=100):
    for _ in range(20): fn()
    torch.cuda.synchronize()
    s=torch.cuda.Event(True);e=torch.cuda.Event(True);s.record()
    for _ in range(it): fn()
    e.record();torch.cuda.synchronize()
    return s.elapsed_time(e)/it*1e3

# achievable BW calibration: copy 256MB
n=128*1024*1024
x=torch.randn(n,dtype=torch.bfloat16,device=dev); y=torch.empty_like(x)
t=tm(lambda: y.copy_(x))
print(f"copy 256MB r+w: {t:.1f}us -> {2*n*2/t*1e6/1e12:.2f} TB/s")

@triton.jit
def _empty(): pass
t=tm(lambda: _empty[(1,)]())
print(f"empty triton launch: {t:.2f}us")

# random-row gather BW
d=reference.get_inputs(dict(batch_size=1,seq_len=8192,hidden_size=3072,num_experts_per_tok=8),dev)
a=(d['final_hidden_states'],d['expert_outputs'],d['token_indices'])
N,H=a[0].shape; M=a[2].shape[0]
cnt=torch.bincount(a[2],minlength=N).int()
print("maxcnt",cnt.max().item())
slots=torch.empty(N*32,dtype=torch.int32,device=dev)
ovf=torch.zeros(2*M,dtype=torch.int32,device=dev); novf=torch.zeros(1,dtype=torch.int32,device=dev)
out=torch.empty_like(a[0])
c2=torch.zeros(N,dtype=torch.int32,device=dev)
kern_impl._build[(triton.cdiv(M,1024),)](a[2],c2,slots,ovf,novf,M,C=32,BLOCK=1024,num_warps=4)
print("build 1024/w4", tm(lambda: kern_impl._build[(triton.cdiv(M,1024),)](a[2],torch.zeros(N,dtype=torch.int32,device=dev),slots,ovf,torch.zeros(1,dtype=torch.int32,device=dev),M,C=32,BLOCK=1024,num_warps=4)))
for BH,KB,nw in [(1024,16,4),(1024,32,4),(512,32,4),(3072,16,8),(1536,16,8),(1024,16,8),(768,16,4),(512,16,4),(1024,8,4)]:
    NH=triton.cdiv(H,BH)
    try:
        t=tm(lambda: kern_impl._gather[(N,NH)](a[0],a[1],c2,slots,ovf,novf,out,H=H,C=32,NH=NH,BLOCK_H=BH,KB=KB,num_warps=nw))
        print(f"gather BH={BH} KB={KB} w={nw}: {t:.1f}us  eff={10*N*H*2/t*1e6/1e12:.2f} TB/s")
    except Exception as ex: print("  fail",BH,KB,nw,str(ex)[:100])

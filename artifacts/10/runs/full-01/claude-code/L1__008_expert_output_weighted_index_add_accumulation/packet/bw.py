import torch, triton, triton.language as tl, itertools
dev=torch.device("cuda:0")
def tm(fn,it=50):
    for _ in range(10): fn()
    torch.cuda.synchronize(); s=torch.cuda.Event(True);e=torch.cuda.Event(True);s.record()
    for _ in range(it): fn()
    e.record();torch.cuda.synchronize(); return s.elapsed_time(e)/it*1e3

@triton.jit
def cp(x,y,n,BLOCK:tl.constexpr,CM:tl.constexpr):
    p=tl.program_id(0); o=p*BLOCK+tl.arange(0,BLOCK); m=o<n
    v=tl.load(x+o,mask=m,cache_modifier=CM)
    tl.store(y+o,v,mask=m,cache_modifier=CM)

nbytes=512*1024*1024
n=nbytes//2
x=torch.randn(n,dtype=torch.bfloat16,device=dev); y=torch.empty_like(x)
print("torch copy_:", round(tm(lambda:y.copy_(x)),1), "us ->", round(2*nbytes/tm(lambda:y.copy_(x))*1e6/1e12,2),"TB/s")
best=[]
for B,nw,cm in itertools.product([1024,2048,4096,8192,16384],[1,2,4,8],["", ".cg",".cs" ]):
    try:
        t=tm(lambda: cp[(triton.cdiv(n,B),)](x,y,n,BLOCK=B,CM=cm,num_warps=nw))
        best.append((t,B,nw,cm))
    except Exception as ex: pass
best.sort()
for t,B,nw,cm in best[:8]: print(f"  B={B} w={nw} cm='{cm}': {t:.1f}us {2*nbytes/t*1e6/1e12:.2f} TB/s")

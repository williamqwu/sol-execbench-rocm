import torch, time, triton
import reference, kern_impl
dev=torch.device("cuda:0")
def tm(fn,it=200):
    for _ in range(20): fn()
    torch.cuda.synchronize()
    s=torch.cuda.Event(True);e=torch.cuda.Event(True);s.record()
    for _ in range(it): fn()
    e.record();torch.cuda.synchronize()
    g=s.elapsed_time(e)/it*1e3
    torch.cuda.synchronize(); t=time.perf_counter()
    for _ in range(it): fn()
    torch.cuda.synchronize()
    cpu=(time.perf_counter()-t)/it*1e6
    return round(g,1), round(cpu,1)

for bs,sl in [(1,131),(1,8192)]:
    d=reference.get_inputs(dict(batch_size=bs,seq_len=sl,hidden_size=3072,num_experts_per_tok=8),dev)
    a=(d['final_hidden_states'],d['expert_outputs'],d['token_indices'])
    N,H=a[0].shape; M=a[2].shape[0]
    print(bs,sl,"full", tm(lambda: kern_impl.run_impl(*a,BLOCK_H=1024,KB=16)))
    print("   allocs", tm(lambda: (torch.zeros(N+1,dtype=torch.int32,device=dev),torch.empty(N*32,dtype=torch.int32,device=dev),torch.empty(2*M,dtype=torch.int32,device=dev),torch.empty_like(a[0]))))
    print("   clone ", tm(lambda: a[0].clone()))
    # count distribution
    cnt=torch.bincount(a[2],minlength=N)
    print("   maxcnt",cnt.max().item(),"mean",cnt.float().mean().item())

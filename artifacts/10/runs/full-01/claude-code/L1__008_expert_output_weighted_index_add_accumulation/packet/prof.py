import torch, time
import reference, kern_impl, triton
dev=torch.device("cuda:0")
d=reference.get_inputs(dict(batch_size=1,seq_len=131,hidden_size=3072,num_experts_per_tok=8),dev)
a=(d['final_hidden_states'],d['expert_outputs'],d['token_indices'])
def tm(fn,it=100):
    for _ in range(10): fn()
    torch.cuda.synchronize(); t=time.perf_counter()
    for _ in range(it): fn()
    cpu=(time.perf_counter()-t)/it*1e6
    torch.cuda.synchronize()
    s=torch.cuda.Event(True);e=torch.cuda.Event(True);s.record()
    for _ in range(it): fn()
    e.record();torch.cuda.synchronize()
    return s.elapsed_time(e)/it*1e3, cpu
print("full        ", tm(lambda: kern_impl.run_impl(*a,BLOCK_H=1024,KB=16)))
N,H=a[0].shape; M=a[2].shape[0]
print("allocs      ", tm(lambda: (torch.zeros(N+1,dtype=torch.int32,device=dev),torch.empty(N*32,dtype=torch.int32,device=dev),torch.empty(2*M,dtype=torch.int32,device=dev),torch.empty_like(a[0]))))
buf=torch.zeros(N+1,dtype=torch.int32,device=dev); slots=torch.empty(N*32,dtype=torch.int32,device=dev)
ovf=torch.empty(2*M,dtype=torch.int32,device=dev); out=torch.empty_like(a[0])
print("build only  ", tm(lambda: kern_impl._build[(triton.cdiv(M,1024),)](a[2],buf[:N],slots,ovf,buf[N:],M,C=32,BLOCK=1024,num_warps=4)))
print("gather only ", tm(lambda: kern_impl._gather[(N,3)](a[0],a[1],buf[:N],slots,ovf,buf[N:],out,H=H,C=32,NH=3,BLOCK_H=1024,KB=16,num_warps=4)))
print("clone       ", tm(lambda: a[0].clone()))

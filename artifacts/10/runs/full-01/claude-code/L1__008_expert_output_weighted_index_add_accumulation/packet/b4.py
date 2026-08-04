import torch, itertools, triton
import reference, k2
dev=torch.device("cuda:0")
def mk(bs,sl): return reference.get_inputs(dict(batch_size=bs,seq_len=sl,hidden_size=3072,num_experts_per_tok=8),dev)
def tm(fn,it=50):
    for _ in range(10): fn()
    torch.cuda.synchronize(); s=torch.cuda.Event(True);e=torch.cuda.Event(True);s.record()
    for _ in range(it): fn()
    e.record();torch.cuda.synchronize(); return s.elapsed_time(e)/it*1e3
for bs,sl in [(1,131),(2,1024),(1,8192)]:
    d=mk(bs,sl); f,e,i=d['final_hidden_states'],d['expert_outputs'],d['token_indices']
    N,H=f.shape; M=i.shape[0]
    print(f"=== N={N} M={M}")
    print("  torch.zeros(N+1)", round(tm(lambda: torch.zeros(N+1,dtype=torch.int32,device=dev)),2))
    print("  torch.empty(N*32)", round(tm(lambda: torch.empty(N*32,dtype=torch.int32,device=dev)),2))
    print("  empty_like(f)", round(tm(lambda: torch.empty_like(f)),2))
    print("  f.clone()", round(tm(lambda: f.clone()),2))
    buf=torch.zeros(N+1,dtype=torch.int32,device=dev); slots=torch.empty(N*32,dtype=torch.int32,device=dev)
    ovf=torch.empty(2*M,dtype=torch.int32,device=dev); out=torch.empty_like(f)
    best=[]
    for BB,bnw in itertools.product([64,128,256,512,1024],[1,2,4,8]):
        try:
            t=tm(lambda: k2._build2[(triton.cdiv(M,BB),)](i,buf,slots,ovf,buf[N:],M,C=32,BLOCK=BB,num_warps=bnw))
            best.append((t,BB,bnw))
        except Exception as ex: pass
    best.sort(); print("  build best:", [(round(t,1),b,w) for t,b,w in best[:4]])
    g=[]
    for BH,U,nw in itertools.product([1024,1536,3072],[1,2,4,8,16],[2,4,8,16]):
        NH=triton.cdiv(H,BH)
        try:
            t=tm(lambda: k2._gath_s[(N,NH)](f,e,buf,slots,ovf,buf[N:],out,H=H,C=32,BLOCK_H=BH,U=U,num_warps=nw))
            g.append((t,BH,U,nw))
        except Exception as ex: pass
    g.sort(); print("  gather best:", [(round(t,1),b,u,w) for t,b,u,w in g[:5]])

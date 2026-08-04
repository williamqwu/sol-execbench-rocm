import torch, itertools, triton, sys
import reference, k2
dev=torch.device("cuda:0")
def mk(bs,sl): return reference.get_inputs(dict(batch_size=bs,seq_len=sl,hidden_size=3072,num_experts_per_tok=8),dev)
def ref(f,e,i):
    o=f.clone(); o.index_add_(0,i,e); return o
def tm(fn,it=50):
    for _ in range(10): fn()
    torch.cuda.synchronize(); s=torch.cuda.Event(True);e=torch.cuda.Event(True);s.record()
    for _ in range(it): fn()
    e.record();torch.cuda.synchronize(); return s.elapsed_time(e)/it*1e3
def rat(o,r,atol=0.234375,rtol=0.375):
    o=o.float();r=r.float();d=(o-r).abs(); return ((d<=atol)|(d<=rtol*r.abs())).float().mean().item()

for bs,sl in [(1,131),(1,8192),(2,1024)]:
    d=mk(bs,sl); a=(d['final_hidden_states'],d['expert_outputs'],d['token_indices'])
    r=ref(*a); N=bs*sl
    print(f"--- N={N} SOL5={10*N*3072*2/5e12*1e6:.1f}us ref={tm(lambda:ref(*a)):.1f}us")
    res=[]
    for BH,U,BB,nw,bnw in itertools.product([1024,1536,3072],[1,2,4,8],[128,256,512],[4,8],[4]):
        try:
            o=k2.run2(*a,BLOCK_H=BH,U=U,BB=BB,nw=nw,bnw=bnw)
            t=tm(lambda: k2.run2(*a,BLOCK_H=BH,U=U,BB=BB,nw=nw,bnw=bnw))
            res.append((t,f"gath BH{BH} U{U} BB{BB} w{nw}",rat(o,r)))
        except Exception as ex: pass
    for BH,nw in itertools.product([512,1024,1536,3072],[1,2,4,8]):
        try:
            o=k2.run_at(*a,BLOCK_H=BH,nw=nw)
            t=tm(lambda: k2.run_at(*a,BLOCK_H=BH,nw=nw))
            res.append((t,f"ATOM BH{BH} w{nw}",rat(o,r)))
        except Exception as ex: pass
    res.sort()
    for t,n,rr in res[:6]: print(f"   {t:8.1f}us  {n:28s} rat={rr:.5f}")

import torch, itertools, triton
import reference, k3
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
for bs,sl in [(1,131),(1,1024),(2,1024),(1,8192)]:
    d=mk(bs,sl); a=(d['final_hidden_states'],d['expert_outputs'],d['token_indices'])
    r=ref(*a); N=bs*sl
    print(f"=== N={N} SOL6.4={10*N*3072*2/6.4e12*1e6:.1f}us ref={tm(lambda:ref(*a)):.1f}us")
    res=[]
    for BH,nw,RPB in itertools.product([512,1024,1536,3072],[1,2,4,8],[1,2,4]):
        try:
            o=k3.run_at(*a,BLOCK_H=BH,nw=nw,RPB=RPB)
            t=tm(lambda: k3.run_at(*a,BLOCK_H=BH,nw=nw,RPB=RPB))
            res.append((t,BH,nw,RPB,rat(o,r)))
        except Exception as ex: pass
    res.sort()
    for t,BH,nw,RPB,rr in res[:5]: print(f"   {t:7.1f}us BH{BH} w{nw} RPB{RPB} rat={rr:.5f}")

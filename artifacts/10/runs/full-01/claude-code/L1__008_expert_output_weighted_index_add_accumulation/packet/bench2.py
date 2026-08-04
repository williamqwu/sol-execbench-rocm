import torch, itertools, json
import reference, kern_impl
dev = torch.device("cuda:0")
def make(bs, sl):
    return reference.get_inputs(dict(batch_size=bs, seq_len=sl, hidden_size=3072,
                                     num_experts_per_tok=8), dev)
def ref(f,e,i):
    o=f.clone(); o.index_add_(0,i,e); return o
def tm(fn, args, it=30):
    for _ in range(5): fn(*args)
    torch.cuda.synchronize()
    s=torch.cuda.Event(True); en=torch.cuda.Event(True); s.record()
    for _ in range(it): fn(*args)
    en.record(); torch.cuda.synchronize()
    return s.elapsed_time(en)/it*1000
def ratio(o,r,atol,rtol):
    o=o.float(); r=r.float(); d=(o-r).abs()
    ok=(d<=atol)|(d<=rtol*r.abs())
    return ok.float().mean().item()

wl=[json.loads(l) for l in open('workload.jsonl')]
shapes=sorted({(w['axes']['batch_size']*w['axes']['seq_len']) for w in wl})
print("Ns:",shapes)
for bs,sl in [(1,131),(1,256),(2,256),(1,1024),(2,1024),(4,541),(2,1879),(1,8192),(64,128),(32,256)]:
    d=make(bs,sl); a=(d['final_hidden_states'],d['expert_outputs'],d['token_indices'])
    r=ref(*a); N=bs*sl
    sol=10*N*3072*2/8e12*1e6
    tr=tm(ref,a)
    res=[]
    for BH,KB,nw in itertools.product([256,384,512,768,1024,1536,3072],[4,8,16],[4,8]):
        if BH*KB>32768: continue
        try:
            o=kern_impl.run_impl(*a,BLOCK_H=BH,KB=KB,nw=nw)
            t=tm(lambda *x: kern_impl.run_impl(*x,BLOCK_H=BH,KB=KB,nw=nw), a)
        except Exception as ex: continue
        res.append((t,BH,KB,nw,ratio(o,r,0.234375,0.375)))
    res.sort()
    print(f"N={N:6d} SOL={sol:7.1f} ref={tr:7.1f} | "+" | ".join(f"{t:.1f}us BH{b} KB{k} w{w} rat{rr:.4f}" for t,b,k,w,rr in res[:4]))

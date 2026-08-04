import torch, sys, itertools
import reference, kern_impl
dev = torch.device("cuda:0")
def make(bs, sl):
    return reference.get_inputs(dict(batch_size=bs, seq_len=sl, hidden_size=3072,
                                     num_experts_per_tok=8), dev)
def ref(f,e,i):
    o=f.clone(); o.index_add_(0,i,e); return o
def tm(fn, args, it=50):
    for _ in range(10): fn(*args)
    torch.cuda.synchronize()
    s=torch.cuda.Event(True); en=torch.cuda.Event(True)
    s.record()
    for _ in range(it): fn(*args)
    en.record(); torch.cuda.synchronize()
    return s.elapsed_time(en)/it*1000

shapes=[(2,1024),(1,8192),(1,131),(64,128),(32,256)]
for bs,sl in shapes:
    d=make(bs,sl); a=(d['final_hidden_states'],d['expert_outputs'],d['token_indices'])
    r=ref(*a)
    N=bs*sl; H=3072
    sol = 10*N*H*2/8e12*1e6
    tr=tm(ref,a)
    best=None
    for BH,KB,nw in itertools.product([512,1024,2048],[4,8,16],[4,8]):
        try:
            o=kern_impl.run_impl(*a,BLOCK_H=BH,KB=KB,nw=nw)
        except Exception as ex:
            print("  fail",BH,KB,nw,type(ex).__name__,str(ex)[:80]); continue
        err=(o.float()-r.float()).abs().max().item()
        t=tm(lambda *x: kern_impl.run_impl(*x,BLOCK_H=BH,KB=KB,nw=nw), a)
        if best is None or t<best[0]: best=(t,BH,KB,nw,err)
    print(f"N={N:6d} SOL={sol:8.1f}us ref={tr:8.1f}us  best={best[0]:8.1f}us cfg BH={best[1]} KB={best[2]} nw={best[3]} err={best[4]:.4f}")

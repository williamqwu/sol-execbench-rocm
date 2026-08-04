import torch, time, sys
import reference
dev = torch.device('cuda:0')
names=["grad_hidden_states","grad_fc1_weight","grad_fc1_bias","grad_fc2_weight","grad_fc2_bias","grad_ln_weight","grad_ln_bias"]

def gen(B,S,seed=0):
    torch.manual_seed(seed)
    return reference.get_inputs({"batch_size":B,"seq_len":S,"hidden_size":512,"intermediate_size":2048}, dev)

def compare(out, ref, atol, rtol, label=""):
    worst=1.0; lines=[]
    for n,a,b in zip(names,out,ref):
        a=a.float(); b=b.float()
        err=(a-b).abs()
        tolm = atol + rtol*b.abs()
        matched = (err<=tolm).float().mean().item()
        worst=min(worst,matched)
        lines.append(f"    {'OK ' if matched>=0.99 else 'FAIL'} {n:20s} maxerr={err.max().item():.3e} matched={matched:.5f}")
    print(f"  [{label}] worst_matched={worst:.5f} {'PASS' if worst>=0.99 else 'FAIL'}")
    for l in lines: print(l)
    return worst

def bench(fn, iters=10, warm=3):
    for _ in range(warm): fn()
    torch.cuda.synchronize()
    t=time.perf_counter()
    for _ in range(iters): fn()
    torch.cuda.synchronize()
    return (time.perf_counter()-t)/iters*1e3

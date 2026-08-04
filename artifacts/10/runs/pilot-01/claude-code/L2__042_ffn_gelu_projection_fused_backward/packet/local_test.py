import torch, json, time, importlib, sys
import reference

dev = torch.device('cuda:0')

def gen(B,S,seed=0):
    torch.manual_seed(seed)
    return reference.get_inputs({"batch_size":B,"seq_len":S,"hidden_size":512,"intermediate_size":2048}, dev)

names=["grad_hidden_states","grad_fc1_weight","grad_fc1_bias","grad_fc2_weight","grad_fc2_bias","grad_ln_weight","grad_ln_bias"]

def check(out, ref, atol, rtol, tag=""):
    ok=True
    for n,a,b in zip(names,out,ref):
        a=a.float(); b=b.float()
        err=(a-b).abs()
        tolm = atol + rtol*b.abs()
        matched = (err<=tolm).float().mean().item()
        mx = err.max().item()
        st = "OK " if matched>=0.99 else "FAIL"
        if matched<0.99: ok=False
        print(f"  {st} {n:22s} maxerr={mx:.3e} matched={matched:.5f} |ref|max={b.abs().max().item():.3e}")
    return ok

if __name__=="__main__":
    B,S=32,4096
    inp=gen(B,S)
    ref=reference.run(**inp)
    for n,r in zip(names,ref):
        print(f"{n:22s} shape={tuple(r.shape)} absmax={r.abs().max().item():.4e} absmean={r.abs().mean().item():.4e}")

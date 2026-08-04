from lt import *
import torch, reference
import json
wls=[json.loads(l) for l in open('workload.jsonl')]
for w in [wls[8], wls[15], wls[0]]:
    B=w['axes']['batch_size']; S=w['axes']['seq_len']; atol=w['tolerance']['max_atol']; rtol=w['tolerance']['max_rtol']
    H=512;I=2048;N=B*S
    inp=gen(B,S,seed=B*1000+S)
    ref=reference.run(**inp)
    go=inp['grad_output']; norm=inp['normalized']; var=inp['var']; lnw=inp['ln_weight']; eps=inp['eps']
    gelu=inp['gelu_output']; f2w=inp['fc2_weight']
    gn=go*lnw; std=torch.sqrt(var+eps)
    m1=gn.mean(-1,keepdim=True); m2=(gn*norm).mean(-1,keepdim=True)
    gro=((1.0/std)*(gn-m1-norm*m2)).view(N,H)
    def m(a,b):
        e=(a-b).abs(); t=atol+rtol*b.abs(); return (e<=t).float().mean().item()
    # alt formulations of grad_fc2_weight
    a1=gro.t()@gelu.view(N,I)
    a2=(gelu.view(N,I).t()@gro).t().contiguous()
    a3=torch.einsum('nh,ni->hi', gro, gelu.view(N,I))
    print(f"B={B} S={S} N={N}: t()@ ={m(a1,ref[3]):.4f}  transposed={m(a2,ref[3]):.4f} einsum={m(a3,ref[3]):.4f}")
    # split-k manual
    if N>=2048:
        h=N//2
        a4=gro[:h].t()@gelu.view(N,I)[:h] + gro[h:].t()@gelu.view(N,I)[h:]
        print(f"    splitk2={m(a4,ref[3]):.4f}")

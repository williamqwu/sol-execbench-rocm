from lt import *
import torch, reference, json
wls=[json.loads(l) for l in open('workload.jsonl')]
for idx in [15,8,0]:
    w=wls[idx]; B=w['axes']['batch_size']; S=w['axes']['seq_len']
    atol=w['tolerance']['max_atol']; rtol=w['tolerance']['max_rtol']
    H=512;I=2048;N=B*S
    inp=gen(B,S,seed=B*1000+S)
    go=inp['grad_output']; norm=inp['normalized']; var=inp['var']; lnw=inp['ln_weight']; eps=inp['eps']
    f2w=inp['fc2_weight']
    gn=go*lnw; std=torch.sqrt(var+eps)
    m1=gn.mean(-1,keepdim=True); m2=(gn*norm).mean(-1,keepdim=True)
    gro=((1.0/std)*(gn-m1-norm*m2)).view(N,H)
    ref=gro@f2w
    exact=(gro.double()@f2w.double())
    e=(ref.double()-exact).abs(); tol=atol+rtol*ref.abs().double()
    print(f"B={B} S={S}: ggelu GEMM torch_err/tol max={(e/tol).max().item():.2f} matched_if_exact={((e<=tol).float().mean().item()):.4f}")
    # what would an exact-fp64 answer score?
    ex32=exact.float()
    e2=(ex32-ref).abs(); t2=atol+rtol*ref.abs()
    print(f"    fp64-computed vs torch: matched={(e2<=t2).float().mean().item():.4f}")

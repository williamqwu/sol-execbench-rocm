from lt import *
import torch, reference, json
wls=[json.loads(l) for l in open('workload.jsonl')]
for idx in [15,8,0]:
    w=wls[idx]; B=w['axes']['batch_size']; S=w['axes']['seq_len']
    atol=w['tolerance']['max_atol']; rtol=w['tolerance']['max_rtol']
    H=512;I=2048;N=B*S
    inp=gen(B,S,seed=B*1000+S)
    ref=reference.run(**inp)
    go=inp['grad_output']; norm=inp['normalized']
    # exact fp64 vs torch fp32 for grad_ln_weight
    exact=(go.double()*norm.double()).sum(dim=(0,1))
    t32=ref[5]
    e=(t32.double()-exact).abs(); tol=atol+rtol*t32.abs().double()
    print(f"B={B} S={S} N={N} grad_ln_weight: torch_err_vs_exact max={e.max().item():.3e} tol_typ={tol.median().item():.3e} ratio={ (e/tol).max().item():.2f}")
    exact2=go.double().sum(dim=(0,1)); e2=(ref[6].double()-exact2).abs(); tol2=atol+rtol*ref[6].abs().double()
    print(f"    grad_ln_bias: err={e2.max().item():.3e} ratio={(e2/tol2).max().item():.2f}")

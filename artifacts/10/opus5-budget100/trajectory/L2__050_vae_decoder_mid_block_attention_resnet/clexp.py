import torch, sys
import torch.nn.functional as F
sys.path.insert(0,'/var/tmp/solbench/agent/opus5-budget100/L2__050_vae_decoder_mid_block_attention_resnet')
from harness import make, bench, TOL, check
import reference
dev='cuda'

def run_cl(hidden_states, temb, r1n1w,r1n1b,r1c1w,r1c1b,r1tw,r1tb,r1n2w,r1n2b,r1c2w,r1c2b,
        agnw,agnb,qw,qb,kw,kb,vw,vb,ow,ob,
        r2n1w,r2n1b,r2c1w,r2c1b,r2tw,r2tb,r2n2w,r2n2b,r2c2w,r2c2b, eps):
    B,C,H,W = hidden_states.shape
    ng=32; S=H*W; scale=C**-0.5
    CL = torch.channels_last
    x = hidden_states.to(memory_format=CL)
    c1 = r1c1w.to(memory_format=CL); c2 = r1c2w.to(memory_format=CL)
    c3 = r2c1w.to(memory_format=CL); c4 = r2c2w.to(memory_format=CL)
    tp1 = F.linear(F.silu(temb), r1tw, r1tb)
    tp2 = F.linear(F.silu(temb), r2tw, r2tb)
    res1 = x
    h = F.conv2d(F.silu(F.group_norm(x, ng, r1n1w, r1n1b, eps)), c1, r1c1b, padding=1)
    h = h + tp1[:,:,None,None]
    h = F.conv2d(F.silu(F.group_norm(h, ng, r1n2w, r1n2b, eps)), c2, r1c2b, padding=1)
    hs = h + res1
    ar = hs
    h = F.group_norm(hs, ng, agnw, agnb, eps)
    hh = h.contiguous().view(B,C,S).transpose(1,2)
    qkv = F.linear(hh, torch.cat((qw,kw,vw),0), torch.cat((qb,kb,vb),0))
    q,k,v = qkv.split(C, dim=-1)
    s = torch.matmul(q, k.transpose(-2,-1))*scale
    p = F.softmax(s,dim=-1)
    o = torch.matmul(p, v)
    o = F.linear(o, ow, ob)
    o = o.transpose(1,2).reshape(B,C,H,W).to(memory_format=CL)
    hs = o + ar
    res2 = hs
    h = F.conv2d(F.silu(F.group_norm(hs, ng, r2n1w, r2n1b, eps)), c3, r2c1b, padding=1)
    h = h + tp2[:,:,None,None]
    h = F.conv2d(F.silu(F.group_norm(h, ng, r2n2w, r2n2b, eps)), c4, r2c2b, padding=1)
    return (h + res2).contiguous()

for shape in TOL:
    B,H,W = shape
    args = make(B,H,W)
    ref = reference.run(*args)
    o = run_cl(*args)
    atol,rtol = TOL[shape]
    r,mx = check(o, ref, atol, rtol)
    tr = bench(lambda: reference.run(*args)); tm = bench(lambda: run_cl(*args))
    print(f"B{B} {H}x{W}: {'PASS' if r>=0.99 else 'FAIL'} ratio={r:.5f} exact={mx==0} max={mx:.2e} sp={tr/tm:.3f}")

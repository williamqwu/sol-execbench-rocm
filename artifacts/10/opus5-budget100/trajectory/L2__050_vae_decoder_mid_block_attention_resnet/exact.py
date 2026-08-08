"""Search for bit-exact restructurings: must give maxabs==0 on tight shapes."""
import torch, sys, math
import torch.nn.functional as F
sys.path.insert(0,'/var/tmp/solbench/agent/opus5-budget100/L2__050_vae_decoder_mid_block_attention_resnet')
from harness import make, bench
import reference
dev='cuda'

TIGHT=[(1,32,32),(32,32,32),(2,41,41)]

def variant(mode):
    def run(hidden_states, temb, r1n1w,r1n1b,r1c1w,r1c1b,r1tw,r1tb,r1n2w,r1n2b,r1c2w,r1c2b,
            agnw,agnb,qw,qb,kw,kb,vw,vb,ow,ob,
            r2n1w,r2n1b,r2c1w,r2c1b,r2tw,r2tb,r2n2w,r2n2b,r2c2w,r2c2b, eps):
        B,C,Hh,Ww = hidden_states.shape
        ng=32; S=Hh*Ww; scale=C**-0.5
        st = F.silu(temb)
        if 'temb_once' in mode:
            st2 = st
        else:
            st2 = F.silu(temb)
        if 'temb_cat' in mode:
            tp_all = F.linear(st, torch.cat((r1tw,r2tw),0), torch.cat((r1tb,r2tb),0))
            tp1, tp2 = tp_all.split(C, dim=-1)
        else:
            tp1 = F.linear(st, r1tw, r1tb); tp2 = F.linear(st2, r2tw, r2tb)

        res1 = hidden_states
        h = F.conv2d(F.silu(F.group_norm(hidden_states, ng, r1n1w, r1n1b, eps)), r1c1w, r1c1b, padding=1)
        h = h + tp1[:,:,None,None]
        h = F.conv2d(F.silu(F.group_norm(h, ng, r1n2w, r1n2b, eps)), r1c2w, r1c2b, padding=1)
        hs = h + res1

        ar = hs
        h = F.group_norm(hs, ng, agnw, agnb, eps)
        h = h.view(B,C,S).transpose(1,2)
        if 'qkv_cat' in mode:
            qkv = F.linear(h, torch.cat((qw,kw,vw),0), torch.cat((qb,kb,vb),0))
            q,k,v = qkv.split(C, dim=-1)
        else:
            q=F.linear(h,qw,qb); k=F.linear(h,kw,kb); v=F.linear(h,vw,vb)
        if 'bmm' in mode:
            s = torch.bmm(q, k.transpose(-2,-1))*scale
        else:
            s = torch.matmul(q, k.transpose(-2,-1))*scale
        p = F.softmax(s,dim=-1)
        h = torch.matmul(p, v)
        h = F.linear(h, ow, ob)
        if 'outT' in mode:
            h = h.transpose(1,2).reshape(B,C,Hh,Ww)
        else:
            h = h.transpose(1,2).view(B,C,Hh,Ww)
        hs = h + ar

        res2 = hs
        h = F.conv2d(F.silu(F.group_norm(hs, ng, r2n1w, r2n1b, eps)), r2c1w, r2c1b, padding=1)
        h = h + tp2[:,:,None,None]
        h = F.conv2d(F.silu(F.group_norm(h, ng, r2n2w, r2n2b, eps)), r2c2w, r2c2b, padding=1)
        return h + res2
    return run

MODES = ['base','temb_once','temb_once+temb_cat','temb_once+qkv_cat','temb_once+qkv_cat+bmm','temb_once+qkv_cat+outT']
for shape in TIGHT:
    B,Hh,Ww = shape
    args = make(B,Hh,Ww)
    ref = reference.run(*args)
    tr = bench(lambda: reference.run(*args))
    print(f"--- {shape} ref={tr:.4f}ms")
    for m in MODES:
        f = variant(m)
        try:
            o = f(*args)
            mx = (o-ref).abs().max().item()
            t = bench(lambda: f(*args))
            print(f"   {m:26s} maxabs={mx:.3e} {'EXACT' if mx==0 else 'DIFF '} t={t:.4f} sp={tr/t:.3f}")
        except Exception as e:
            print(f"   {m:26s} ERR {type(e).__name__}: {str(e)[:70]}")

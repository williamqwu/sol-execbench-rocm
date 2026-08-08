import torch, sys
import torch.nn.functional as F
sys.path.insert(0,'/var/tmp/solbench/agent/opus5-budget100/L2__050_vae_decoder_mid_block_attention_resnet')
from harness import make, ORDER, TOL, check
import reference
dev='cuda'

def body(hidden_states, temb, r1n1w,r1n1b,r1c1w,r1c1b,r1tw,r1tb,r1n2w,r1n2b,r1c2w,r1c2b,
          agnw,agnb,qw,qb,kw,kb,vw,vb,ow,ob,
          r2n1w,r2n1b,r2c1w,r2c1b,r2tw,r2tb,r2n2w,r2n2b,r2c2w,r2c2b, eps, mode='ref'):
    batch, channels, height, width = hidden_states.shape
    ng=32
    def gn(x,w,b):
        if mode=='gn_manual':
            B,C,H,W=x.shape
            xg = x.view(B,ng,-1)
            m = xg.mean(-1,keepdim=True); v = xg.var(-1,unbiased=False,keepdim=True)
            y = (xg-m)*torch.rsqrt(v+eps)
            return y.view(B,C,H,W)*w[None,:,None,None]+b[None,:,None,None]
        return F.group_norm(x, ng, w, b, eps)
    def silu(x):
        if mode=='silu_manual': return x*torch.sigmoid(x)
        return F.silu(x)
    conv = lambda x,w,b: F.conv2d(x,w,b,padding=1)

    residual1 = hidden_states
    h = conv(silu(gn(hidden_states, r1n1w, r1n1b)), r1c1w, r1c1b)
    tp = F.linear(F.silu(temb), r1tw, r1tb)
    h = h + tp[:,:,None,None]
    h = conv(silu(gn(h, r1n2w, r1n2b)), r1c2w, r1c2b)
    hs = h + residual1

    ar = hs
    h = gn(hs, agnw, agnb)
    S = height*width
    h = h.view(batch, channels, S).transpose(1,2)
    if mode=='qkv_fused':
        W3 = torch.cat([qw,kw,vw],0); B3 = torch.cat([qb,kb,vb],0)
        qkv = F.linear(h, W3, B3)
        q,k,v = qkv.split(channels, dim=-1)
    else:
        q = F.linear(h, qw, qb); k = F.linear(h, kw, kb); v = F.linear(h, vw, vb)
    if mode=='sdpa':
        h = F.scaled_dot_product_attention(q.unsqueeze(1),k.unsqueeze(1),v.unsqueeze(1)).squeeze(1)
    else:
        s = torch.matmul(q, k.transpose(-2,-1)) * (channels**-0.5)
        if mode=='softmax_manual':
            s = s - s.amax(-1,keepdim=True); e=torch.exp(s); p = e/e.sum(-1,keepdim=True)
        else:
            p = F.softmax(s, dim=-1)
        h = torch.matmul(p, v)
    h = F.linear(h, ow, ob)
    h = h.transpose(1,2).view(batch, channels, height, width)
    hs = h + ar

    residual2 = hs
    h = conv(silu(gn(hs, r2n1w, r2n1b)), r2c1w, r2c1b)
    tp = F.linear(F.silu(temb), r2tw, r2tb)
    h = h + tp[:,:,None,None]
    h = conv(silu(gn(h, r2n2w, r2n2b)), r2c2w, r2c2b)
    return h + residual2

for shape in [(1,32,32),(32,32,32),(2,41,41),(1,61,61),(16,32,32)]:
    B,H,W = shape
    args = make(B,H,W)
    ref = reference.run(*args)
    atol,rtol = TOL[shape]
    print(shape, 'refmax', ref.abs().max().item())
    for mode in ['ref','gn_manual','silu_manual','qkv_fused','sdpa','softmax_manual']:
        try:
            o = body(*args, mode=mode)
            r,mx = check(o, ref, atol, rtol)
            print(f"   {mode:16s} ratio={r:.5f} max={mx:.3e} {'PASS' if r>=0.99 else 'FAIL'}")
        except Exception as e:
            print(f"   {mode:16s} ERR {type(e).__name__}: {str(e)[:80]}")

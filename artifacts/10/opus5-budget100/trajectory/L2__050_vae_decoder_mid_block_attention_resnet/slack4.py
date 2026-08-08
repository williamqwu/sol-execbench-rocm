import torch, sys
import torch.nn.functional as F
sys.path.insert(0,'/var/tmp/solbench/agent/opus5-budget100/L2__050_vae_decoder_mid_block_attention_resnet')
from harness import make, TOL, check
import reference
dev='cuda'
ng=32

def body(hidden_states, temb, r1n1w,r1n1b,r1c1w,r1c1b,r1tw,r1tb,r1n2w,r1n2b,r1c2w,r1c2b,
          agnw,agnb,qw,qb,kw,kb,vw,vb,ow,ob,
          r2n1w,r2n1b,r2c1w,r2c1b,r2tw,r2tb,r2n2w,r2n2b,r2c2w,r2c2b, eps, mode='ref'):
    batch, channels, height, width = hidden_states.shape
    def gn(x,w,b):
        if mode=='gn_f64':
            B,C,H,W=x.shape
            xg = x.double().reshape(B,ng,-1)
            m = xg.mean(-1,keepdim=True); v = (xg*xg).mean(-1,keepdim=True)-m*m
            y = (xg-m)*torch.rsqrt(v+eps)
            return (y.reshape(B,C,H,W)*w.double()[None,:,None,None]+b.double()[None,:,None,None]).float()
        return F.group_norm(x, ng, w, b, eps)
    def silu(x):
        if mode=='silu_f64': return (x.double()*torch.sigmoid(x.double())).float()
        return F.silu(x)
    def lin(x,w,b):
        if mode=='lin_f64': return (x.double()@w.double().t()+b.double()).float()
        return F.linear(x,w,b)
    def sm(x):
        if mode=='softmax_f64': return F.softmax(x.double(),dim=-1).float()
        return F.softmax(x,dim=-1)
    def mm(a,b):
        if mode=='mm_f64': return (a.double()@b.double()).float()
        return torch.matmul(a,b)
    def conv(x,w,b):
        if mode=='conv_f64':
            return F.conv2d(x.double(), w.double(), b.double(), padding=1).float()
        if mode=='conv_last_f64':
            return F.conv2d(x,w,b,padding=1)
        return F.conv2d(x,w,b,padding=1)

    residual1 = hidden_states
    h = conv(silu(gn(hidden_states, r1n1w, r1n1b)), r1c1w, r1c1b)
    tp = lin(F.silu(temb), r1tw, r1tb)
    h = h + tp[:,:,None,None]
    h = conv(silu(gn(h, r1n2w, r1n2b)), r1c2w, r1c2b)
    hs = h + residual1

    ar = hs
    h = gn(hs, agnw, agnb)
    S = height*width
    h = h.view(batch, channels, S).transpose(1,2)
    q = lin(h,qw,qb); k = lin(h,kw,kb); v = lin(h,vw,vb)
    s = mm(q, k.transpose(-2,-1)) * (channels**-0.5)
    p = sm(s)
    h = mm(p, v)
    h = lin(h, ow, ob)
    h = h.transpose(1,2).view(batch, channels, height, width)
    hs = h + ar

    residual2 = hs
    h = conv(silu(gn(hs, r2n1w, r2n1b)), r2c1w, r2c1b)
    tp = lin(F.silu(temb), r2tw, r2tb)
    h = h + tp[:,:,None,None]
    h2 = silu(gn(h, r2n2w, r2n2b))
    if mode=='conv_last_f64':
        h = F.conv2d(h2.double(), r2c2w.double(), r2c2b.double(), padding=1).float()
    else:
        h = conv(h2, r2c2w, r2c2b)
    return h + residual2

for shape in [(1,32,32),(32,32,32),(2,41,41)]:
    B,H,W = shape
    args = make(B,H,W)
    ref = reference.run(*args)
    atol,rtol = TOL[shape]
    print(shape)
    for mode in ['ref','gn_f64','silu_f64','lin_f64','softmax_f64','mm_f64','conv_f64','conv_last_f64']:
        try:
            o = body(*args, mode=mode)
            r,mx = check(o, ref, atol, rtol)
            print(f"   {mode:16s} ratio={r:.5f} max={mx:.3e} {'PASS' if r>=0.99 else 'FAIL'}")
        except Exception as e:
            print(f"   {mode:16s} ERR {type(e).__name__}: {str(e)[:100]}")

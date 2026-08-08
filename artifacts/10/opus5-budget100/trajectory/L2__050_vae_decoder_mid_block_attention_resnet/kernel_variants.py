import torch
import torch.nn.functional as F

def _body(conv, hidden_states, temb, r1n1w,r1n1b,r1c1w,r1c1b,r1tw,r1tb,r1n2w,r1n2b,r1c2w,r1c2b,
          agnw,agnb,qw,qb,kw,kb,vw,vb,ow,ob,
          r2n1w,r2n1b,r2c1w,r2c1b,r2tw,r2tb,r2n2w,r2n2b,r2c2w,r2c2b, eps):
    batch, channels, height, width = hidden_states.shape
    ng=32
    residual1 = hidden_states
    h = F.group_norm(hidden_states, ng, r1n1w, r1n1b, eps)
    h = F.silu(h)
    h = conv(h, r1c1w, r1c1b)
    tp = F.linear(F.silu(temb), r1tw, r1tb)
    h = h + tp[:,:,None,None]
    h = F.group_norm(h, ng, r1n2w, r1n2b, eps)
    h = F.silu(h)
    h = conv(h, r1c2w, r1c2b)
    hs = h + residual1

    ar = hs
    h = F.group_norm(hs, ng, agnw, agnb, eps)
    S = height*width
    h = h.view(batch, channels, S).transpose(1,2)
    q = F.linear(h, qw, qb); k = F.linear(h, kw, kb); v = F.linear(h, vw, vb)
    s = torch.matmul(q, k.transpose(-2,-1)) * (channels**-0.5)
    p = F.softmax(s, dim=-1)
    h = torch.matmul(p, v)
    h = F.linear(h, ow, ob)
    h = h.transpose(1,2).view(batch, channels, height, width)
    hs = h + ar

    residual2 = hs
    h = F.group_norm(hs, ng, r2n1w, r2n1b, eps)
    h = F.silu(h)
    h = conv(h, r2c1w, r2c1b)
    tp = F.linear(F.silu(temb), r2tw, r2tb)
    h = h + tp[:,:,None,None]
    h = F.group_norm(h, ng, r2n2w, r2n2b, eps)
    h = F.silu(h)
    h = conv(h, r2c2w, r2c2b)
    return h + residual2

def run_cl(*a):
    def conv(x,w,b):
        return F.conv2d(x.to(memory_format=torch.channels_last), w.to(memory_format=torch.channels_last), b, padding=1).contiguous()
    return _body(conv, *a)

def run_unfold(*a):
    def conv(x,w,b):
        B,C,H,W = x.shape
        u = F.unfold(x, 3, padding=1)  # B, C*9, H*W
        O = w.shape[0]
        return (w.view(O,-1) @ u + b[None,:,None]).view(B,O,H,W)
    return _body(conv, *a)

def _split2(t):
    h = t.half()
    l = ((t - h.float())*2048.0).half()
    return h,l

def run_fp16x2(*a):
    def conv(x,w,b):
        B,C,H,W = x.shape
        O = w.shape[0]
        u = F.unfold(x,3,padding=1)   # B, C*9, HW
        wm = w.view(O,-1)
        xh,xl = _split2(u); wh,wl = _split2(wm)
        acc = (wh.float() @ xh.float())
        return None
    return _body(conv, *a)

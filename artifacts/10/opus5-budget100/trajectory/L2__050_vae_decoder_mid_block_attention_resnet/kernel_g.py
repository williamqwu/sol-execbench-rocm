import torch
import torch.nn.functional as F

_CACHE = {}


def _body(hidden_states, temb,
    r1n1w, r1n1b, r1c1w, r1c1b, r1tw, r1tb, r1n2w, r1n2b, r1c2w, r1c2b,
    agnw, agnb, qw, qb, kw, kb, vw, vb, ow, ob,
    r2n1w, r2n1b, r2c1w, r2c1b, r2tw, r2tb, r2n2w, r2n2b, r2c2w, r2c2b, eps):
    batch, channels, height, width = hidden_states.shape
    ng = 32
    scale = channels ** -0.5
    res1 = hidden_states
    h = F.group_norm(hidden_states, ng, r1n1w, r1n1b, eps)
    h = F.silu(h)
    h = F.conv2d(h, r1c1w, r1c1b, padding=1)
    tp = F.linear(F.silu(temb), r1tw, r1tb)
    h = h + tp[:, :, None, None]
    h = F.group_norm(h, ng, r1n2w, r1n2b, eps)
    h = F.silu(h)
    h = F.conv2d(h, r1c2w, r1c2b, padding=1)
    hs = h + res1

    ar = hs
    h = F.group_norm(hs, ng, agnw, agnb, eps)
    S = height * width
    h = h.view(batch, channels, S).transpose(1, 2)
    q = F.linear(h, qw, qb); k = F.linear(h, kw, kb); v = F.linear(h, vw, vb)
    s = torch.matmul(q, k.transpose(-2, -1)) * scale
    p = F.softmax(s, dim=-1)
    h = torch.matmul(p, v)
    h = F.linear(h, ow, ob)
    h = h.transpose(1, 2).view(batch, channels, height, width)
    hs = h + ar

    res2 = hs
    h = F.group_norm(hs, ng, r2n1w, r2n1b, eps)
    h = F.silu(h)
    h = F.conv2d(h, r2c1w, r2c1b, padding=1)
    tp = F.linear(F.silu(temb), r2tw, r2tb)
    h = h + tp[:, :, None, None]
    h = F.group_norm(h, ng, r2n2w, r2n2b, eps)
    h = F.silu(h)
    h = F.conv2d(h, r2c2w, r2c2b, padding=1)
    return h + res2


@torch.no_grad()
def run(*args):
    tensors = [a for a in args if isinstance(a, torch.Tensor)]
    eps = args[-1]
    key = (tuple((tuple(t.shape), t.dtype, t.stride()) for t in tensors),
           tensors[0].device, eps)
    ent = _CACHE.get(key)
    if ent is None:
        static = [t.clone() for t in tensors]
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            for _ in range(3):
                _body(*static, eps)
        torch.cuda.current_stream().wait_stream(s)
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            out = _body(*static, eps)
        ent = (static, g, out)
        _CACHE[key] = ent
    static, g, out = ent
    torch._foreach_copy_(static, tensors)
    g.replay()
    return out.clone()

import torch
import torch.nn.functional as F

_CACHE = {}
_ZERO = {}


def _zero(dev):
    z = _ZERO.get(dev)
    if z is None:
        z = torch.zeros(1, device=dev)
        _ZERO[dev] = z
    return z


def _body(hidden_states, temb,
    r1n1w, r1n1b, r1c1w, r1c1b, r1tw, r1tb, r1n2w, r1n2b, r1c2w, r1c2b,
    agnw, agnb, qw, qb, kw, kb, vw, vb, ow, ob,
    r2n1w, r2n1b, r2c1w, r2c1b, r2tw, r2tb, r2n2w, r2n2b, r2c2w, r2c2b,
    eps, w_qkv, b_qkv, w_temb, b_temb):
    """Bit-identical to the reference, with the two independent time-embedding
    projections and the three QKV projections each issued as one GEMM."""
    batch, channels, height, width = hidden_states.shape
    ng = 32
    scale = channels ** -0.5

    tp_all = F.linear(F.silu(temb), w_temb, b_temb)
    tp1, tp2 = tp_all.split(channels, dim=-1)

    res1 = hidden_states
    h = F.group_norm(hidden_states, ng, r1n1w, r1n1b, eps)
    h = F.silu(h)
    h = F.conv2d(h, r1c1w, r1c1b, padding=1)
    h = h + tp1[:, :, None, None]
    h = F.group_norm(h, ng, r1n2w, r1n2b, eps)
    h = F.silu(h)
    h = F.conv2d(h, r1c2w, r1c2b, padding=1)
    hs = h + res1

    ar = hs
    h = F.group_norm(hs, ng, agnw, agnb, eps)
    S = height * width
    h = h.view(batch, channels, S).transpose(1, 2)
    qkv = F.linear(h, w_qkv, b_qkv)
    q, k, v = qkv.split(channels, dim=-1)
    # alpha folds the 1/sqrt(d) into the GEMM epilogue: bit-identical to a
    # separate multiply, but saves a full S x S read/write pass.
    s = torch.baddbmm(_zero(q.device), q, k.transpose(-2, -1), beta=0.0, alpha=scale)
    p = F.softmax(s, dim=-1)
    h = torch.matmul(p, v)
    h = F.linear(h, ow, ob)
    h = h.transpose(1, 2).view(batch, channels, height, width)
    hs = h + ar

    res2 = hs
    h = F.group_norm(hs, ng, r2n1w, r2n1b, eps)
    h = F.silu(h)
    h = F.conv2d(h, r2c1w, r2c1b, padding=1)
    h = h + tp2[:, :, None, None]
    h = F.group_norm(h, ng, r2n2w, r2n2b, eps)
    h = F.silu(h)
    h = F.conv2d(h, r2c2w, r2c2b, padding=1)
    return h + res2


def _fused(t):
    """Concatenated weights for the two batched projections. Concatenation
    changes neither the values nor the per-output-row reduction order, so the
    GEMM results are bit-identical to the separate projections."""
    return (torch.cat((t[14], t[16], t[18]), 0), torch.cat((t[15], t[17], t[19]), 0),
            torch.cat((t[6], t[26]), 0), torch.cat((t[7], t[27]), 0))


def _time(fn, rounds=5, iters=10):
    best = float('inf')
    for _ in range(rounds):
        for _ in range(3):
            fn()
        torch.cuda.synchronize()
        st = torch.cuda.Event(True)
        en = torch.cuda.Event(True)
        st.record()
        for _ in range(iters):
            fn()
        en.record()
        torch.cuda.synchronize()
        best = min(best, st.elapsed_time(en) / iters)
    return best


def _build(tensors, eps):
    def eager_path():
        return _body(*tensors, eps, *_fused(tensors))

    graph = None
    try:
        static = [t.clone() for t in tensors]
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            for _ in range(3):
                _body(*static, eps, *_fused(static))
        torch.cuda.current_stream().wait_stream(s)
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            out = _body(*static, eps, *_fused(static))

        def graph_path():
            torch._foreach_copy_(static, tensors)
            g.replay()
            return out.clone()

        # Interleave the two measurements so slow drift cannot bias the choice.
        t_e = _time(eager_path)
        t_g = _time(graph_path)
        t_e = min(t_e, _time(eager_path))
        t_g = min(t_g, _time(graph_path))
        if t_g < 0.98 * t_e:
            graph = (static, g, out)
    except Exception:
        graph = None
    return graph


@torch.no_grad()
def run(*args):
    tensors = list(args[:32])
    eps = args[32]
    key = (tensors[0].shape, tensors[1].shape, tensors[0].device, eps)
    if key in _CACHE:
        ent = _CACHE[key]
    else:
        ent = _build(tensors, eps)
        _CACHE[key] = ent
    if ent is not None:
        static, g, out = ent
        torch._foreach_copy_(static, tensors)
        g.replay()
        return out.clone()
    return _body(*tensors, eps, *_fused(tensors))

import os
import torch
import torch.nn.functional as F

import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ext_src import DECL, SRC

_EXT = None


def _ext():
    global _EXT
    if _EXT is None:
        from torch.utils.cpp_extension import load_inline
        os.environ.setdefault('TORCH_EXTENSIONS_DIR', '/var/tmp/solbench/te_kernel')
        _EXT = load_inline(
            name='vae_mid_ext', cpp_sources=DECL, cuda_sources=SRC,
            functions=['gn_silu', 'gn_plain', 'bias_add_gn_silu', 'add_bias', 'add_res'],
            verbose=False, extra_cuda_cflags=['-O3'])
    return _EXT


def _body(hidden_states, temb,
    r1n1w, r1n1b, r1c1w, r1c1b, r1tw, r1tb, r1n2w, r1n2b, r1c2w, r1c2b,
    agnw, agnb, qw, qb, kw, kb, vw, vb, ow, ob,
    r2n1w, r2n1b, r2c1w, r2c1b, r2tw, r2tb, r2n2w, r2n2b, r2c2w, r2c2b,
    eps, w_qkv, b_qkv, w_temb, b_temb):
    e = _ext()
    batch, channels, height, width = hidden_states.shape
    ng = 32
    scale = channels ** -0.5

    tp_all = F.linear(F.silu(temb), w_temb, b_temb)
    tp1, tp2 = tp_all.split(channels, dim=-1)
    tp1 = tp1.contiguous(); tp2 = tp2.contiguous()

    res1 = hidden_states
    h = e.gn_silu(hidden_states, r1n1w, r1n1b, ng, eps)
    h = F.conv2d(h, r1c1w, r1c1b, padding=1)
    h = e.bias_add_gn_silu(h, tp1, r1n2w, r1n2b, ng, eps)
    h = F.conv2d(h, r1c2w, r1c2b, padding=1)
    hs = e.add_res(h, res1)

    ar = hs
    h = e.gn_plain(hs, agnw, agnb, ng, eps)
    S = height * width
    h = h.view(batch, channels, S).transpose(1, 2)
    qkv = F.linear(h, w_qkv, b_qkv)
    q, k, v = qkv.split(channels, dim=-1)
    s = torch.matmul(q, k.transpose(-2, -1)) * scale
    p = F.softmax(s, dim=-1)
    h = torch.matmul(p, v)
    h = F.linear(h, ow, ob)
    h = h.transpose(1, 2).view(batch, channels, height, width)
    hs = e.add_res(h.contiguous(), ar)

    res2 = hs
    h = e.gn_silu(hs, r2n1w, r2n1b, ng, eps)
    h = F.conv2d(h, r2c1w, r2c1b, padding=1)
    h = e.bias_add_gn_silu(h, tp2, r2n2w, r2n2b, ng, eps)
    h = F.conv2d(h, r2c2w, r2c2b, padding=1)
    return e.add_res(h, res2)


def _fused(t):
    return (torch.cat((t[14], t[16], t[18]), 0), torch.cat((t[15], t[17], t[19]), 0),
            torch.cat((t[6], t[26]), 0), torch.cat((t[7], t[27]), 0))


@torch.no_grad()
def run(*args):
    tensors = list(args[:32])
    eps = args[32]
    return _body(*tensors, eps, *_fused(tensors))

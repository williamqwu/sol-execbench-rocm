import torch
import os
os.environ["TORCHINDUCTOR_COORDINATE_DESCENT_TUNING"] = "1"
os.environ["TORCHINDUCTOR_AGGRESSIVE_FUSION"] = "1"


def _impl(hidden_states, qw, kw, vw, qn, kn, eps):
    bs, sl, _ = hidden_states.shape
    nh, hd = 8, 128
    q = torch.matmul(hidden_states, qw.t()).view(bs, sl, nh, hd)
    k = torch.matmul(hidden_states, kw.t()).view(bs, sl, nh, hd)
    v = torch.matmul(hidden_states, vw.t()).view(bs, sl, nh, hd)
    qv = q.pow(2).mean(dim=-1, keepdim=True)
    qs = (q / torch.sqrt(qv + eps)) * qn
    kv = k.pow(2).mean(dim=-1, keepdim=True)
    ks = (k / torch.sqrt(kv + eps)) * kn
    return qs, ks, v


_compiled = torch.compile(_impl, dynamic=True, mode="default")


@torch.no_grad()
def run(hidden_states, qw, kw, vw, qn, kn, eps):
    return _compiled(hidden_states, qw, kw, vw, qn, kn, eps)

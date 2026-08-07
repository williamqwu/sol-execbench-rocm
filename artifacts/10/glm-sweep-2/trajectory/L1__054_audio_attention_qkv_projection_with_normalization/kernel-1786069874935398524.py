import torch


def _impl(hidden_states, qw, kw, vw, qn, kn, eps):
    bs, sl, _ = hidden_states.shape
    nh, hd = 8, 128
    # HD as constexpr for the reduction
    q = torch.matmul(hidden_states, qw.t()).view(bs, sl, nh, hd)
    k = torch.matmul(hidden_states, kw.t()).view(bs, sl, nh, hd)
    v = torch.matmul(hidden_states, vw.t()).view(bs, sl, nh, hd)
    # Reformulate RMSNorm as x * rsqrt(mean(x^2)+eps) * w: one fused mul chain.
    q_mean = q.float().pow(2).mean(dim=-1, keepdim=True)
    qs = (q * torch.rsqrt(q_mean + eps)) * qn
    k_mean = k.float().pow(2).mean(dim=-1, keepdim=True)
    ks = (k * torch.rsqrt(k_mean + eps)) * kn
    return qs, ks, v


_compiled = torch.compile(_impl, dynamic=True, mode="default")


@torch.no_grad()
def run(hidden_states, qw, kw, vw, qn, kn, eps):
    return _compiled(hidden_states, qw, kw, vw, qn, kn, eps)

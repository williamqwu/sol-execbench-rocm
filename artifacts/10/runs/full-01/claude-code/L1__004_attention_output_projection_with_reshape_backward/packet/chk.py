import torch

def tol_ok(out, ref, atol, rtol=0.0078125, ratio=0.99):
    """Mirror the harness rule: elementwise |a-b| <= atol + rtol*|ref|, >= ratio matched."""
    a = out.float(); b = ref.float()
    d = (a - b).abs()
    lim = atol + rtol * b.abs()
    m = (d <= lim) | (torch.isnan(a) & torch.isnan(b))
    return m.float().mean().item(), d.max().item()

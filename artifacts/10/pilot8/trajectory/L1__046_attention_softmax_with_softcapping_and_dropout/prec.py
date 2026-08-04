import torch
x = (torch.randn(4, 8, 256, 256, device='cuda', dtype=torch.bfloat16) * 12.0)
s = x / 30.0
print("scaled dtype", s.dtype)
c = torch.tanh(s)
sc = c * 30.0
print("softcapped dtype", sc.dtype)
xf = x.float()


def rb(t):
    return t.bfloat16().float()


e = rb(rb(rb(xf / 30.0).tanh()) * 30.0)
print("emul-bf16 vs ref softcapped maxabs:", (e - sc.float()).abs().max().item())
p = torch.tanh(xf / 30.0) * 30.0
print("pure-fp32 vs ref softcapped maxabs:", (p - sc.float()).abs().max().item())

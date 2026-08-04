import torch
x = torch.randn(4096, device='cuda') * 100
a = x / 448.0
b = x * (1.0 / 448.0)
exact = (x.double() / 448.0).float()
print("torch x/448 == x*(1/448):", torch.equal(a, b))
print("torch x/448 == correctly rounded:", torch.equal(a, exact))
print("torch x*(1/448) == correctly rounded:", torch.equal(b, exact))
# cpu
xc = x.cpu()
print("cpu x/448 == correctly rounded:", torch.equal(xc / 448.0, (xc.double() / 448.0).float()))

# and the divide-by-tensor case (q = x / s)
s = torch.rand(4096, device='cuda') + 0.1
p = x / s
pe = (x.double() / s.double()).float()
print("torch tensor-div correctly rounded:", torch.equal(p, pe),
      (p - pe).abs().max().item())
print("torch tensor-div == x*recip:", torch.equal(p, x * torch.reciprocal(s)))

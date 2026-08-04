import torch
dev='cuda:0'
torch.manual_seed(0)
x=torch.rand(1<<22,device=dev)*100
print("torch: x/448.0 == x*(1.0/448.0)  ->", torch.equal(x/448.0, x*(1.0/448.0)))
print("torch: x/448.0 == x/tensor(448)  ->", torch.equal(x/448.0, x/torch.tensor(448.0,device=dev)))
# how often do the two differ at all?
d=(x/torch.tensor(448.0,device=dev))
print("frac where truediv != recipmul:", (d.view(torch.int32)!=(x*(1.0/448.0)).view(torch.int32)).float().mean().item())

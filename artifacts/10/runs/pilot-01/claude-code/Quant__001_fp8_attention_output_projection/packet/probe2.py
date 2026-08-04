import torch, triton, triton.language as tl
dev='cuda:0'

# --- A: does triton f32->fp8e4nv cast match torch .to(float8_e4m3fn) bitwise?
@triton.jit
def cast_k(x_ptr, o_ptr, n, BLK: tl.constexpr):
    off = tl.program_id(0)*BLK + tl.arange(0,BLK)
    m = off < n
    x = tl.load(x_ptr+off, mask=m, other=0.)
    tl.store(o_ptr+off, x.to(tl.float8e4nv), mask=m)

torch.manual_seed(0)
n = 1<<22
x = (torch.rand(n, device=dev)*2-1)*448.0
# also include values near ties
x = torch.cat([x, torch.linspace(-448,448,steps=1<<20,device=dev)])
n = x.numel()
o = torch.empty(n, device=dev, dtype=torch.float8_e4m3fn)
cast_k[(triton.cdiv(n,1024),)](x,o,n,1024)
ref = x.to(torch.float8_e4m3fn)
diff = (o.view(torch.uint8) != ref.view(torch.uint8))
print("cast mismatches:", diff.sum().item(), "/", n)
if diff.any():
    idx = diff.nonzero()[:10,0]
    print(x[idx], o[idx].float(), ref[idx].float())

# --- B: division rounding: triton a/b vs torch a/b
@triton.jit
def div_k(a_ptr,b_ptr,o_ptr,n,BLK: tl.constexpr):
    off = tl.program_id(0)*BLK+tl.arange(0,BLK)
    m=off<n
    a=tl.load(a_ptr+off,mask=m); b=tl.load(b_ptr+off,mask=m,other=1.)
    tl.store(o_ptr+off, a/b, mask=m)
n2=1<<22
a=(torch.rand(n2,device=dev)*2-1)*10
b=torch.rand(n2,device=dev)*0.1+1e-3
o2=torch.empty(n2,device=dev)
div_k[(triton.cdiv(n2,1024),)](a,b,o2,n2,1024)
r2=a/b
print("div mismatches:", (o2.view(torch.int32)!=r2.view(torch.int32)).sum().item(), "/", n2)

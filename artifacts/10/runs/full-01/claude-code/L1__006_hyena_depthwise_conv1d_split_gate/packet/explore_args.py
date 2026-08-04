"""Does cutting kernel arg count / using constexpr shapes reduce Triton dispatch cost?"""
import time
import torch, triton
import triton.language as tl

def cpu(fn, iters=3000):
    for _ in range(200): fn()
    torch.cuda.synchronize()
    t = time.perf_counter()
    for _ in range(iters): fn()
    dt = (time.perf_counter()-t)/iters*1e6
    torch.cuda.synchronize()
    return dt

@triton.jit
def k1(a, b, c, d, e, f, g, h, i, BLOCK: tl.constexpr):
    pass

@triton.jit
def k2(a, b, c, d, BLOCK: tl.constexpr):
    pass

@triton.jit
def k3(a, b, c, d, S: tl.constexpr, B: tl.constexpr, BLOCK: tl.constexpr):
    pass

@triton.jit
def k4(a, BLOCK: tl.constexpr):
    pass

x = torch.empty(1024, device="cuda")
G = (4, 256, 1)
print("noop dispatch cost by arg count (grid=(4,256,1)):")
print(f"  9 int/ptr args : {cpu(lambda: k1[G](x,x,x,x,512,131072,131072,393216,0,BLOCK=256)):7.2f}us")
print(f"  4 ptr args     : {cpu(lambda: k2[G](x,x,x,x,BLOCK=256)):7.2f}us")
print(f"  4 ptr+2 cexpr  : {cpu(lambda: k3[G](x,x,x,x,S=512,B=1,BLOCK=256)):7.2f}us")
print(f"  1 arg          : {cpu(lambda: k4[G](x,BLOCK=256)):7.2f}us")

# real kernel, current form
import kernel as K
B,S,D = 1,512,256
u = torch.randn((B,768,S), device="cuda"); w = torch.randn((768,1,3), device="cuda"); bi = torch.randn((768,), device="cuda")
DS=D*S; out = torch.empty((3,B,D,S), device="cuda")
blk,nw,nsb,even = K._cfg(S)
print()
print(f"  real kernel launch (9 args): {cpu(lambda: K._hyena_fused[(nsb,D,B)](u,w,bi,out,S,DS,B*DS,3*DS,0,BLOCK_S=blk,EVEN=even,num_warps=nw,num_stages=1)):7.2f}us")
print(f"  full K.run()               : {cpu(lambda: K.run(u,w,bi)):7.2f}us")

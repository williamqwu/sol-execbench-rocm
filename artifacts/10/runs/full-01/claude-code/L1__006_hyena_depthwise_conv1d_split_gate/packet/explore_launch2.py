"""Use the supported ck[grid] runner; measure overhead vs the JIT dispatch path."""
import time
import torch, triton
import kernel as K

def cpu_only(fn, iters=3000):
    for _ in range(100): fn()
    torch.cuda.synchronize()
    t = time.perf_counter()
    for _ in range(iters): fn()
    dt = (time.perf_counter()-t)/iters*1e6
    torch.cuda.synchronize()
    return dt

def wall(fn, iters=3000):
    for _ in range(100): fn()
    torch.cuda.synchronize()
    t = time.perf_counter()
    for _ in range(iters): fn()
    torch.cuda.synchronize()
    return (time.perf_counter()-t)/iters*1e6

B,S,D = 1,512,256
u = torch.randn((B,768,S), device="cuda"); w = torch.randn((768,1,3), device="cuda"); bi = torch.randn((768,), device="cuda")
blk,nw,nsb,even = K._cfg(S); DS = D*S
out = torch.empty((3,B,D,S), device="cuda")

print(f"{'variant':44s} {'cpu_us':>9s} {'wall_us':>9s}")
print(f"{'current run()':44s} {cpu_only(lambda: K.run(u,w,bi)):9.2f} {wall(lambda: K.run(u,w,bi)):9.2f}")

ck = K._hyena_fused.warmup(u,w,bi,out,S,DS,B*DS,3*DS,0, BLOCK_S=blk,EVEN=even,num_warps=nw,num_stages=1, grid=(nsb,D,B))
ck._init_handles()
runner = ck[(nsb,D,B)]

def viack():
    o = torch.empty((3,B,D,S), device="cuda")
    runner(u,w,bi,o,S,DS,B*DS,3*DS,0)
    return o[0],o[1],o[2]

r = viack(); torch.cuda.synchronize()
ref = K.run(u,w,bi)
print("matches:", all(torch.equal(a,b) for a,b in zip(r,ref)))
print(f"{'ck[grid] runner + alloc':44s} {cpu_only(viack):9.2f} {wall(viack):9.2f}")

def ck_noalloc():
    runner(u,w,bi,out,S,DS,B*DS,3*DS,0)
print(f"{'ck[grid] runner, no alloc':44s} {cpu_only(ck_noalloc):9.2f} {wall(ck_noalloc):9.2f}")

from triton.runtime import driver
from triton import knobs
dev = driver.active.get_current_device()
stream = driver.active.get_current_stream(dev)
import inspect
try:
    print("launch_metadata sig:", inspect.signature(ck.launch_metadata))
except Exception as e: print(e)

print(f"{'torch.empty (3,B,D,S)':44s} {cpu_only(lambda: torch.empty((3,B,D,S),device='cuda')):9.2f}")
u3 = torch.empty((3,B,D,S), device="cuda")
print(f"{'3 slices of existing':44s} {cpu_only(lambda: (u3[0],u3[1],u3[2])):9.2f}")
print(f"{'unbind(0)':44s} {cpu_only(lambda: u3.unbind(0)):9.2f}")
print(f"{'torch.empty 3 separate':44s} {cpu_only(lambda: [torch.empty((B,D,S),device='cuda') for _ in range(3)]):9.2f}")

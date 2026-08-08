import torch, time
torch.backends.cuda.matmul.allow_tf32 = True
dev='cuda'
H=2560
shapes=[(16,512),(4,128),(8,1024),(1,1571),(4,1024),(2,2053),(8,997),(16,256),(64,128),(32,256),(8,512),(1,1024),(16,128),(2,293),(1,2048),(1,256)]
w=torch.randn(H,H,device=dev,dtype=torch.bfloat16)*0.02
wt=w.t().contiguous()

def bench(fn,*a,it=50,wu=10):
    for _ in range(wu): fn(*a)
    torch.cuda.synchronize()
    t=time.perf_counter()
    for _ in range(it): fn(*a)
    torch.cuda.synchronize()
    return (time.perf_counter()-t)/it*1e3

print(f"{'shape':>14} {'M':>6} {'ref':>8} {'mmonly':>8} {'add':>8} {'addmm':>8} {'mm_wt':>8}")
for b,s in shapes:
    M=b*s
    x=torch.randn(b,s,H,device=dev,dtype=torch.bfloat16)*0.1
    r=torch.randn(b,s,H,device=dev,dtype=torch.bfloat16)*0.1
    x2=x.view(M,H); r2=r.view(M,H)
    ref=bench(lambda: torch.matmul(x,w.t())+r)
    mm=bench(lambda: torch.matmul(x,w.t()))
    p=torch.matmul(x2,w.t())
    add=bench(lambda: p+r2)
    am=bench(lambda: torch.addmm(r2,x2,w.t()))
    mw=bench(lambda: torch.matmul(x2,wt))
    print(f"{str((b,s)):>14} {M:>6} {ref*1e3:8.1f} {mm*1e3:8.1f} {add*1e3:8.1f} {am*1e3:8.1f} {mw*1e3:8.1f}")

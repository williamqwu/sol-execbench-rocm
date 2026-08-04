import torch, time
dev='cuda'
def bench(f,n=20):
    for _ in range(5): f()
    torch.cuda.synchronize(); t=time.perf_counter()
    for _ in range(n): f()
    torch.cuda.synchronize(); return (time.perf_counter()-t)/n*1000
N=4096
a=torch.randn(N,N,device=dev); b=torch.randn(N,N,device=dev)
ref=(a.double()@b.double())
def relerr(r): return ((r.double()-ref).abs()/ref.abs().clamp(min=1e-3)).median().item()
for mode in ['ieee','tf32','bf16','none']:
    try:
        torch.backends.cuda.matmul.fp32_precision=mode
    except Exception as e:
        print(mode,"unsupported",e); continue
    t=bench(lambda:a@b); r=a@b
    print(f"{mode}: {2*N**3/t*1e-9:.0f} TFLOPS medrelerr={relerr(r):.2e}")
torch.backends.cuda.matmul.fp32_precision='ieee'
# bf16 3-split
def split3(x):
    h=x.bfloat16(); r=x-h.float(); m=r.bfloat16(); r2=r-m.float(); l=r2.bfloat16()
    return h,m,l
ah,am,al=split3(a); bh,bm,bl=split3(b)
def mm6():
    o=(ah@bh).float()
    o+= (ah@bm).float(); o+=(am@bh).float()
    o+= (ah@bl).float(); o+=(al@bh).float(); o+=(am@bm).float()
    return o
def mm3():
    return (ah@bh).float()+(ah@bm).float()+(am@bh).float()
print("bf16x3:", bench(mm3), relerr(mm3()))
print("bf16x6:", bench(mm6), relerr(mm6()))
print("fp32 ieee:", bench(lambda:a@b), relerr(a@b))
